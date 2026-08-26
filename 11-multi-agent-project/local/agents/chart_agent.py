"""Chart Agent — turns rows into a Plotly figure spec.

    python agents/chart_agent.py     serves on :9203

Takes a `table` result and returns a figure specification: traces plus layout,
as JSON. Not a PNG — a spec renders in a browser, exports to an image, and can
be restyled without asking the model again.

IT IS ALLOWED TO DECLINE

`chart_type: "none"` with an empty figures list. Some tables should not be
charted — a single row, a list of identifiers, free text. Forcing a chart out of
those produces something worse than no chart, because a rendered panel looks
like a finding.

The two fields have to agree, so a validator checks them. Claiming a chart_type
for a chart that does not exist means a consumer reading only that field renders
an empty panel.
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "10-a2a-protocol"))

from pydantic import BaseModel, Field

from a2a_mini import A2AServer, AgentCard, Artifact, EventQueue, Message, Part, Skill, Task
from langchain_core.prompts import ChatPromptTemplate

from agents_lib import config, llm

CARD = AgentCard(
    name="chart",
    description="Turns tabular data into a Plotly figure specification.",
    url=config.endpoint("chart"),
    capabilities={"streaming": False, "pushNotifications": False},
    skills=[Skill(
        id="chart-table",
        name="Chart a table",
        description=("Produces a Plotly figure spec from columns and rows. "
                     "Invoked by code when a result is table-shaped, not chosen "
                     "by a model."),
        examples=['{"columns": ["sponsor", "trials"], "rows": [["Gilead", 4]]}'])],
)

SYSTEM = """You are a data visualisation expert working in Plotly.

You receive columns and rows. Produce ONE figure that shows what the data
actually says.

Choosing:
  bar        comparing a value across categories
  line       a value over time or an ordered sequence
  scatter    the relationship between two numeric columns
  pie        parts of one whole, and only with few categories
  box        a distribution, and which points are unusual
  histogram  the shape of one numeric column
  none       this table should not be charted

Decline with chart_type "none" and an empty figures list when the table is a
single row, a list of identifiers with no measure, or free text. A chart of
those communicates nothing and looks like a finding.

Write real Plotly traces: each figure has `data` (a list of trace objects with
`type`, `x`, `y`, `name`) and `layout` (with `title`). Put the actual values in.
"""

class Figure(BaseModel):
    """One Plotly figure: traces plus layout."""
    data: list[dict] = Field(description="Plotly traces, each with type, x, y, name")
    layout: dict = Field(description="Plotly layout, with at least a title")


class ChartSpec(BaseModel):
    """What the agent returns.

    `chart_type: "none"` with an empty figures list is a valid, useful answer —
    some tables should not be charted, and a rendered panel of nothing looks
    like a finding.
    """
    chart_type: str = Field(description="bar, line, scatter, pie, box, histogram, or none")
    insight: str = Field(description="One sentence on what the chart shows")
    figures: list[Figure] = Field(default_factory=list)

# Plotly renders a small set of HTML in text fields — titles, axis titles, trace
# text, hovertemplate. So a string in this spec reaches a browser's DOM.
#
# The values being charted come from the data, not from the model: sponsor
# names, site names, drug names. The realistic path is not the model
# misbehaving, it is the model faithfully echoing whatever a name field
# contains into a chart label.
#
# Sanitising here rather than in the renderer means every consumer is covered —
# the browser, an image export, a future one that does not exist yet — without
# each of them having to remember.
#
# Deliberately NOT stripping all HTML: Plotly legitimately supports <br>, <b>,
# <i>, <sub> and <sup>, and analysts rely on them for readable titles. A blanket
# strip breaks real charts, and a guard that breaks real work gets removed.
ALLOWED_TAGS = {"br", "b", "i", "sub", "sup", "em", "strong"}
TAG = re.compile(r"</?([a-zA-Z][a-zA-Z0-9]*)[^>]*>")

# Elements whose CONTENT must go too, not just their tags. Stripping only the
# tags from `<script>alert(1)</script>` leaves `alert(1)` sitting in a chart
# title — inert, but it is attacker-chosen text displayed as if it were data,
# and the next consumer might not be a renderer that treats it as inert.
WITH_CONTENT = re.compile(
    r"<(script|style|iframe|object|embed)\b[^>]*>.*?</\1\s*>",
    re.IGNORECASE | re.DOTALL)


def _clean(text: str) -> str:
    """Drop dangerous elements entirely, then any tag not on the allow list.

    Order matters. Remove the content-bearing elements first — strip their tags
    first and the closing tag no longer pairs with anything, so the content
    survives.
    """
    text = WITH_CONTENT.sub("", text)
    return TAG.sub(lambda m: m.group(0) if m.group(1).lower() in ALLOWED_TAGS else "",
                   text)


def sanitize(value):
    """Walk the spec and clean every string in it, at any depth."""
    if isinstance(value, str):
        return _clean(value)
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    return value


async def execute(task: Task, message: Message, queue: EventQueue) -> None:
    try:
        payload = json.loads(message.text)
    except json.JSONDecodeError:
        server.add_artifact(task, Artifact(name="chart", parts=[Part(
            text=json.dumps({"chart_type": "none", "figures": [],
                             "insight": "input was not JSON"}))]))
        return

    columns, rows = payload.get("columns", []), payload.get("rows", [])
    if not rows:
        server.add_artifact(task, Artifact(name="chart", parts=[Part(
            text=json.dumps({"chart_type": "none", "figures": [],
                             "insight": "no rows to chart"}))]))
        return

    # Enough rows for the model to see the pattern, not so many that the spec
    # becomes a copy of the data. A chart of 400 points does not need 400 in the
    # prompt to be chosen correctly.
    preview = {"columns": columns, "rows": rows[:60], "total_rows": len(rows)}

    # No agent loop here, deliberately. This is one transformation: rows in,
    # figure spec out. There is nothing for a second turn to improve, and a loop
    # would only give the model a chance to second-guess a correct answer.
    #
    # `with_structured_output` still goes through LangChain, so the provider
    # switch works the same way as everywhere else.
    model = llm.chat_model().with_structured_output(ChartSpec)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM), ("human", "{data}")])
    spec = (prompt | model).invoke({"data": json.dumps(preview)})
    result = spec.model_dump()

    figures = [f if isinstance(f, dict) else f.model_dump()
               for f in (result.get("figures") or [])]
    chart_type = result.get("chart_type", "none")

    # The two fields must agree. A chart_type with no figures renders an empty
    # panel; figures with chart_type "none" get dropped by a consumer that reads
    # the type first. Either way the analyst sees something broken.
    if chart_type == "none" or not figures:
        chart_type, figures = "none", []

    out = {"chart_type": chart_type,
           "insight": result.get("insight", ""),
           "figures": sanitize(figures)}
    server.add_artifact(task, Artifact(name="chart",
                                       parts=[Part(text=json.dumps(out))]))


server = A2AServer(CARD, execute)

if __name__ == "__main__":
    print(f"chart agent -> model {llm.describe()}")
    server.run(port=config.PORTS["chart"])
