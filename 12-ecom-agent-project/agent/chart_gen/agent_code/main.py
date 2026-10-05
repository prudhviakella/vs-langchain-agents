"""chart_gen's A2A server — what AgentCore Runtime runs (port 9000, /).

    supervisor ──A2A message/stream {question, columns, rows}──► ChartExecutor
        progress {"type":"progress","agent":"chart_gen","phase":"charting"}  ─► status-update
        result   {chart_type, insight, figures, usage} | {"error"}            ─► artifact "chart_result"

WHAT THIS DOES NOT DO
    A chart failure is returned as {"error"} in a COMPLETED task, not a failed
    one: the supervisor answers without a chart rather than failing the turn.
"""
from __future__ import annotations

import json
import logging
import sys

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, Part, TaskState, TextPart
from a2a.utils import new_task
from bedrock_agentcore.runtime import serve_a2a

from chart_gen.core import generate_chart, parse_request

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
log = logging.getLogger("chart_gen.main")


def _part(text: str) -> Part:
    return Part(root=TextPart(text=text))


class ChartExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task or new_task(context.message)
        if context.current_task is None:
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        try:
            req = parse_request(context.get_user_input())
            await updater.update_status(TaskState.working, message=updater.new_agent_message([_part(json.dumps(
                {"type": "progress", "agent": "chart_gen", "phase": "charting",
                 "detail": f"choosing a chart for {len(req['rows'])} rows"}))]), final=False)
            result = await generate_chart(**req)
        except Exception as exc:                       # malformed request: still a completed task
            log.exception("[chart_gen] request failed")
            result = {"error": f"{type(exc).__name__}: {exc}", "usage": {}}
        await updater.add_artifact([_part(json.dumps(result, default=str))], name="chart_result")
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("charts run to completion")


def agent_card(url: str = "http://localhost:9000/") -> AgentCard:
    return AgentCard(name="chart_gen", version="1.0.0", url=url,
                     description="Turns a table (columns + rows) into 0-3 Plotly figures and one insight.",
                     capabilities=AgentCapabilities(streaming=True),
                     default_input_modes=["text"], default_output_modes=["text"],
                     skills=[AgentSkill(id="chart", name="Chart a table", tags=["chart", "plotly"],
                                        description="Bar, line, box, funnel, sunburst, choropleth … or none.")])


if __name__ == "__main__":
    serve_a2a(ChartExecutor(), agent_card())
