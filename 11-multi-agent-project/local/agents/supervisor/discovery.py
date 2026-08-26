"""Who exists, discovered every turn.

The agent NAME is data, not code. There is no `@tool` per specialist and no
hardcoded list — the supervisor fetches the cards, and what it finds becomes the
AVAILABLE AGENTS block in the prompt.

Two things follow, and both matter:

Adding an agent is configuration. No new tool to write, no capability list to
update, and therefore no second list to drift out of step with the first.

An agent that is down cannot be routed to. It is absent from the block the model
reads, so the model cannot pick something that is not there — rather than picking
it and getting an error it has to recover from.

Discovery runs PER TURN, not at startup. An agent that was down when this process
booted should not be invisible forever, and in a real deployment a specialist
restarting is ordinary.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Found:
    name: str
    description: str
    endpoint: str
    skills: list[str] = field(default_factory=list)


@dataclass
class Registry:
    """What was reachable this turn."""

    routable: dict[str, Found] = field(default_factory=dict)   # data agents
    renderers: dict[str, Found] = field(default_factory=dict)  # chart, and so on

    def names(self) -> list[str]:
        return sorted(self.routable)

    def endpoint(self, name: str) -> str:
        found = self.routable.get(name) or self.renderers.get(name)
        return found.endpoint if found else ""

    def as_prompt_block(self) -> str:
        """The AVAILABLE AGENTS block, built from the cards themselves.

        Not written by hand anywhere. A description kept in the prompt is a
        second copy of what the agent already says about itself, and it is how a
        prompt ends up describing an agent that was disabled last week.
        """
        if not self.routable:
            return "No data agents are reachable."
        lines = []
        for found in self.routable.values():
            lines.append(f"- {found.name}: {found.description}")
            for skill in found.skills:
                lines.append(f"    {skill}")
        return "\n".join(lines)


async def discover(endpoints: dict[str, str], renderers: set[str]) -> Registry:
    """Fetch every card. Anything unreachable is simply absent."""
    from a2a_mini import A2AClient

    registry = Registry()
    for name, url in endpoints.items():
        try:
            card = await A2AClient(url).discover()
        except Exception:
            continue          # down, not broken — the model is told what exists
        found = Found(name=card.name, description=card.description, endpoint=url,
                      skills=[s.description for s in card.skills])
        target = registry.renderers if name in renderers else registry.routable
        target[name] = found
    return registry
