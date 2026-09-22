"""Shared A2A SDK server helpers for the three learning agents."""

from collections.abc import Awaitable, Callable

import uvicorn
from starlette.applications import Starlette

from a2a.server.agent_execution import AgentExecutor
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH


def make_card(
    *,
    name: str,
    description: str,
    url: str,
    skill_id: str,
    skill_name: str,
    skill_description: str,
    examples: list[str],
    streaming: bool,
) -> AgentCard:
    return AgentCard(
        name=name,
        description=description,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=streaming),
        supported_interfaces=[
            AgentInterface(
                protocol_binding="JSONRPC",
                url=url,
                protocol_version="1.0",
            )
        ],
        skills=[
            AgentSkill(
                id=skill_id,
                name=skill_name,
                description=skill_description,
                input_modes=["text/plain"],
                output_modes=["text/plain"],
                tags=["a2a", "learning-example"],
                examples=examples,
            )
        ],
    )


def run_agent(card: AgentCard, executor: AgentExecutor, port: int) -> None:
    """Create the standard SDK request handler/routes and start Uvicorn."""
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )

    app = Starlette(
        routes=[
            *create_agent_card_routes(card),
            *create_jsonrpc_routes(handler, "/"),
        ]
    )

    print(f"{card.name} listening on {card.supported_interfaces[0].url}")
    print(
        f"Agent Card: "
        f"http://127.0.0.1:{port}{AGENT_CARD_WELL_KNOWN_PATH}"
    )
    uvicorn.run(app, host="127.0.0.1", port=port)
