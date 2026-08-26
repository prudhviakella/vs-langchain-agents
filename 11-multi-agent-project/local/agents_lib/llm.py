"""The chat model, for either provider.

Returns a LangChain chat model, because every agent here is a LangChain agent.
One function, so the provider is a detail of this file rather than a shape the
rest of the code is written around.
"""

from __future__ import annotations

import os

from . import config


def chat_model(temperature: float = 0.0):
    """A LangChain chat model for whichever provider is configured.

    Both return a `BaseChatModel`, so `create_agent` binds tools to either one
    without knowing the difference. That is the whole reason to go through
    LangChain rather than calling the APIs directly: tool calling, structured
    output and streaming all work the same way on both.
    """
    if config.PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=config.ANTHROPIC_MODEL,
                             temperature=temperature, max_tokens=2000,
                             api_key=os.environ["ANTHROPIC_API_KEY"])

    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=config.OPENAI_MODEL, temperature=temperature,
                      api_key=os.environ["OPENAI_API_KEY"])


def describe() -> str:
    """Which model is actually in use. Printed at startup, so a surprising bill
    or a surprising answer has an obvious first thing to check."""
    model = (config.ANTHROPIC_MODEL if config.PROVIDER == "anthropic"
             else config.OPENAI_MODEL)
    return f"{config.PROVIDER}:{model}"
