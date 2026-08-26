"""Shared pieces for the agent fleet.

    config      ports, provider, budgets
    contracts   what the agents promise each other
    llm         one interface, two providers
"""

from . import config, contracts, llm

__all__ = ["config", "contracts", "llm"]
