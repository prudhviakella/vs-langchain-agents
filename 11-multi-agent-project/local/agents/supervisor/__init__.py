"""The supervisor.

    main.py         A2A bridge — one task in, events and artifacts out
    core.py         the loop, then render, then compose
    tools.py        ONE tool: call_agent
    middleware.py   the policies the model cannot route around
    prompt.py       what it may decide; deliberately no agent list
    state.py        where the data lives instead of in the model
    discovery.py    who exists, fetched every turn
    context.py      per-turn callbacks; the model never sees these
"""
