"""What the agents promise each other.

This file is the interface between four programs that share no code at runtime.
It is the most important file in the module, and it is also the shortest.

THE PROBLEM IT SOLVES

The supervisor cannot see inside a specialist. It cannot check the specialist's
memory, read its prompt, or inspect what it did. All it gets back is an
artifact.

So the artifact has to say what it is. Not "here are some results" — the
supervisor then has to guess whether they are rows it could chart, passages it
should quote, or nothing at all. Guessing produces a system that works on the
examples you tried.

Every specialist therefore declares a `shape`, and the supervisor routes on it.
Opacity forces the contract into the message. That is not a workaround; it is
the reason A2A works between teams.
"""

from __future__ import annotations

from typing import Any, Literal

# What a specialist can return. Adding a shape means adding a branch in the
# supervisor, which is the correct amount of friction — a new shape nobody
# handles is a silent no-op.
Shape = Literal["passages", "table", "graph", "empty", "error"]


def passages(items: list[dict[str, Any]], note: str = "") -> dict[str, Any]:
    """Text found by search. Quotable, not chartable."""
    return {"shape": "passages", "passages": items,
            "count": len(items), "note": note}


def table(columns: list[str], rows: list[list[Any]], note: str = "") -> dict[str, Any]:
    """Rows and columns. The only shape the chart agent accepts."""
    return {"shape": "table", "columns": columns, "rows": rows,
            "row_count": len(rows), "note": note}


def graph(nodes: list[dict], edges: list[dict], note: str = "") -> dict[str, Any]:
    """Nodes and edges. Chartable only as a network; usually shown as-is."""
    return {"shape": "graph", "nodes": nodes, "edges": edges,
            "node_count": len(nodes), "edge_count": len(edges), "note": note}


def empty(reason: str) -> dict[str, Any]:
    """The query ran and found nothing.

    Distinct from `error` on purpose. "No trials match that sponsor" is an
    answer — the analyst learns something true. "Neo4j refused the connection"
    is a failure. Collapsing them into one shape means the supervisor tells a
    user there is no data when in fact the database was down, which is the worst
    possible outcome of the two.
    """
    return {"shape": "empty", "reason": reason}


def error(reason: str) -> dict[str, Any]:
    """Something broke. Say so; do not answer anyway."""
    return {"shape": "error", "reason": reason}


def summarise_for_model(result: dict[str, Any], row_preview: int = 3) -> str:
    """What the supervisor's MODEL sees of a result.

    Never the whole thing. The model decides what to do next and writes the
    final prose; it does not process the data. Giving it 400 rows costs tokens,
    slows the turn, and tempts it to retype values into its answer — and a
    retyped identifier that is one digit wrong is worse than no answer, because
    it looks right.

    The full result stays in the supervisor's memory and travels to the chart
    agent by reference. The model gets shape, size, and a few examples.
    """
    shape = result.get("shape")

    if shape == "table":
        head = result["rows"][:row_preview]
        return (f"table: {result['row_count']} rows, columns "
                f"{result['columns']}. First {len(head)}: {head}")

    if shape == "passages":
        heads = [p.get("text", "")[:120] for p in result["passages"][:row_preview]]
        return f"passages: {result['count']} found. Openings: {heads}"

    if shape == "graph":
        return (f"graph: {result['node_count']} nodes, "
                f"{result['edge_count']} edges")

    if shape == "empty":
        return f"empty: {result['reason']}"

    return f"error: {result.get('reason', 'unknown')}"
