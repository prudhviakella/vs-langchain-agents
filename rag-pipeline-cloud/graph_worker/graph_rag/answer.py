"""Turning a traversal into an answer.

Two modes, answering different question shapes:

    local    entity match -> traverse -> fetch those chunks -> answer from them
    global   community summaries -> answer from the summaries

Local handles "what connects X and Y". Global handles "what are the themes across
all of this" — a question top-k retrieval cannot answer at all, because it samples
five chunks from three thousand and calls that the corpus.
"""

import os
import re

from . import config, store
from .chunks import fetch_text
from .schema import normalise


def _client():
    from openai import OpenAI

    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# Words too short or too common to be worth matching an entity key against. Without
# this, "the" matches every entity whose name contains it and the traversal returns
# the whole graph.
_STOPWORDS = {
    "what", "which", "where", "when", "who", "how", "why", "does", "did", "are",
    "and", "the", "for", "with", "from", "this", "that", "these", "those", "about",
    "between", "across", "have", "has", "was", "were", "been", "being", "into",
}


def query_terms(question: str, minimum_length: int = 4) -> list[str]:
    """Words from a question worth matching against entity keys.

    Normalised with the same function the keys were built with, so "Pfizer's" in a
    question matches the node stored as "pfizer".
    """
    words = normalise(re.sub(r"[^\w\s]", " ", question)).split()
    return [w for w in words if len(w) >= minimum_length and w not in _STOPWORDS]


def local(session, index, question: str, hops: int = 1,
          limit: int = 8) -> dict:
    """Answer from chunks reached by traversing the graph.

    The graph decides *which* chunks; the vector store returns *what they say*. The
    fetch is by id, so it is exact — no similarity involved, which is why this is
    precise where a vector search is approximate.
    """
    terms = query_terms(question)
    rows = store.find_chunks(session, terms, hops=hops, limit=limit)
    if not rows:
        # Worth returning explicitly rather than as an empty answer: "no entity in
        # this question is in the graph" is a different failure from "the graph has
        # nothing to say", and only the first is fixable by extraction.
        return {"answer": None, "reason": "no entity in the question matched the graph",
                "terms": terms, "chunks": []}

    texts = fetch_text(index, [row["chunkId"] for row in rows])
    passages = []
    for row in rows:
        meta = texts.get(row["chunkId"])
        if meta:
            passages.append({**row, "text": meta.get("text", "")})

    context = "\n\n".join(
        f"[{p['chunkId']} p{p['page']}  via {', '.join(p['via'])}]\n{p['text']}"
        for p in passages)

    response = _client().chat.completions.create(
        model=config.LLM_MODEL, temperature=0,
        messages=[
            {"role": "system", "content":
             "Answer from the provided passages only. Cite pages as [p12]. State "
             "plainly if they do not contain the answer."},
            {"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return {"answer": response.choices[0].message.content,
            "reason": None, "terms": terms, "chunks": passages}


def summarise_community(members: list[str]) -> str:
    """One paragraph describing what a cluster of entities has in common.

    Computed once per community, not per query. That is the whole point: a
    corpus-level question is answered from precomputed summaries because the
    alternative is reading the entire corpus at query time.
    """
    response = _client().chat.completions.create(
        model=config.LLM_MODEL, temperature=0,
        messages=[
            {"role": "system", "content":
             "Describe what this group of entities from a document corpus has in "
             "common, in one paragraph. Name them. State only what the grouping "
             "shows; do not speculate about significance."},
            {"role": "user", "content": ", ".join(members)},
        ],
    )
    return response.choices[0].message.content.strip()


def build_summaries(session, min_size: int = 3, top: int = 12) -> list[dict]:
    """Summarise the largest communities in the graph.

    Capped at `top` because summarisation is one call each and the long tail of
    two-node communities carries no theme worth naming.
    """
    found = store.communities(session, min_size=min_size)[:top]
    for community in found:
        community["summary"] = summarise_community(community["members"])
    return found


def glob(summaries: list[dict], question: str) -> dict:
    """Answer a corpus-level question from community summaries.

    Named `glob` rather than `global`, which is a keyword.
    """
    if not summaries:
        return {"answer": None, "reason": "no communities were built"}

    context = "\n\n".join(
        f"[{c['type']} cluster around {c['seed']}, {c['size']} entities]\n{c['summary']}"
        for c in summaries)

    response = _client().chat.completions.create(
        model=config.LLM_MODEL, temperature=0,
        messages=[
            {"role": "system", "content":
             "Answer using these cluster summaries of a document corpus. They "
             "describe the whole corpus, not a sample, so a question about overall "
             "themes can be answered from them. Say so if they cannot."},
            {"role": "user", "content": f"Clusters:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return {"answer": response.choices[0].message.content, "reason": None,
            "clusters": len(summaries)}
