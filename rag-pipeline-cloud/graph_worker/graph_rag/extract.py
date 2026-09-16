"""Turning chunk text into entities and relations.

One LLM call per chunk. That is the dominant cost of building a graph, and the
reason the cache exists: extraction is paid once per chunk per schema, not once per
run.
"""

import hashlib
import json
import os
import time
from collections import Counter

from . import config
from .schema import extraction_prompt, validate


def _client():
    """An OpenAI client, created lazily so importing this module needs no key."""
    from openai import OpenAI

    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def extract_chunk(text: str, chunk_id: str) -> dict:
    """Extract entities and relations from one chunk.

    Cached by (model, prompt, text). The prompt is in the key because changing the
    schema changes the output — a cache that ignored it would return extractions
    made under a schema that no longer exists, and the change would appear to have
    done nothing.
    """
    digest = hashlib.sha256(
        (config.EXTRACT_MODEL + "\x00" + extraction_prompt() + "\x00" + text).encode()
    ).hexdigest()[:24]
    cached = config.CACHE_DIR / f"{digest}.json"
    if cached.exists():
        return json.loads(cached.read_text())

    for attempt in range(4):
        try:
            response = _client().chat.completions.create(
                model=config.EXTRACT_MODEL, temperature=0, seed=0,
                # Structured output rather than parsing prose. Without it roughly one
                # call in twenty comes back wrapped in a code fence or preceded by an
                # apology, and the parse fails on a chunk at random.
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": extraction_prompt()},
                    # Truncated: a pathological chunk can be long, and the schema is
                    # satisfied by what the opening carries.
                    {"role": "user", "content": text[:8000]},
                ],
            )
            payload = json.loads(response.choices[0].message.content)
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)

    entities, relations, rejected = validate(payload)
    result = {"chunk_id": chunk_id, "entities": entities,
              "relations": relations, "rejected": rejected}
    cached.write_text(json.dumps(result, indent=2))
    return result


def extract_all(chunks: list[dict], verbose: bool = True) -> list[dict]:
    """Extract from every chunk, reporting what the schema rejected.

    Sequential. Extraction is I/O-bound and would parallelise, but a thread pool
    competes with rate limits for no benefit at this corpus size, and the cache means
    the cost is paid once.
    """
    results: list[dict] = []
    rejected: Counter = Counter()

    for n, chunk in enumerate(chunks, 1):
        result = extract_chunk(chunk["text"], chunk["chunk_id"])
        results.append(result)
        rejected.update(result["rejected"])
        if verbose and n % 25 == 0:
            print(f"  extracted {n}/{len(chunks)}", flush=True)

    if verbose:
        report(results, rejected)
    return results


def report(results: list[dict], rejected: Counter | None = None) -> dict:
    """Print what extraction found, and return it.

    Mentions per entity is the number worth watching. Close to 1.0 means nearly
    every entity was seen once, which usually means resolution failed — and a graph
    of isolated nodes cannot be traversed, while looking identical to one that simply
    has no answers.
    """
    mentions = sum(len(r["entities"]) for r in results)
    unique = {e["key"] for r in results for e in r["entities"]}
    empty = sum(1 for r in results if not r["entities"])
    entity_types = Counter(e["type"] for r in results for e in r["entities"])
    relation_types = Counter(r_["type"] for r in results for r_ in r["relations"])

    print(f"  {mentions} mentions of {len(unique)} distinct entities "
          f"({mentions / max(len(unique), 1):.1f} per entity)", flush=True)
    print(f"  {sum(relation_types.values())} relations, "
          f"{empty}/{len(results)} chunks with nothing", flush=True)

    if rejected:
        print("  rejected by the schema:", flush=True)
        for reason, count in rejected.most_common(8):
            print(f"    {count:>4}  {reason}", flush=True)

    return {"mentions": mentions, "unique": len(unique), "empty": empty,
            "entity_types": dict(entity_types),
            "relation_types": dict(relation_types)}
