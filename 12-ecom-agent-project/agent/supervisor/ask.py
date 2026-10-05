"""Ask the DEPLOYED supervisor one question and watch the stream — a smoke test of the whole chain.

    python ask.py "What is revenue by year?"

    deployment.json ─► invoke_agent_runtime (SigV4, accept text/event-stream, A2A message/stream)
        status     ─►  · nlq: executing
        reasoning  ─►  the plan, the SQL, row counts, the chart
        token      ─►  the answer, as it is written
        final      ─►  a summary: table size, charts, agents, tokens, trace id

WHAT THIS DOES NOT DO
    It keeps no conversation (each run is a new one) and stores nothing.
"""
import json
import sys
import time
import uuid
from pathlib import Path

import boto3

HERE = Path(__file__).resolve().parent


def _envelopes(frame: dict):
    r = frame.get("result") or {}
    status = r.get("status") or {}
    parts = (status.get("message") or {}).get("parts") or (r.get("artifact") or {}).get("parts") or []
    if status.get("state") == "failed":
        yield {"type": "error", "text": " ".join(p.get("text", "") for p in parts)}
    for p in parts:
        text = p.get("text")
        if text:
            try:
                yield json.loads(text)
            except ValueError:
                pass


def main() -> None:
    question = " ".join(sys.argv[1:]) or "What is revenue by year?"
    dep = json.loads((HERE / "deployment.json").read_text(encoding="utf-8"))
    conversation = f"ask-{uuid.uuid4().hex}"
    body = {"jsonrpc": "2.0", "id": "1", "method": "message/stream", "params": {"message": {
        "role": "user", "messageId": f"msg-{uuid.uuid4().hex}", "contextId": conversation,
        "parts": [{"kind": "text", "text": json.dumps({"question": question, "history": [],
                                                       "conversation_id": conversation})}]}}}
    client = boto3.client("bedrock-agentcore", region_name=dep["region"])
    print(f"Q: {question}\n")
    t0 = time.time()
    resp = client.invoke_agent_runtime(agentRuntimeArn=dep["runtime_arn"], qualifier="DEFAULT",
                                       runtimeSessionId=f"{conversation}:supervisor", contentType="application/json",
                                       accept="text/event-stream", payload=json.dumps(body).encode())
    answering = False
    for raw in resp["response"].iter_lines():
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        if not line.startswith("data:"):
            continue
        for ev in _envelopes(json.loads(line[5:])):
            kind = ev.get("type")
            if kind == "status":
                print(f"  · {ev.get('agent')}: {ev.get('phase')}" + (f" — {ev['detail']}" if ev.get("detail") else ""), flush=True)
            elif kind == "reasoning":
                print("  " + ev["text"].rstrip().replace("\n", "\n  "), flush=True)
            elif kind == "token":
                if not answering:
                    print("\nA: ", end="", flush=True)
                    answering = True
                print(ev["text"], end="", flush=True)
            elif kind == "final":
                arts = ev.get("artifacts") or {}
                table = arts.get("table") or {}
                usage = {u["agent"]: u["usage"].get("total_tokens", 0) for u in ev.get("usage_by_agent") or []}
                print(f"\n\n— {time.time() - t0:.1f}s · agents {ev.get('agents_used')} · "
                      f"table {len(table.get('rows') or [])} rows × {len(table.get('columns') or [])} cols · "
                      f"charts {len(arts.get('charts') or [])} · tokens {usage} · trace {ev.get('trace_id')}")
            elif kind == "error":
                print(f"\nFAILED: {ev['text']}")


if __name__ == "__main__":
    main()
