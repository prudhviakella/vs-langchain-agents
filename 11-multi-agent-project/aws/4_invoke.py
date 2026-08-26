"""Step 4 — call the deployed supervisor.

    python 4_invoke.py "which sponsors run more than one trial?"
    python 4_invoke.py "..." --stream

Signed with SigV4 from your own AWS credentials. No token to fetch, nothing to
log in to — boto3 signs the request with whatever `aws configure` set up.

This is also how the supervisor calls its specialists in the cloud, which is why
there is no Cognito anywhere in the agent-to-agent path.
"""

import argparse
import json
from pathlib import Path

import boto3

import config


def main() -> None:
    ap = argparse.ArgumentParser(description="Invoke the deployed supervisor")
    ap.add_argument("question")
    ap.add_argument("--agent", default="supervisor", choices=list(config.AGENTS))
    ap.add_argument("--stream", action="store_true",
                    help="use message/stream and print events as they arrive")
    args = ap.parse_args()

    state = json.loads(Path(config.STATE_FILE).read_text())
    arn = state["runtimes"][args.agent]

    client = boto3.client("bedrock-agentcore", region_name=config.REGION)

    # The same A2A envelope as the local module. The transport changed; the
    # message did not.
    payload = {
        "jsonrpc": "2.0", "id": "1",
        "method": "message/stream" if args.stream else "message/send",
        "params": {"message": {"role": "user",
                               "parts": [{"kind": "text", "text": args.question}]}},
    }

    response = client.invoke_agent_runtime(
        agentRuntimeArn=arn,
        # A stable session id keeps the same warm microVM across turns, which is
        # what makes a conversation feel continuous rather than cold-starting
        # every time. AgentCore requires 33+ characters.
        runtimeSessionId=("cli-session-" + "0" * 33)[:48],
        payload=json.dumps(payload).encode(),
        qualifier="DEFAULT")

    body = response["response"].read().decode()

    if args.stream:
        for line in body.splitlines():
            if line.startswith("data: "):
                event = json.loads(line[6:])
                note = event.get("note") or event.get("state", "")
                print(f"  · {note}")
        return

    result = json.loads(body).get("result", {})
    for artifact in result.get("artifacts", []):
        print(f"\n── {artifact['name']}")
        print(artifact["parts"][0]["text"][:1200])


if __name__ == "__main__":
    main()
