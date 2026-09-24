# trial_graph — AgentCore Runtime container.
#
# Two requirements verified directly against AWS's own documentation and
# the bedrock_agentcore SDK's source before writing this, not assumed:
#   1. Platform MUST be linux/arm64 — AgentCore Runtime does not run
#      x86_64 images. Build with:
#        docker buildx build --platform linux/arm64 -t <tag> --push .
#   2. Port 9000 — the A2A protocol's contract port, hardcoded into
#      bedrock_agentcore.runtime.a2a as A2A_CONTRACT_PORT. Binding
#      elsewhere fails every deployed invocation with HTTP 424, since
#      the Runtime only proxies to 9000 — confirmed from the SDK's own
#      source, not from a generic AgentCore example (several public
#      examples show port 8080, which is for a different, non-A2A
#      agent contract).
#
# No EXPOSE for a /ping route to build ourselves — serve_a2a() in
# main.py registers GET /ping automatically.

FROM --platform=linux/arm64 python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir --break-system-packages .

COPY trial_graph ./trial_graph
COPY main.py ./

EXPOSE 9000

CMD ["python", "main.py"]
