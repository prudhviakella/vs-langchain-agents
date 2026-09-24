# supervisor — AgentCore Runtime container. Same hard requirements as
# trial_graph/agent_code/Dockerfile: linux/arm64, port 9000. See that
# file's own comments for the verification behind both.

FROM --platform=linux/arm64 python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir --break-system-packages .

COPY supervisor ./supervisor
COPY main.py ./

EXPOSE 9000

CMD ["python", "main.py"]
