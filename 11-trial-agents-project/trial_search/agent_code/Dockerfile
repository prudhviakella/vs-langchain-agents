# trial_search — AgentCore Runtime container. Same requirements as
# trial_graph/agent_code/Dockerfile; see that file's own comments for
# why linux/arm64 and port 9000 are both hard requirements, verified
# directly against AWS's own docs and the bedrock_agentcore SDK source.

FROM --platform=linux/arm64 python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir --break-system-packages .

COPY trial_search ./trial_search
COPY main.py ./

EXPOSE 9000

CMD ["python", "main.py"]
