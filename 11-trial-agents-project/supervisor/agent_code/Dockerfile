# supervisor — AgentCore Runtime container.
#
#   build   docker buildx build --platform linux/arm64 ...  (Runtime is arm64 only)
#   run     opentelemetry-instrument python main.py         (port 9000, A2A)
#
# OBSERVABILITY: `opentelemetry-instrument` loads aws-opentelemetry-distro, which
# configures the exporters. On AgentCore Runtime the OTEL environment is preset by
# the runtime; spans reach CloudWatch (aws/spans, GenAI Observability) once
# CloudWatch Transaction Search is enabled for the account. Running plain
# `python main.py` — what an earlier version did — creates spans and exports none,
# with no error anywhere.

# No --platform here: `buildx build --platform linux/arm64` selects the arm64
# base image. A constant FROM --platform only adds a build warning.
FROM python:3.12-slim

WORKDIR /app

# OTEL_SERVICE_NAME names this agent in traces and in the GenAI Observability
# dashboard. PYTHONUNBUFFERED: logs reach CloudWatch as they happen, not on flush.
ENV PYTHONUNBUFFERED=1 \
    OTEL_SERVICE_NAME=supervisor

COPY pyproject.toml ./
RUN pip install --no-cache-dir .

COPY supervisor ./supervisor
COPY main.py ./

# Fail the BUILD if a top-level file shadows a stdlib module a dependency imports
# (a stray secrets.py breaks `from secrets import token_hex` at runtime).
RUN for m in secrets token types io json logging queue select signal socket; do \
        if [ -f "/app/$m.py" ]; then echo "/app/$m.py shadows stdlib '$m'" >&2; exit 1; fi; \
    done

# Not root.
RUN useradd -m -u 1000 bedrock_agentcore && chown -R bedrock_agentcore /app
USER bedrock_agentcore

EXPOSE 9000

CMD ["opentelemetry-instrument", "python", "main.py"]
