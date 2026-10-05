# supervisor — AgentCore Runtime container (linux/arm64, port 9000, non-root)
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    OTEL_SERVICE_NAME=ecom_supervisor
COPY pyproject.toml ./
RUN pip install --no-cache-dir .
COPY supervisor ./supervisor
COPY main.py ./
# a module named like a stdlib one (secrets.py, json.py…) would shadow it at import
RUN for m in secrets token types io json logging queue select signal socket; do \
        if [ -f "/app/$m.py" ]; then echo "/app/$m.py shadows stdlib '$m'" >&2; exit 1; fi; \
    done
RUN useradd -m -u 1000 bedrock_agentcore && chown -R bedrock_agentcore /app
USER bedrock_agentcore
EXPOSE 9000
CMD ["opentelemetry-instrument", "python", "main.py"]
