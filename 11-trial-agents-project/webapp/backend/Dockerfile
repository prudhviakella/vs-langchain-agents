# Backend: FastAPI on uvicorn, port 8000. Built for linux/amd64 (ECS Fargate
# default in the CloudFormation stack).
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--timeout-keep-alive", "75"]
