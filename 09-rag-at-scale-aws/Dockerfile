FROM python:3.11-slim

# Docling's layout and table models need these at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model weights at build time. Doing this at runtime would add several
# minutes and an outbound dependency to every single task.
RUN python -c "from docling.utils.model_downloader import download_models; download_models()"

# The package, then the entrypoint that runs it.
COPY rag/ ./rag/
COPY ingest.py ./

ENTRYPOINT ["python", "ingest.py"]
