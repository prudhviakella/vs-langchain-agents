# Stage 1 worker: parse -> inspect -> figures -> chunk -> embed -> upsert.
#
# This image is multi-gigabyte because Docling's model weights are baked in
# at build time (below). That is deliberate, not an oversight — see
# infra/batch.py's module docstring for why this project runs on EC2 rather
# than Fargate: EC2 pulls this image ONCE per instance and reuses it across
# every job that instance runs, where Fargate would pull it fresh per task.
# The size only matters once per instance's lifetime, not once per document.
#
# aws s3 sync is used by ingest.py at runtime for cache round-tripping, so
# the AWS CLI (not just boto3) has to be present in the image — a common
# thing to miss since boto3 alone does not provide the `aws` command.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 poppler-utils curl unzip \
    && rm -rf /var/lib/apt/lists/*

# The AWS CLI v2, for `aws s3 sync` inside ingest.py. Installed from AWS's
# own bundle rather than pip's awscli, which is unmaintained for v2 and
# regularly falls behind on S3 API changes.
RUN curl -sL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip \
    && unzip -q awscliv2.zip && ./aws/install && rm -rf awscliv2.zip aws

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Model weights, downloaded once at BUILD time. Doing this at container
# START time instead would add several minutes and an outbound dependency
# to every single job, on top of the per-instance cost this Dockerfile is
# already structured to minimise.
RUN python -c "from docling.utils.model_downloader import download_models; download_models()"

COPY rag/ ./rag/
COPY ingest.py ./

ENTRYPOINT ["python", "ingest.py"]
