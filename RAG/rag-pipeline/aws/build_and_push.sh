#!/usr/bin/env bash
# Build the ingest image and push it to ECR.
set -euo pipefail

PROJECT="${PROJECT:-rag-pipeline}"
REGION="${AWS_REGION:-us-east-1}"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
REPO="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${PROJECT}"

cd "$(dirname "$0")/.."

aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"

# Fargate runs x86_64. Building on an Apple Silicon machine without this flag
# produces an arm64 image that fails to start with an exec format error.
docker build --platform linux/amd64 -t "${PROJECT}:latest" .
docker tag "${PROJECT}:latest" "${REPO}:latest"
docker push "${REPO}:latest"

echo "pushed ${REPO}:latest"
