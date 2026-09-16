# Stage 2 graph worker: structure -> registry -> link.
#
# Deliberately a separate, thin image from the Stage 1 worker. This job
# reads chunks.json and talks to Neo4j and ClinicalTrials.gov — it imports
# neither Docling, OpenAI, nor Pinecone (confirmed against graph_rag's
# actual import statements, all of which are lazy and function-scoped, not
# module-level). Sharing Stage 1's multi-gigabyte image here would mean
# every graph job pays the same per-instance pull cost as a parse job, for
# an image most of which it never uses — reintroducing, for this stage, the
# exact per-task image-pull cost that running on EC2 rather than Fargate
# was meant to avoid for Stage 1.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY graph_rag/ ./graph_rag/
COPY build_graph.py ./

ENTRYPOINT ["python", "build_graph.py"]
