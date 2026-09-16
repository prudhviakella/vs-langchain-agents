"""Two ECR repositories, not one.

The Stage 1 worker carries Docling's model weights baked in — multi-gigabyte.
The Stage 2 graph worker reads chunks.json and talks to Neo4j and
ClinicalTrials.gov; it needs none of that. Sharing one repository would mean
every graph-worker pull costs the same as a parse-worker pull, for an image
that does not need to be that size — exactly the per-task pull cost this
project's EC2-over-Fargate decision was meant to avoid, reintroduced through
the back door.
"""

import boto3
from botocore.exceptions import ClientError

from . import config

ecr = boto3.client("ecr", region_name=config.REGION)


def exists(fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
        return True
    except ClientError:
        return False


def create_repo(name: str) -> str:
    print(f"-- ECR repository {name} --")
    if exists(ecr.describe_repositories, repositoryNames=[name]):
        print("   exists")
    else:
        ecr.create_repository(
            repositoryName=name,
            imageScanningConfiguration={"scanOnPush": True},
        )
        print("   created, scan-on-push enabled")
    return ecr.describe_repositories(repositoryNames=[name]) \
              ["repositories"][0]["repositoryUri"]


def create_repos() -> dict:
    return {
        "worker": create_repo(config.ECR_REPO_WORKER),
        "graph_worker": create_repo(config.ECR_REPO_GRAPH),
    }
