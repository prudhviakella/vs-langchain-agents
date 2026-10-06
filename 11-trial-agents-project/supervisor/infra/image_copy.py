"""Prebuilt images: publish once to Docker Hub (instructor), copy into ECR without Docker (students).

    instructor (has Docker):   publish("<you>/trial-graph-agent:1.0", agent_code/)
        docker buildx build --platform linux/arm64 --push        → Docker Hub

    student (no Docker):       copy("<you>/trial-graph-agent:1.0", <account>.dkr.ecr…/trial-graph-agent)
        STEP 1  Docker Hub: anonymous pull token; the manifest (from an index: the linux/arm64 one)
        STEP 2  each blob (config + layers): already in ECR? skip — else download, CHECK its sha256,
                upload with ECR's layer API (initiate → parts → complete)
        STEP 3  ECR put_image: the SAME manifest bytes (so the digest is unchanged), tag :latest
                → "<repo uri>:latest", which deploy_runtime uses exactly as a locally built image

WHY THIS WORKS WITHOUT DOCKER
    An image is just a manifest (JSON) plus blobs (tar.gz files) named by their sha256. Copying
    it is plain HTTPS reads from Docker Hub and plain AWS API writes to ECR.

WHY ECR AT ALL
    AgentCore runs images from Amazon ECR only — a Docker Hub image cannot be used directly.

WHAT THIS DOES NOT DO
    It copies ONE platform, linux/arm64 (the only one AgentCore runs). It reads public Docker Hub
    images only (no Docker Hub login). It never modifies the image.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile

import boto3

HUB_REGISTRY = "https://registry-1.docker.io"
HUB_AUTH = "https://auth.docker.io/token"
INDEX_TYPES = ("application/vnd.oci.image.index.v1+json", "application/vnd.docker.distribution.manifest.list.v2+json")
MANIFEST_TYPES = ("application/vnd.oci.image.manifest.v1+json", "application/vnd.docker.distribution.manifest.v2+json")
PART = 10 * 1024 * 1024                        # ECR layer part size


def parse(ref: str) -> tuple[str, str]:
    """'<you>/trial-graph-agent:1.0' -> ('<you>/trial-graph-agent', '1.0'); default tag 'latest'."""
    ref = ref.removeprefix("docker.io/")
    name, _, tag = ref.rpartition(":") if ":" in ref.split("/")[-1] else (ref, "", "latest")
    if "/" not in name:
        name = f"library/{name}"
    return name, tag or "latest"


def publish(ref: str, context: str, sh=subprocess.run) -> str:
    """Instructor only: build linux/arm64 and push to Docker Hub (run `docker login` first)."""
    name, tag = parse(ref)
    image = f"{name}:{tag}"
    print(f"  building and pushing {image} (linux/arm64) to Docker Hub")
    sh(["docker", "buildx", "build", "--platform", "linux/arm64", "--provenance=false",
        "-t", image, "--push", context], check=True)
    return image


def _client():
    import httpx
    return httpx.Client(timeout=300, follow_redirects=True)


def _hub_token(http, name: str) -> str:
    r = http.get(HUB_AUTH, params={"service": "registry.docker.io", "scope": f"repository:{name}:pull"})
    r.raise_for_status()
    return r.json()["token"]


def _manifest(http, name: str, ref: str, token: str) -> tuple[bytes, str]:
    r = http.get(f"{HUB_REGISTRY}/v2/{name}/manifests/{ref}",
                 headers={"Authorization": f"Bearer {token}", "Accept": ", ".join(INDEX_TYPES + MANIFEST_TYPES)})
    if r.status_code == 404:
        raise SystemExit(f"image {name}:{ref} not found on Docker Hub — check the name and that it is public")
    r.raise_for_status()
    return r.content, r.headers.get("content-type", "").split(";")[0]


def _arm64(http, name: str, tag: str, token: str) -> tuple[bytes, str]:
    """The linux/arm64 image manifest — directly, or picked out of a multi-platform index."""
    body, media = _manifest(http, name, tag, token)
    if media in INDEX_TYPES or json.loads(body).get("manifests"):
        entries = json.loads(body)["manifests"]
        match = next((m for m in entries if (m.get("platform") or {}).get("os") == "linux"
                      and (m.get("platform") or {}).get("architecture") == "arm64"), None)
        if not match:
            found = sorted({f"{(m.get('platform') or {}).get('os')}/{(m.get('platform') or {}).get('architecture')}"
                            for m in entries})
            raise SystemExit(f"{name}:{tag} has no linux/arm64 image (found: {found}) — AgentCore needs arm64")
        body, media = _manifest(http, name, match["digest"], token)
    return body, media


def _download(http, name: str, digest: str, token: str, out) -> int:
    """Stream one blob to a file, verifying its sha256 — a bad download never reaches ECR."""
    sha, size = hashlib.sha256(), 0
    with http.stream("GET", f"{HUB_REGISTRY}/v2/{name}/blobs/{digest}", headers={"Authorization": f"Bearer {token}"}) as r:
        r.raise_for_status()
        for chunk in r.iter_bytes(1024 * 1024):
            sha.update(chunk)
            out.write(chunk)
            size += len(chunk)
    if f"sha256:{sha.hexdigest()}" != digest:
        raise SystemExit(f"blob {digest[:19]}… failed its checksum — download corrupted; run again")
    out.flush()
    return size


def _upload(ecr, repo: str, digest: str, path: str, size: int) -> None:
    upload_id = ecr.initiate_layer_upload(repositoryName=repo)["uploadId"]
    with open(path, "rb") as f:
        first = 0
        while first < size:
            blob = f.read(PART)
            ecr.upload_layer_part(repositoryName=repo, uploadId=upload_id, partFirstByte=first,
                                  partLastByte=first + len(blob) - 1, layerPartBlob=blob)
            first += len(blob)
    try:
        ecr.complete_layer_upload(repositoryName=repo, uploadId=upload_id, layerDigests=[digest])
    except ecr.exceptions.LayerAlreadyExistsException:
        pass


def copy(ref: str, repo_uri: str, region: str | None = None, tag: str = "latest", http=None, ecr=None) -> str:
    """Copy a public Docker Hub image into ECR; returns '<repo_uri>:<tag>'."""
    name, src_tag = parse(ref)
    repo = repo_uri.split("/", 1)[1]
    ecr = ecr or boto3.client("ecr", region_name=region)
    http = http or _client()
    print(f"  copying {name}:{src_tag} (linux/arm64) from Docker Hub into ECR {repo} — no Docker needed")
    # STEP 1
    token = _hub_token(http, name)
    manifest, media = _arm64(http, name, src_tag, token)
    doc = json.loads(manifest)
    blobs = [doc["config"]] + doc["layers"]
    # STEP 2
    have = ecr.batch_check_layer_availability(repositoryName=repo, layerDigests=[b["digest"] for b in blobs])
    present = {l["layerDigest"] for l in have.get("layers", []) if l.get("layerAvailability") == "AVAILABLE"}
    for i, b in enumerate(blobs, 1):
        mb = b.get("size", 0) / 1e6
        if b["digest"] in present:
            print(f"    [{i}/{len(blobs)}] {b['digest'][7:19]} {mb:7.1f} MB  already in ECR")
            continue
        with tempfile.NamedTemporaryFile() as tmp:
            size = _download(http, name, b["digest"], token, tmp)
            _upload(ecr, repo, b["digest"], tmp.name, size)
        print(f"    [{i}/{len(blobs)}] {b['digest'][7:19]} {mb:7.1f} MB  copied")
    # STEP 3
    try:
        ecr.put_image(repositoryName=repo, imageManifest=manifest.decode(), imageManifestMediaType=media, imageTag=tag)
    except ecr.exceptions.ImageAlreadyExistsException:
        pass                                                   # same manifest already tagged — nothing to do
    image = f"{repo_uri}:{tag}"
    print(f"  {image} ready")
    return image
