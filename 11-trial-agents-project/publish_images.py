#!/usr/bin/env python3
"""Publish every image of this project to Docker Hub, then prove students can pull them.

    docker login                                   (once — your Docker Hub account)
    python publish_images.py --user <dockerhub-user> --tag 1.0
    python publish_images.py --user <dockerhub-user> --tag 1.0 --only supervisor

    for each component
        │
        ├─ STEP 1  python <folder>/deploy.py --publish <user>/<image>:<tag>
        │            (the frontend: --publish-frontend, a scratch image of the built files)
        │            the component's OWN publish path: docker buildx, linux/arm64,
        │            pushed to Docker Hub. Nothing in AWS is created or changed.
        │
        └─ STEP 2  check, with NO login: Docker Hub hands out an anonymous pull
                   token and serves the linux/arm64 manifest
                     -> what a student's `deploy.py --image` does first
        │
        v
    summary: each image OK / FAILED, and the command each student runs

    component          folder              Docker Hub image
    trial_graph        trial_graph/        <user>/trial-graph-agent:<tag>
    trial_search       trial_search/       <user>/trial-search-agent:<tag>
    supervisor         supervisor/         <user>/trial-supervisor-agent:<tag>
    webapp-backend     webapp/deploy/      <user>/trial-webapp-backend:<tag>
    webapp-frontend    webapp/deploy/      <user>/trial-webapp-frontend:<tag>   built files, not a server

WHY STEP 2

A push succeeding says nothing about whether a student can pull. A Docker Hub
repository created by a push takes the account's default visibility; if that
is private, every student's `--image` fails with "not found". The check runs
exactly the anonymous request a student's machine makes.

WHY A VERSION TAG, NOT ONLY :latest

A student who deployed 1.0 keeps getting 1.0 until told otherwise. Pushing a
fix as 1.1 changes nothing for anyone who did not ask for it; overwriting
:latest changes it for everyone, mid-class.

WHAT THIS DOES NOT DO

    - It does not log in to Docker Hub. Run `docker login` first; the script
      never sees your password.
    - It does not deploy anything to AWS. Students do that with --image.
"""
import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# component -> (folder holding its deploy.py, Docker Hub repository name,
#               the deploy.py flag that publishes it, the flag students deploy it with)
COMPONENTS = {
    "trial_graph":     ("trial_graph",   "trial-graph-agent",      "--publish", "--image"),
    "trial_search":    ("trial_search",  "trial-search-agent",     "--publish", "--image"),
    "supervisor":      ("supervisor",    "trial-supervisor-agent", "--publish", "--image"),
    "webapp-backend":  ("webapp/deploy", "trial-webapp-backend",   "--publish", "--image"),
    "webapp-frontend": ("webapp/deploy", "trial-webapp-frontend",  "--publish-frontend",
                        "--frontend-image"),
}
# Any component's copy of image_copy.py — they are identical. Used for STEP 2.
IMAGE_COPY = HERE / "trial_graph" / "infra" / "image_copy.py"


def _image_copy():
    spec = importlib.util.spec_from_file_location("image_copy", IMAGE_COPY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_docker() -> None:
    if subprocess.run(["docker", "version"], capture_output=True).returncode != 0:
        raise SystemExit("Docker is not running — start Docker Desktop, then `docker login`.")


def publish(folder: str, flag: str, ref: str) -> bool:
    """STEP 1 — the component's own deploy.py publish flag, run in its folder."""
    print(f"\n=== {ref}  ({folder}/deploy.py {flag}) ===", flush=True)
    result = subprocess.run([sys.executable, "deploy.py", flag, ref], cwd=HERE / folder)
    return result.returncode == 0


def pullable(ref: str) -> str:
    """STEP 2 — "" if an anonymous client can pull the linux/arm64 image,
    otherwise the reason it cannot."""
    image_copy = _image_copy()
    name, tag = image_copy.parse(ref)
    try:
        with image_copy._client() as http:
            token = image_copy._hub_token(http, name)
            image_copy._arm64(http, name, tag, token)
    except SystemExit as exc:                       # not found / no arm64, already explained
        return str(exc)
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            return ("the repository is PRIVATE — make it public on hub.docker.com: "
                    f"Repositories -> {name.split('/')[-1]} -> Settings -> Visibility")
        return f"{type(exc).__name__}: {exc}"
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--user", required=True, help="your Docker Hub username")
    ap.add_argument("--tag", required=True, help="a version, e.g. 1.0 — see WHY A VERSION TAG")
    ap.add_argument("--only", nargs="+", choices=sorted(COMPONENTS),
                    help="publish only these components")
    args = ap.parse_args()
    check_docker()

    chosen = args.only or list(COMPONENTS)
    results = {}
    for component in chosen:
        folder, repo, flag, _ = COMPONENTS[component]
        ref = f"{args.user}/{repo}:{args.tag}"
        if not publish(folder, flag, ref):
            results[component] = (ref, "build or push failed — see the output above")
            continue
        print(f"  checking an anonymous pull of {ref} (linux/arm64) ...", flush=True)
        results[component] = (ref, pullable(ref))

    print("\n=== summary ===")
    for component, (ref, problem) in results.items():
        print(f"  {'OK    ' if not problem else 'FAILED'}  {ref}" + (f"\n          {problem}" if problem else ""))

    if any(problem for _, problem in results.values()):
        raise SystemExit("\nsome images are not usable by students yet — fix the above and "
                         "re-run with --only <component>")

    print("\nstudents deploy with (no Docker, no Node):")
    for folder, flags in student_commands(results).items():
        print(f"  cd {folder} && python deploy.py " + " ".join(flags))


def student_commands(results: dict) -> dict[str, list[str]]:
    """One command per folder, in deploy order: the web app's backend and
    frontend images go on the same deploy.py line."""
    commands: dict[str, list[str]] = {}
    for component, (ref, _) in results.items():
        folder, _, _, student_flag = COMPONENTS[component]
        commands.setdefault(folder, []).append(f"{student_flag} {ref}")
    return commands


if __name__ == "__main__":
    main()
