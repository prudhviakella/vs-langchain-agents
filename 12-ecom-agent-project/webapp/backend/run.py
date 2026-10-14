"""Run the backend locally: loads .env (this folder), then uvicorn on http://localhost:8000.

    cd webapp/backend
    python run.py
"""
import os
from pathlib import Path

import uvicorn

env = Path(__file__).resolve().parent / ".env"
if env.exists():
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
