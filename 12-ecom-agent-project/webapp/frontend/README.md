# Web frontend — React + Vite, ported from ACT's frontend

```
browser :5173 ──/api──► Vite proxy ──► backend :8000 ──► AgentCore supervisor
  sign in · sidebar · thread · composer
  each answer:  summary · ▸ Reasoning (live while it runs) · chart(s) · table · ▸ sources · 👍 👎
```

ACT's theme, primitives, table and Plotly chart views, and SSE reader; the conversation screen
trimmed to this app. Locally, Vite proxies `/api` to the backend; on AWS, CloudFront does.

## Run locally (PyCharm Terminal, project root) — the backend must be running

```bash
cd webapp/frontend
npm install --legacy-peer-deps
npm run dev                      # http://localhost:5173
```

`--legacy-peer-deps`: npm 10's peer resolution crashes on this tree ("reading 'edgesOut'"); the
committed package-lock.json pins the versions that build and pass the tests.

## Check

```bash
cd webapp/frontend
npm run typecheck
npm test
npm run build                    # -> dist/ (uploaded to S3 on AWS)
```
