# Trial agents — frontend

```
/login            username + password; a new username adds first + last name
/                 new conversation — the first question creates it
/c/:id            questions and answers; under each answer:
                    agents · latency · tokens · cost · Details
                    Details: How it was answered | Queries | Citations | Data | Memory | Usage
                    👍 / 👎 + comment
/feedback         everything I have rated
/agentops         tokens, cost, p50/p95 latency, agents, memory tools,
                  interactions with CloudWatch trace links, feedback
                  (mine; everyone's for ADMIN_USERS)
sidebar           new · search · rename · delete · Sign out
```

React 18 + Vite + TypeScript + Tailwind + React Query + React Router — the
reference app's stack and palette; Neo4j NVL draws graph answers (lazy-loaded).

## Run locally

```bash
npm install
npm run dev          # http://localhost:5173 — /api is proxied to localhost:8000
```

Start the backend first (see `../backend/README.md`).

## Test and build

```bash
npm test             # vitest + Testing Library, backend mocked
npm run build        # type check + production build into dist/
```

`.npmrc` sets `legacy-peer-deps=true`: `@neo4j-nvl/react` declares its React
peer as exactly 18.0.0 or ^19, which npm rejects for 18.3.1 although it runs
on it. The reference app carries the same setting.
