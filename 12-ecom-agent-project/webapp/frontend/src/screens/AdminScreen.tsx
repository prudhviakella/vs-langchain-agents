// The admin console (AgentOps) — read-only views over what the app already stores.
//
//   Usage           GET /admin/usage?days=      totals · answers per day (chart) · by agent · by user
//   Feedback        GET /admin/feedback?month=   each 👍/👎 with its question, answer excerpt, comment
//   Knowledge base  GET /admin/kb?q=&namespace=  the published KB records (Pinecone ecom-kb)
//                   GET /admin/kb/search?q=      by meaning, with the grounding embedding model
//
// What this does NOT do: change anything. Cost shows "—" until MODEL_PRICES is set on the backend.
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api/client";
import { ChartView } from "../components/artifacts/ChartView";
import { Button, cx, Loading } from "../components/ui/primitives";

type Tab = "usage" | "feedback" | "kb";

export function AdminScreen() {
  const [tab, setTab] = useState<Tab>("usage");
  const navigate = useNavigate();
  return (
    <div className="min-h-screen bg-canvas text-body">
      <header className="flex items-center gap-4 border-b border-line bg-card px-6 py-3">
        <button className="text-sm text-muted hover:text-ink" onClick={() => navigate("/")}>← Conversations</button>
        <h1 className="text-lg font-bold text-ink">Admin</h1>
        <nav className="ml-4 flex gap-1">
          {([["usage", "Usage"], ["feedback", "Feedback"], ["kb", "Knowledge base"]] as [Tab, string][]).map(([id, label]) => (
            <button key={id} onClick={() => setTab(id)}
                    className={cx("rounded-lg px-3 py-1.5 text-sm", tab === id ? "bg-brand-50 font-semibold text-ink" : "text-muted hover:text-ink")}>
              {label}
            </button>
          ))}
        </nav>
      </header>
      <main className="mx-auto max-w-6xl px-6 py-6">
        {tab === "usage" && <Usage />}
        {tab === "feedback" && <FeedbackHub />}
        {tab === "kb" && <KnowledgeBase />}
      </main>
    </div>
  );
}

const fmt = (n?: number | null) => (n === null || n === undefined ? "—" : n.toLocaleString());
const usd = (n?: number | null) => (n === null || n === undefined ? "—" : `$${n.toFixed(n < 1 ? 4 : 2)}`);
const ms = (n?: number | null) => (n === null || n === undefined ? "—" : `${(n / 1000).toFixed(1)} s`);

function Kpi({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-line bg-card px-4 py-3">
      <div className="text-[11px] uppercase tracking-wide text-faint">{label}</div>
      <div className="mt-1 text-xl font-bold text-ink">{value}</div>
    </div>
  );
}

function Table({ cols, rows }: { cols: string[]; rows: (string | number)[][] }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-line bg-card">
      <table className="w-full text-sm">
        <thead className="bg-canvas text-left text-[11px] uppercase text-faint">
          <tr>{cols.map((c) => <th key={c} className="px-3 py-2 font-semibold">{c}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-t border-line">{r.map((v, j) => <td key={j} className="px-3 py-2">{v}</td>)}</tr>
          ))}
          {rows.length === 0 && <tr><td className="px-3 py-4 text-muted" colSpan={cols.length}>Nothing in this range.</td></tr>}
        </tbody>
      </table>
    </div>
  );
}

// ── usage ────────────────────────────────────────────────────────────────
type UsageResp = {
  days: number; priced: boolean; model: string;
  totals: { interactions: number; users: number; failed: number; total_tokens: number; input_tokens: number;
            output_tokens: number; cost_usd: number | null; avg_latency_ms: number | null; p95_latency_ms: number | null };
  by_day: { day: string; interactions: number; tokens: number; cost_usd: number | null }[];
  by_agent: { agent: string; calls: number; input_tokens: number; output_tokens: number; total_tokens: number; cost_usd: number | null }[];
  by_user: { user_id: string; interactions: number; tokens: number; cost_usd: number | null }[];
};

export function Usage() {
  const [days, setDays] = useState(7);
  const q = useQuery({ queryKey: ["admin-usage", days], queryFn: () => api<UsageResp>("/admin/usage", { query: { days: String(days) } }) });
  return (
    <div className="space-y-5">
      <div className="flex gap-2">
        {[7, 30, 90].map((d) => (
          <Button key={d} variant={d === days ? "primary" : "subtle"} onClick={() => setDays(d)}>{d} days</Button>
        ))}
      </div>
      {q.isLoading && <Loading />}
      {q.data && (
        <>
          {!q.data.priced && (
            <p className="text-xs text-faint">Cost shows "—": no price is set for {q.data.model} (MODEL_PRICES on the backend).</p>
          )}
          <div className="grid grid-cols-2 gap-3 md:grid-cols-6">
            <Kpi label="Answers" value={fmt(q.data.totals.interactions)} />
            <Kpi label="Users" value={fmt(q.data.totals.users)} />
            <Kpi label="Tokens" value={fmt(q.data.totals.total_tokens)} />
            <Kpi label="Cost" value={usd(q.data.totals.cost_usd)} />
            <Kpi label="Latency avg / p95" value={`${ms(q.data.totals.avg_latency_ms)} / ${ms(q.data.totals.p95_latency_ms)}`} />
            <Kpi label="Failed" value={fmt(q.data.totals.failed)} />
          </div>
          <ChartView chart={{ kind: "chart", spec: {
            data: [{ type: "bar", x: q.data.by_day.map((d) => d.day), y: q.data.by_day.map((d) => d.interactions),
                     marker: { color: "#2563EB" }, name: "answers" }],
            layout: { title: { text: "Answers per day" }, yaxis: { title: { text: "answers" } } } } }} />
          <h3 className="text-sm font-semibold text-ink">By agent</h3>
          <Table cols={["Agent", "Calls", "Input tokens", "Output tokens", "Total tokens", "Cost"]}
                 rows={q.data.by_agent.map((a) => [a.agent, fmt(a.calls), fmt(a.input_tokens), fmt(a.output_tokens), fmt(a.total_tokens), usd(a.cost_usd)])} />
          <h3 className="text-sm font-semibold text-ink">By user</h3>
          <Table cols={["User", "Answers", "Tokens", "Cost"]}
                 rows={q.data.by_user.map((u) => [u.user_id, fmt(u.interactions), fmt(u.tokens), usd(u.cost_usd)])} />
        </>
      )}
    </div>
  );
}

// ── feedback ─────────────────────────────────────────────────────────────
type FbItem = { conversation_id: string; message_id: string; user_id: string; user_name?: string; vote?: "up" | "down" | null;
                comment?: string | null; at_ns: number; question?: string; answer?: string; conversation_title?: string };

function shiftMonth(m: string, by: number) {
  const [y, mo] = m.split("-").map(Number);
  const d = new Date(Date.UTC(y, mo - 1 + by, 1));
  return `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, "0")}`;
}

export function FeedbackHub() {
  const now = new Date();
  const [month, setMonth] = useState(`${now.getUTCFullYear()}-${String(now.getUTCMonth() + 1).padStart(2, "0")}`);
  const [vote, setVote] = useState("");
  const q = useQuery({ queryKey: ["admin-feedback", month, vote],
                       queryFn: () => api<{ items: FbItem[]; likes: number; dislikes: number; comments: number }>(
                         "/admin/feedback", { query: { month, vote } }) });
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <Button onClick={() => setMonth(shiftMonth(month, -1))}>‹</Button>
        <span className="w-20 text-center text-sm font-semibold text-ink">{month}</span>
        <Button onClick={() => setMonth(shiftMonth(month, 1))}>›</Button>
        {[["", "All"], ["up", "👍"], ["down", "👎"]].map(([v, l]) => (
          <Button key={v} variant={v === vote ? "primary" : "subtle"} onClick={() => setVote(v)}>{l}</Button>
        ))}
        {q.data && <span className="text-sm text-muted">{q.data.likes} 👍 · {q.data.dislikes} 👎 · {q.data.comments} comments</span>}
      </div>
      {q.isLoading && <Loading />}
      {q.data?.items.length === 0 && <p className="text-sm text-muted">No feedback this month.</p>}
      {q.data?.items.map((f) => (
        <div key={f.message_id + f.user_id} className="rounded-xl border border-line bg-card px-4 py-3 text-sm">
          <div className="flex items-center gap-2 text-xs text-faint">
            <span className="text-base">{f.vote === "up" ? "👍" : f.vote === "down" ? "👎" : "💬"}</span>
            <span>{f.user_name ?? f.user_id}</span>
            <span>· {new Date(f.at_ns / 1e6).toLocaleString()}</span>
            <Link className="ml-auto text-brand hover:underline" to={`/conversation/${f.conversation_id}`}>open ↗</Link>
          </div>
          {f.question && <p className="mt-2 font-semibold text-ink">{f.question}</p>}
          {f.answer && <p className="mt-1 text-body">{f.answer}</p>}
          {f.comment && <p className="mt-2 rounded-lg bg-canvas px-3 py-2 italic">“{f.comment}”</p>}
        </div>
      ))}
    </div>
  );
}

// ── knowledge base ───────────────────────────────────────────────────────
type KbRecord = { id: string; namespace: string; type: string; title: string; text: string;
                  metadata: Record<string, unknown>; score?: number };

export function KnowledgeBase() {
  const [input, setInput] = useState("");
  const [q, setQ] = useState("");
  const [mode, setMode] = useState<"keyword" | "meaning">("keyword");
  const [ns, setNs] = useState("");
  const res = useQuery({
    queryKey: ["admin-kb", mode, q, ns],
    queryFn: () => mode === "meaning" && q
      ? api<{ records: KbRecord[] }>("/admin/kb/search", { query: { q } })
      : api<{ records: KbRecord[]; total: number; counts: Record<string, number> }>("/admin/kb", { query: { q, namespace: ns } }),
  });
  const counts = (res.data as { counts?: Record<string, number> } | undefined)?.counts;
  return (
    <div className="space-y-4">
      <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); setQ(input.trim()); }}>
        <input value={input} onChange={(e) => setInput(e.target.value)} placeholder="Search tables, columns, terms, examples…"
               className="flex-1 rounded-lg border border-line bg-card px-3 py-2 text-sm" />
        <select value={mode} onChange={(e) => setMode(e.target.value as "keyword" | "meaning")}
                className="rounded-lg border border-line bg-card px-2 text-sm">
          <option value="keyword">keyword</option>
          <option value="meaning">by meaning</option>
        </select>
        <Button variant="primary" type="submit">Search</Button>
      </form>
      {counts && (
        <div className="flex flex-wrap gap-2">
          {[["", "all"], ...Object.keys(counts).map((k) => [k, k])].map(([k, l]) => (
            <button key={k} onClick={() => setNs(k)}
                    className={cx("rounded-full border px-3 py-1 text-xs", ns === k ? "border-brand bg-brand-50 text-ink" : "border-line")}>
              {l}{k ? ` · ${counts[k]}` : ""}
            </button>
          ))}
        </div>
      )}
      {res.isLoading && <Loading />}
      {res.error && <p className="text-sm text-flag-crit">{(res.error as Error).message}</p>}
      <div className="space-y-2">
        {res.data?.records.map((r) => <KbRow key={r.namespace + r.id} r={r} />)}
      </div>
    </div>
  );
}

function KbRow({ r }: { r: KbRecord }) {
  const [open, setOpen] = useState(false);
  const query = typeof r.metadata.query === "string" ? r.metadata.query : "";
  return (
    <div className="rounded-xl border border-line bg-card px-4 py-2 text-sm">
      <button className="flex w-full items-baseline gap-2 text-left" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        <span className="rounded-sm bg-canvas px-1.5 py-0.5 font-mono text-[10px] text-muted">{r.type}</span>
        <span className="font-semibold text-ink">{r.title}</span>
        <span className="ml-auto text-[11px] text-faint">{r.namespace}{r.score !== undefined ? ` · ${r.score}` : ""}</span>
      </button>
      {open && (
        <div className="mt-2 space-y-2 text-xs">
          {r.text && <p className="whitespace-pre-wrap text-body">{r.text}</p>}
          {query && <pre className="overflow-x-auto rounded-md bg-ink px-3 py-2 font-mono text-[11px] text-white">{query}</pre>}
          <pre className="max-h-64 overflow-auto rounded-md bg-canvas px-3 py-2 font-mono text-[10px] text-muted">
            {JSON.stringify(r.metadata, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
