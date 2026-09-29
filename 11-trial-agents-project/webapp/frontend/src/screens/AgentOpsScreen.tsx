// AgentOps: token usage, cost, latency, agents and tools used, feedback, and a
// trace link per interaction. Mine — or everyone's for ADMIN_USERS — over a window.
import { useState } from "react";
import { useAllFeedback, useInteractions, useUsage } from "../api/hooks";
import { Badge, Card, Spinner, Tabs, cx } from "../components/ui";
import { AGENT_LABEL, TOOL_LABEL, fmtInt, fmtMs, fmtUsd, fmtWhen } from "../lib/format";

const WINDOWS = [{ days: 1, label: "Today" }, { days: 7, label: "7 days" }, { days: 30, label: "30 days" },
                 { days: 90, label: "90 days" }];

export function AgentOpsScreen() {
  const [days, setDays] = useState(7);
  const [tab, setTab] = useState<"interactions" | "feedback">("interactions");
  const usage = useUsage(days);
  const interactions = useInteractions(days);
  const feedback = useAllFeedback(days);
  const u = usage.data;

  return (
    <div className="h-screen flex-1 overflow-y-auto px-6 py-4">
      <div className="mx-auto max-w-6xl space-y-4">
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-semibold text-ink">AgentOps</h1>
          {u && <Badge tone="info">{u.scope === "everyone" ? "All users" : "My usage"}</Badge>}
          <div className="ml-auto flex gap-1">
            {WINDOWS.map((w) => (
              <button key={w.days} onClick={() => setDays(w.days)} className={cx("rounded-lg px-2.5 py-1 text-sm",
                days === w.days ? "bg-primary text-white" : "text-body hover:bg-white")}>{w.label}</button>))}
          </div>
        </div>

        {usage.isLoading && <Spinner />}
        {u && (
          <>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-6">
              <Stat label="Interactions" value={fmtInt(u.interactions)} sub={u.errors ? `${u.errors} failed` : undefined} />
              <Stat label="Total tokens" value={fmtInt(u.total_tokens)}
                sub={`${fmtInt(u.cached_input_tokens)} cached input`} />
              <Stat label="Cost (estimate)" value={fmtUsd(u.cost_usd)} />
              <Stat label="Latency p50" value={fmtMs(u.latency_p50_ms)} />
              <Stat label="Latency p95" value={fmtMs(u.latency_p95_ms)} />
              <Stat label="Feedback" value={`👍 ${u.feedback_up} · 👎 ${u.feedback_down}`} />
            </div>
            <div className="grid gap-3 md:grid-cols-2">
              <Counts title="Agents invoked" counts={u.agents} labels={AGENT_LABEL} />
              <Counts title="Memory tools" counts={u.tools} labels={TOOL_LABEL} />
            </div>
          </>)}

        <Card>
          <Tabs tabs={[{ id: "interactions" as const, label: "Interactions", count: interactions.data?.length },
                       { id: "feedback" as const, label: "Feedback", count: feedback.data?.length }]}
                active={tab} onChange={setTab} />
          <div className="overflow-x-auto pt-2">
            {tab === "interactions" && (
              <table className="min-w-full text-left text-sm">
                <thead className="text-xs text-muted"><tr>
                  {["When", "User", "Question", "Agents / tools", "Tokens", "Cost", "Latency", "Trace"].map((h) =>
                    <th key={h} className="border-b border-line px-2 py-1.5 font-medium">{h}</th>)}</tr></thead>
                <tbody>{(interactions.data ?? []).map((r) => (
                  <tr key={r.message_id} className="align-top">
                    <td className="whitespace-nowrap border-b border-line px-2 py-1.5">{fmtWhen(r.created_at)}</td>
                    <td className="border-b border-line px-2 py-1.5">{r.username}</td>
                    <td className="max-w-xs border-b border-line px-2 py-1.5">
                      {r.status !== "complete" && <Badge tone="bad">failed</Badge>} {r.question}</td>
                    <td className="border-b border-line px-2 py-1.5 text-xs">
                      {[...r.agents.map((a) => AGENT_LABEL[a] ?? a), ...r.tools.map((t) => TOOL_LABEL[t] ?? t)].join(", ") || "—"}</td>
                    <td className="border-b border-line px-2 py-1.5">{fmtInt(r.total_tokens)}</td>
                    <td className="border-b border-line px-2 py-1.5">{fmtUsd(r.cost_usd)}</td>
                    <td className="border-b border-line px-2 py-1.5">{fmtMs(r.latency_ms)}</td>
                    <td className="border-b border-line px-2 py-1.5">{r.trace_url
                      ? <a className="text-primary hover:underline" href={r.trace_url} target="_blank" rel="noreferrer">Trace ↗</a> : "—"}</td>
                  </tr>))}</tbody>
              </table>)}
            {tab === "feedback" && ((feedback.data ?? []).length === 0
              ? <p className="py-2 text-sm text-muted">No feedback in this window.</p>
              : <ul className="divide-y divide-line">{(feedback.data ?? []).map((f) => (
                  <li key={`${f.message_id}-${f.username}`} className="py-2 text-sm">
                    <Badge tone={f.rating === "up" ? "ok" : "bad"}>{f.rating}</Badge>{" "}
                    <span className="text-muted">{f.username} · {fmtWhen(f.created_at)}</span>
                    <p className="mt-1 text-ink">{f.question}</p>
                    {f.comment && <p className="text-muted">“{f.comment}”</p>}
                  </li>))}</ul>)}
          </div>
        </Card>
      </div>
    </div>
  );
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return <Card><p className="text-xs text-muted">{label}</p><p className="mt-1 text-lg font-semibold text-ink">{value}</p>
    {sub && <p className="text-xs text-faint">{sub}</p>}</Card>;
}

function Counts({ title, counts, labels }: { title: string; counts: Record<string, number>; labels: Record<string, string> }) {
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const max = Math.max(1, ...entries.map(([, n]) => n));
  return (
    <Card>
      <p className="mb-2 text-sm font-medium text-ink">{title}</p>
      {entries.length === 0 ? <p className="text-sm text-muted">None in this window.</p> :
        entries.map(([k, n]) => (
          <div key={k} className="mb-1 flex items-center gap-2 text-sm">
            <span className="w-44 truncate">{labels[k] ?? k}</span>
            <div className="h-2 flex-1 rounded bg-canvas"><div className="h-2 rounded bg-primary" style={{ width: `${(n / max) * 100}%` }} /></div>
            <span className="w-8 text-right text-muted">{n}</span>
          </div>))}
    </Card>
  );
}
