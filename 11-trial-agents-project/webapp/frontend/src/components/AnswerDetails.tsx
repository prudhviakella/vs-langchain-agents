// What sits under an answer: how it was produced, and the evidence.
//
//   strip (always)   agents badges · latency · tokens · cost · "Details"
//   tabs (on open)   How it was answered  each agent's question + the supervisor's
//                                         stated rationale; memory tools; resolved
//                                         question; note
//                    Queries              the Cypher trial_graph ran; search stats
//                    Citations            best re-ranked first, with doc and page
//                    Data                 the table or graph
//                    Memory               what was recalled about the user
//                    Usage                tokens, calls, model, cost, trace link
//
// The rationale is text the supervisor wrote for the analyst as a tool
// argument. The model's private reasoning is never in the response, so it
// cannot be shown here.
import { lazy, Suspense, useState } from "react";
import type { AnswerDetails as Details } from "../api/types";
import { AGENT_LABEL, TOOL_LABEL, fmtInt, fmtMs, fmtUsd } from "../lib/format";
import { TableView } from "./TableView";
import { Badge, Spinner, Tabs } from "./ui";

const GraphView = lazy(() => import("./GraphView"));
type Tab = "how" | "queries" | "citations" | "data" | "memory" | "usage";

export function AnswerDetails({ details }: { details: Details }) {
  const [open, setOpen] = useState(false);
  const [tab, setTab] = useState<Tab>("how");
  const queries = details.agents.filter((a) => a.query || a.stats || a.searches?.length);
  const tabs = [
    { id: "how" as const, label: "How it was answered", count: details.agents.length + details.tools.length },
    { id: "queries" as const, label: "Queries", count: queries.length },
    { id: "citations" as const, label: "Citations", count: details.citations.length },
    ...(details.artifact ? [{ id: "data" as const, label: "Data" }] : []),
    { id: "memory" as const, label: "Memory", count: details.memory.length },
    { id: "usage" as const, label: "Usage" },
  ];

  return (
    <div className="mt-3 border-t border-line pt-2">
      <div className="flex flex-wrap items-center gap-2 text-xs text-muted">
        {details.agents.map((a, i) => (
          <Badge key={i} tone={a.succeeded ? "info" : "bad"}>{AGENT_LABEL[a.agent] ?? a.agent}</Badge>))}
        {details.tools.map((t, i) => <Badge key={`t${i}`}>{TOOL_LABEL[t.tool] ?? t.tool}</Badge>)}
        <span>{fmtMs(details.latency_ms)}</span>
        <span>{fmtInt(details.usage.total_tokens)} tokens</span>
        <span>{fmtUsd(details.cost_usd)}</span>
        <button className="ml-auto text-primary hover:underline" onClick={() => setOpen(!open)}>
          {open ? "Hide details" : "Details"}</button>
      </div>

      {open && (
        <div className="mt-2">
          <Tabs tabs={tabs} active={tab} onChange={setTab} />
          <div className="pt-3 text-sm">
            {tab === "how" && <How details={details} />}
            {tab === "queries" && (queries.length === 0 ? <Empty text="No queries were run for this answer." /> :
              queries.map((a, i) => (
                <div key={i} className="mb-3">
                  <p className="font-medium text-ink">{AGENT_LABEL[a.agent] ?? a.agent}</p>
                  {a.query && <pre className="mt-1 overflow-x-auto rounded-lg bg-canvas p-2 font-mono text-xs">{a.query}</pre>}
                  {a.searches && a.searches.length > 0 && (
                    <ol className="mt-1 space-y-1">{a.searches.map((q, j) => (
                      <li key={j} className="rounded-lg bg-canvas p-2">
                        <p className="font-mono text-xs text-ink">{q.query}</p>
                        <p className="mt-1 flex flex-wrap items-center gap-1.5 text-xs text-muted">
                          {q.doc_id ? <Badge tone="info">in {q.doc_id}</Badge> : <Badge>all 20 protocols</Badge>}
                          {q.content_type && <Badge>{q.content_type} only</Badge>}
                          {q.succeeded
                            ? <span>{q.candidates} candidates → {q.results} kept</span>
                            : <Badge tone="bad">failed</Badge>}
                          {q.succeeded && q.reranked !== null &&
                            <Badge tone={q.reranked ? "ok" : "neutral"}>{q.reranked ? "re-ranked" : "vector order"}</Badge>}
                        </p>
                      </li>))}</ol>)}
                  {a.stats && <p className="mt-1 text-xs text-muted">{Object.entries(a.stats)
                    .map(([k, v]) => `${k.replace(/_/g, " ")}: ${v}`).join(" · ")}</p>}
                </div>)))}
            {tab === "citations" && (details.citations.length === 0 ? <Empty text="No protocol passages were used." /> :
              <ol className="space-y-2">{details.citations.map((c) => (
                <li key={c.n} className="rounded-lg border border-line p-2">
                  <p className="text-xs text-muted">
                    <span className="font-medium text-ink">[{c.n}] {c.doc_id}</span>
                    {c.page !== null && ` · page ${c.page}`}
                    {c.headings.length > 0 && ` · ${c.headings.join(" › ")}`}
                    {c.rerank_score !== null && ` · relevance ${c.rerank_score.toFixed(2)}`}
                  </p>
                  <p className="mt-1 whitespace-pre-line">{c.snippet}</p>
                </li>))}</ol>)}
            {tab === "data" && details.artifact && (details.artifact.kind === "table"
              ? <TableView artifact={details.artifact} />
              : <Suspense fallback={<Spinner />}><GraphView artifact={details.artifact} /></Suspense>)}
            {tab === "memory" && (details.memory.length === 0 ? <Empty text="No memories were recalled." /> :
              <ul className="space-y-1">{details.memory.map((m, i) => (
                <li key={i}><Badge>{m.kind}</Badge> {m.text}
                  <span className="text-xs text-faint"> · {m.created_at.slice(0, 10)}</span></li>))}</ul>)}
            {tab === "usage" && <Usage details={details} />}
          </div>
        </div>
      )}
    </div>
  );
}

function How({ details }: { details: Details }) {
  const d = details.decision;
  return (
    <div className="space-y-3">
      {d.resolved_question && <p className="text-muted">Understood as: <span className="text-ink">{d.resolved_question}</span></p>}
      {d.from_conversation && <p className="text-muted">Answered from the earlier turns of this conversation.</p>}
      <ol className="space-y-2">
        {details.agents.map((a, i) => (
          <li key={i} className="rounded-lg border border-line p-2">
            <p className="font-medium text-ink">{i + 1}. {AGENT_LABEL[a.agent] ?? a.agent}{" "}
              <Badge tone={a.succeeded ? "ok" : "bad"}>{a.succeeded ? a.result_shape : "failed"}</Badge></p>
            <p className="mt-1"><span className="text-muted">Asked: </span>{a.question}</p>
            <p className="mt-1"><span className="text-muted">Why: </span>{a.rationale}</p>
          </li>))}
        {details.tools.map((t, i) => (
          <li key={`t${i}`} className="rounded-lg border border-line p-2">
            <p className="font-medium text-ink">{TOOL_LABEL[t.tool] ?? t.tool}{" "}
              <Badge tone={t.succeeded ? "ok" : "bad"}>{t.succeeded ? "done" : "unavailable"}</Badge></p>
            <p className="mt-1 text-muted">{Object.values(t.args).filter(Boolean).join(" · ")}</p>
          </li>))}
      </ol>
      {details.agents.length === 0 && details.tools.length === 0 &&
        <Empty text="No agent or tool was needed for this answer." />}
      {d.note && <p className="text-muted">Note: {d.note}</p>}
    </div>
  );
}

function Usage({ details }: { details: Details }) {
  const u = details.usage;
  const rows: [string, string][] = [
    ["Input tokens", fmtInt(u.input_tokens)], ["Cached input tokens", fmtInt(u.cached_input_tokens)],
    ["Output tokens", fmtInt(u.output_tokens)], ["Model calls", fmtInt(u.llm_calls)],
    ["Model", u.model_id ?? "—"], ["Latency", fmtMs(details.latency_ms)],
    ["Cost (estimate)", fmtUsd(details.cost_usd)], ["Earlier messages sent", fmtInt(details.history_turns)],
  ];
  return (
    <div>
      <dl className="grid grid-cols-2 gap-x-6 gap-y-1 sm:grid-cols-4">
        {rows.map(([k, v]) => (<div key={k}><dt className="text-xs text-muted">{k}</dt><dd className="text-ink">{v}</dd></div>))}
      </dl>
      {details.trace_url && <a className="mt-3 inline-block text-primary hover:underline" href={details.trace_url}
        target="_blank" rel="noreferrer">Open the trace in CloudWatch ↗</a>}
    </div>
  );
}

const Empty = ({ text }: { text: string }) => <p className="text-muted">{text}</p>;
