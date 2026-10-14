// The conversation screen — ACT's structure, trimmed to what this app shows.
//
//   ┌ Sidebar ─────────┐ ┌ Thread ──────────────────────────────────────────────┐
//   │ + New            │ │ UserBubble                                           │
//   │ search           │ │ AssistantTurn: summary · ▸ Reasoning · charts · table │
//   │ conversations    │ │                ▸ sources · 👍 👎                      │
//   │ user · sign out  │ │ LiveTurn (while streaming): status · Reasoning (open) │
//   └──────────────────┘ │                 · text as it streams · artifacts      │
//                        │ Composer                                             │
//                        └──────────────────────────────────────────────────────┘
//
// SSE events (backend turns.py) -> live state:
//   meta -> ids (navigate to the new conversation) · status -> status line · reasoning / token
//   -> appended · artifact -> table / chart · entities · sources · awaiting_input -> option chips
//   · done -> reload the saved conversation (the answer is saved BEFORE done) · error
//
// What this does NOT do: render the answer as HTML. Markdown-lite (bold, lists, code) is
// turned into React elements; nothing from a model reaches innerHTML.
import { Fragment, useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { useConversations, useDelete, useFeedback, useLatest, useMessages, useRename } from "../api/hooks";
import { resumeChat, stopTurn, streamChat, type SseEvent } from "../api/sse";
import type { Artifacts, Message, SourceRecord, Vote } from "../api/types";
import { ArtifactRenderer } from "../components/artifacts/ArtifactRenderer";
import { Avatar, Button, cx, Spinner } from "../components/ui/primitives";
import { useCurrentUser, useSignOut } from "../lib/auth";

type Live = {
  question: string;
  text: string;
  reasoning: string;
  status: string;
  artifacts: Artifacts;
  sources: SourceRecord[];
  options: string[];
  interactionId?: string;
};

const emptyLive = (question: string): Live => ({
  question, text: "", reasoning: "", status: "Starting…", artifacts: {}, sources: [], options: [],
});

export function ConversationScreen() {
  const { convId } = useParams();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const messages = useMessages(convId);
  const latest = useLatest(convId);
  const [live, setLive] = useState<Live | null>(null);
  const [error, setError] = useState("");
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  // one handler for a new turn AND a resumed one
  function handlerFor(convRef: { id?: string }) {
    return (e: SseEvent) => {
      const d = e.data ?? {};
      switch (e.event) {
        case "meta":
          convRef.id = d.conversation_id ?? convRef.id;
          setLive((p) => p && { ...p, interactionId: d.interaction_id });
          if (!convId && d.conversation_id) navigate(`/conversation/${d.conversation_id}`, { replace: true });
          break;
        case "status":
          setLive((p) => p && { ...p, status: `${d.agent ?? ""} · ${d.phase ?? ""}${d.detail ? ` — ${d.detail}` : ""}` });
          break;
        case "reasoning":
          setLive((p) => p && { ...p, reasoning: p.reasoning + (d.text ?? "") });
          break;
        case "token":
        case "clarification":
          setLive((p) => p && { ...p, text: p.text + (d.text ?? ""), status: "writing the answer" });
          break;
        case "artifact":
          setLive((p) => {
            if (!p) return p;
            const a = { ...p.artifacts };
            if (d.kind === "table") a.table = d.data;
            else if (d.kind === "chart") a.charts = [...(a.charts ?? []), d.data];
            return { ...p, artifacts: a };
          });
          break;
        case "sources":
          setLive((p) => p && { ...p, sources: d.sources ?? [] });
          break;
        case "awaiting_input":
          setLive((p) => p && { ...p, options: d.options ?? [] });
          break;
        case "done":
          qc.invalidateQueries({ queryKey: ["messages", convRef.id] });
          qc.invalidateQueries({ queryKey: ["latest", convRef.id] });
          qc.invalidateQueries({ queryKey: ["conversations"] });
          break;
        case "error":
          setError(String(d.detail ?? "The answer failed"));
          qc.invalidateQueries({ queryKey: ["messages", convRef.id] });
          break;
      }
    };
  }

  async function send(text: string) {
    const q = text.trim();
    if (!q || live) return;
    setError("");
    setLive(emptyLive(q));
    const ac = new AbortController();
    abortRef.current = ac;
    const convRef = { id: convId };
    try {
      await streamChat({ text: q, conversation_id: convId }, handlerFor(convRef), ac.signal);
      await qc.invalidateQueries({ queryKey: ["messages", convRef.id] });
    } catch (err) {
      if (!ac.signal.aborted) setError(err instanceof Error ? err.message : "The answer failed");
    } finally {
      setLive(null);
      abortRef.current = null;
    }
  }

  // a turn still running on the server (page reloaded mid-answer) -> resume it
  useEffect(() => {
    const l = latest.data;
    if (!convId || live || !l?.is_active || !l.interaction_id) return;
    const ac = new AbortController();
    abortRef.current = ac;
    setLive({ ...emptyLive(""), interactionId: l.interaction_id, status: "Resuming…" });
    resumeChat(l.interaction_id, handlerFor({ id: convId }), ac.signal)
      .catch(() => undefined)
      .finally(() => {
        setLive(null);
        qc.invalidateQueries({ queryKey: ["messages", convId] });
      });
    return () => ac.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [convId, latest.data?.interaction_id, latest.data?.is_active]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.data, live?.text, live?.reasoning, live?.artifacts]);

  async function stop() {
    if (live?.interactionId) await stopTurn(live.interactionId);
    abortRef.current?.abort();
  }

  const items = messages.data?.items ?? [];
  // the live turn's own messages are already saved (user + placeholder): hide them until done
  const shown = live?.interactionId ? items.filter((m) => m.interaction_id !== live.interactionId) : items;

  return (
    <div className="flex h-screen bg-canvas text-body">
      <Sidebar activeId={convId} />
      <main className="flex min-w-0 flex-1 flex-col">
        <div className="flex-1 overflow-y-auto">
          <div className="mx-auto max-w-4xl px-6 py-8">
            {!convId && !live && <Welcome onAsk={send} />}
            {shown.map((m) =>
              m.role === "user" ? <UserBubble key={m.message_id} text={m.text} />
                : <AssistantTurn key={m.message_id} m={m} onOption={send} />)}
            {live && (
              <>
                {live.question && <UserBubble text={live.question} />}
                <LiveTurn live={live} />
              </>
            )}
            {error && <p className="mt-4 rounded-lg bg-flag-critbg px-3 py-2 text-sm text-flag-crit">{error}</p>}
            <div ref={bottomRef} />
          </div>
        </div>
        <Composer busy={!!live} onSend={send} onStop={stop} />
      </main>
    </div>
  );
}

// ── sidebar ────────────────────────────────────────────────────────────────
function Sidebar({ activeId }: { activeId?: string }) {
  const navigate = useNavigate();
  const [q, setQ] = useState("");
  const convs = useConversations(q.trim());
  const rename = useRename();
  const del = useDelete();
  const { user } = useCurrentUser();
  const signOut = useSignOut();
  return (
    <aside className="flex w-72 shrink-0 flex-col border-r border-line bg-card">
      <div className="space-y-2 p-3">
        <Button variant="primary" className="w-full" onClick={() => navigate("/")}>+ New conversation</Button>
        <input className="w-full rounded-lg border border-line bg-canvas px-3 py-1.5 text-sm" placeholder="Search"
               value={q} onChange={(e) => setQ(e.target.value)} />
      </div>
      <nav className="flex-1 overflow-y-auto px-2">
        {(convs.data?.conversations ?? []).map((c) => (
          <div key={c.conversation_id}
               className={cx("group flex items-center rounded-lg px-2 py-1.5 text-sm",
                             c.conversation_id === activeId ? "bg-brand-50 text-ink" : "hover:bg-canvas")}>
            <button className="min-w-0 flex-1 truncate text-left" title={c.title}
                    onClick={() => navigate(`/conversation/${c.conversation_id}`)}>{c.title}</button>
            <button className="ml-1 hidden text-xs text-faint hover:text-ink group-hover:block" title="Rename"
                    onClick={() => {
                      const t = window.prompt("Rename conversation", c.title);
                      if (t?.trim()) rename.mutate({ id: c.conversation_id, title: t.trim() });
                    }}>✎</button>
            <button className="ml-1 hidden text-xs text-faint hover:text-flag-crit group-hover:block" title="Delete"
                    onClick={() => {
                      if (window.confirm(`Delete "${c.title}"?`)) {
                        del.mutate(c.conversation_id);
                        if (c.conversation_id === activeId) navigate("/");
                      }
                    }}>✕</button>
          </div>
        ))}
      </nav>
      <div className="flex items-center gap-2 border-t border-line p-3 text-sm">
        {user && <Avatar name={user.name} />}
        <span className="min-w-0 flex-1 truncate">{user?.name}</span>
        {user?.role === "Admin" && (
          <button className="text-xs text-muted hover:text-ink" onClick={() => navigate("/admin")}>Admin</button>
        )}
        <button className="text-xs text-muted hover:text-ink" onClick={signOut}>Sign out</button>
      </div>
    </aside>
  );
}

function Welcome({ onAsk }: { onAsk: (q: string) => void }) {
  const examples = ["Which carrier delivers on time most often?", "What is revenue by month in 2024?",
                    "Top 10 products by revenue", "Plot orders by status"];
  return (
    <div className="py-16 text-center">
      <h2 className="text-2xl font-bold text-ink">Ask about the store's data</h2>
      <p className="mt-2 text-sm text-muted">Orders, revenue, customers, products, delivery, payments, stock and reviews.</p>
      <div className="mt-6 flex flex-wrap justify-center gap-2">
        {examples.map((e) => (
          <button key={e} onClick={() => onAsk(e)}
                  className="rounded-full border border-line bg-card px-3 py-1.5 text-sm hover:border-brand hover:text-ink">{e}</button>
        ))}
      </div>
    </div>
  );
}

// ── one turn ───────────────────────────────────────────────────────────────
function UserBubble({ text }: { text: string }) {
  return (
    <div className="mb-4 flex justify-end">
      <div className="max-w-[80%] whitespace-pre-wrap rounded-2xl bg-brand px-4 py-2.5 text-sm text-white">{text}</div>
    </div>
  );
}

export function AssistantTurn({ m, onOption }: { m: Message; onOption: (q: string) => void }) {
  const failed = m.status === "failed" || m.status === "stopped";
  return (
    <div className="mb-8 rounded-2xl border border-line bg-card px-5 py-4">
      <AnswerText text={m.text} />
      {failed && <p className="mt-2 text-xs text-flag-high">{m.status === "stopped" ? "Stopped." : "This answer did not complete."}</p>}
      {m.options && m.options.length > 0 && <OptionChips options={m.options} onPick={onOption} />}
      {m.reasoning && <Reasoning text={m.reasoning} />}
      <ArtifactRenderer artifacts={m.artifacts} />
      {m.truncation_note && <p className="mt-2 text-xs text-faint">{m.truncation_note}</p>}
      <Sources sources={m.sources} />
      {!failed && <Feedback m={m} />}
    </div>
  );
}

function LiveTurn({ live }: { live: Live }) {
  return (
    <div className="mb-8 rounded-2xl border border-brand/30 bg-card px-5 py-4">
      <div className="mb-2 flex items-center gap-2 text-xs text-muted">
        <Spinner className="h-3.5 w-3.5" />
        <span>{live.status}</span>
      </div>
      {live.text ? <AnswerText text={live.text} /> : null}
      {live.options.length > 0 && <OptionChips options={live.options} onPick={() => undefined} />}
      {live.reasoning && <Reasoning text={live.reasoning} defaultOpen />}
      <ArtifactRenderer artifacts={live.artifacts} />
      <Sources sources={live.sources} />
    </div>
  );
}

function OptionChips({ options, onPick }: { options: string[]; onPick: (q: string) => void }) {
  return (
    <div className="mt-3 flex flex-wrap gap-2">
      {options.map((o) => (
        <button key={o} onClick={() => onPick(o)}
                className="rounded-full border border-line px-3 py-1 text-sm hover:border-brand hover:text-ink">{o}</button>
      ))}
    </div>
  );
}

// ── reasoning: collapsible; ```sql fences shown as code ──────────────────────
export function Reasoning({ text, defaultOpen = false }: { text: string; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  const parts = text.split(/```(?:sql)?\n?/);
  return (
    <div className="mt-3">
      <button onClick={() => setOpen((o) => !o)} aria-expanded={open}
              className="inline-flex items-center gap-1.5 text-xs font-semibold text-muted hover:text-ink">
        <span className={cx("transition-transform", open ? "rotate-90" : "")}>▸</span> Reasoning
      </button>
      {open && (
        <div className="mt-2 space-y-2 rounded-lg border border-line bg-canvas/70 px-3 py-2 text-xs leading-relaxed text-body">
          {parts.map((p, i) =>
            i % 2 === 1 ? (
              <pre key={i} className="whitespace-pre-wrap wrap-break-word rounded-md bg-ink px-3 py-2 font-mono text-[11px] leading-relaxed text-white">
                {formatSql(p.trim())}
              </pre>
            ) : (
              <p key={i} className="whitespace-pre-wrap">{p.replace(/^\n+|\n+$/g, "")}</p>
            ))}
        </div>
      )}
    </div>
  );
}

/** One main clause per line, for reading — display only (the SQL that ran is unchanged).
 *  String literals are left alone, so a keyword inside quotes never moves. */
export function formatSql(sql: string): string {
  const clause = /\s+(FROM|LEFT JOIN|RIGHT JOIN|INNER JOIN|FULL JOIN|CROSS JOIN|JOIN|WHERE|GROUP BY|HAVING|ORDER BY|LIMIT|UNION ALL|UNION)\b/gi;
  return sql
    .split(/('(?:[^']|'')*')/)
    .map((part, i) => (i % 2 === 1 ? part : part.replace(clause, (_m, kw: string) => `\n${kw.toUpperCase()}`)))
    .join("")
    .replace(/\n{2,}/g, "\n");
}

// ── sources: ACT's citation block — collapsed, last, the query one level deeper ─
export function Sources({ sources }: { sources?: SourceRecord[] | null }) {
  const [open, setOpen] = useState(false);
  const cited = (sources ?? []).filter((s) => (s.sources?.length ?? 0) > 0 || s.store);
  if (cited.length === 0) return null;
  return (
    <div className="mt-3 border-t border-line/70 pt-2">
      <button onClick={() => setOpen((o) => !o)} aria-expanded={open}
              className="inline-flex items-center gap-2 text-xs text-faint hover:text-muted">
        <span>{cited.length} source{cited.length === 1 ? "" : "s"}</span>
        <span className={cx("transition-transform", open ? "rotate-90" : "")}>▸</span>
      </button>
      {open && (
        <div className="mt-2 space-y-1.5 rounded-lg border border-line bg-canvas/70 px-3 py-2">
          {cited.map((s, i) => <SourceRow key={i} s={s} />)}
        </div>
      )}
    </div>
  );
}

function SourceRow({ s }: { s: SourceRecord }) {
  const [showQuery, setShowQuery] = useState(false);
  return (
    <div className="text-[11px] leading-relaxed">
      <div className="flex flex-wrap items-baseline gap-x-2">
        <span className="font-semibold text-ink">{s.agent}</span>
        <span className="font-mono text-muted">{s.sources?.length ? s.sources.join(", ") : s.store}</span>
        {typeof s.row_count === "number" && <span className="text-faint">{s.row_count.toLocaleString()} rows</span>}
        {s.query && (
          <button className="text-faint underline-offset-2 hover:text-muted hover:underline"
                  onClick={() => setShowQuery((v) => !v)}>{showQuery ? "hide" : "show"} {s.query_language || "query"}</button>
        )}
      </div>
      {showQuery && (
        <pre className="mt-1 overflow-x-auto rounded-md bg-ink px-3 py-2 font-mono text-[11px] text-white">{s.query}</pre>
      )}
    </div>
  );
}

function Feedback({ m }: { m: Message }) {
  // A thumb is saved at once; the comment box then opens. On 👎 it asks what was wrong —
  // that is what makes a 👎 actionable in the admin Feedback tab.
  const fb = useFeedback();
  const saved = m.feedback;
  const [vote, setVote] = useState<Vote | null>(saved?.my_vote ?? null);
  const [comment, setComment] = useState(saved?.my_comment ?? "");
  const [open, setOpen] = useState(false);
  const [sent, setSent] = useState(false);
  const base = { conversation_id: m.conversation_id, message_id: m.message_id };

  function pick(v: Vote) {
    setVote(v);
    setSent(false);
    setOpen(true);
    fb.mutate({ ...base, vote: v, comment: comment || undefined });
  }
  function send() {
    fb.mutate({ ...base, vote: vote ?? undefined, comment: comment.trim() || undefined },
              { onSuccess: () => { setSent(true); setOpen(false); } });
  }
  return (
    <div className="mt-3 text-sm">
      <div className="flex items-center gap-1">
        {(["up", "down"] as Vote[]).map((v) => (
          <button key={v} onClick={() => pick(v)} aria-pressed={vote === v} title={v === "up" ? "Helpful" : "Not helpful"}
                  className={cx("rounded-md px-1.5 py-0.5", vote === v ? "bg-brand-50" : "opacity-60 hover:opacity-100")}>
            {v === "up" ? "👍" : "👎"}
          </button>
        ))}
        {!open && (saved?.my_comment || sent) && comment && (
          <button className="ml-2 truncate text-xs text-muted hover:text-ink" onClick={() => setOpen(true)} title="Edit comment">
            Your comment: “{comment}” ✎
          </button>
        )}
      </div>
      {open && (
        <div className="mt-2 flex items-end gap-2">
          <textarea autoFocus rows={2} value={comment} onChange={(e) => setComment(e.target.value)} maxLength={2000}
                    placeholder={vote === "down" ? "What was wrong or missing?" : "Anything to add? (optional)"}
                    className="flex-1 resize-none rounded-lg border border-line bg-canvas px-3 py-2 text-sm focus:border-brand focus:outline-hidden" />
          <Button variant="primary" onClick={send} disabled={fb.isPending}>Send</Button>
          <Button variant="ghost" onClick={() => setOpen(false)}>Close</Button>
        </div>
      )}
      {sent && !open && <p className="mt-1 text-xs text-faint">Thanks — feedback saved.</p>}
    </div>
  );
}

// ── composer ───────────────────────────────────────────────────────────────
function Composer({ busy, onSend, onStop }: { busy: boolean; onSend: (q: string) => void; onStop: () => void }) {
  const [text, setText] = useState("");
  const submit = () => {
    if (!text.trim() || busy) return;
    onSend(text);
    setText("");
  };
  return (
    <div className="border-t border-line bg-card px-6 py-4">
      <div className="mx-auto flex max-w-4xl items-end gap-2">
        <textarea rows={2} value={text} onChange={(e) => setText(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } }}
                  placeholder="Ask about orders, revenue, customers, delivery…"
                  className="min-h-[44px] flex-1 resize-none rounded-xl border border-line bg-canvas px-3 py-2 text-sm focus:border-brand focus:outline-hidden" />
        {busy ? <Button variant="danger" onClick={onStop}>Stop</Button>
              : <Button variant="primary" onClick={submit} disabled={!text.trim()}>Send</Button>}
      </div>
    </div>
  );
}

// ── markdown-lite: **bold**, `code`, - / 1. lists, paragraphs — as React elements ─
export function AnswerText({ text }: { text: string }) {
  const blocks = text.split(/\n{2,}/);
  return (
    <div className="space-y-2 text-[15px] leading-relaxed text-ink">
      {blocks.map((b, i) => {
        const lines = b.split("\n");
        if (lines.every((l) => /^\s*([-*]|\d+\.)\s+/.test(l))) {
          const ordered = /^\s*\d+\./.test(lines[0]);
          const List = ordered ? "ol" : "ul";
          return (
            <List key={i} className={cx("space-y-1 pl-5", ordered ? "list-decimal" : "list-disc")}>
              {lines.map((l, j) => <li key={j}>{inline(l.replace(/^\s*([-*]|\d+\.)\s+/, ""))}</li>)}
            </List>
          );
        }
        return <p key={i}>{lines.map((l, j) => <Fragment key={j}>{j > 0 && <br />}{inline(l)}</Fragment>)}</p>;
      })}
    </div>
  );
}

function inline(s: string) {
  return s.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) return <strong key={i}>{part.slice(2, -2)}</strong>;
    if (part.startsWith("`") && part.endsWith("`"))
      return <code key={i} className="rounded-sm bg-canvas px-1 font-mono text-[13px]">{part.slice(1, -1)}</code>;
    return <Fragment key={i}>{part}</Fragment>;
  });
}
