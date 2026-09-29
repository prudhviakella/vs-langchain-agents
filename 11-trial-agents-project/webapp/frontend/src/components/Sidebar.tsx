// Left column: new conversation, search, my conversations (rename, delete),
// navigation, and the signed-in user with Sign out.
import { useState } from "react";
import { NavLink, useMatch, useNavigate } from "react-router-dom";
import { useConversations, useDeleteConversation, useRenameConversation } from "../api/hooks";
import type { User } from "../api/types";
import { Button, Input, cx } from "./ui";

export function Sidebar({ user, onSignOut }: { user: User; onSignOut: () => void }) {
  const [q, setQ] = useState("");
  const { data: conversations = [], isLoading } = useConversations(q);
  const rename = useRenameConversation();
  const remove = useDeleteConversation();
  const navigate = useNavigate();
  // useMatch, not useParams: the sidebar sits outside <Routes>, where
  // useParams is always empty.
  const conversationId = useMatch("/c/:conversationId")?.params.conversationId;
  const [editing, setEditing] = useState<string | null>(null);
  const [title, setTitle] = useState("");

  const onDelete = (id: string, name: string) => {
    if (!window.confirm(`Delete "${name}" and all its messages?`)) return;
    remove.mutate(id, { onSuccess: () => { if (id === conversationId) navigate("/"); } });
  };

  return (
    <aside className="flex h-screen w-72 flex-col border-r border-line bg-white">
      <div className="p-3">
        <p className="mb-2 text-sm font-semibold text-ink">Trial Agents</p>
        <Button className="w-full justify-center" onClick={() => navigate("/")}>+ New conversation</Button>
        <Input className="mt-2" placeholder="Search conversations" value={q}
          onChange={(e) => setQ(e.target.value)} aria-label="Search conversations" />
      </div>
      <nav className="flex-1 overflow-y-auto px-2" aria-label="Conversations">
        {isLoading && <p className="px-2 text-sm text-muted">Loading…</p>}
        {!isLoading && conversations.length === 0 &&
          <p className="px-2 text-sm text-muted">{q ? "No conversation matches." : "No conversations yet."}</p>}
        {conversations.map((c) => (
          <div key={c.conversation_id} className={cx("group flex items-center rounded-lg px-2 py-1.5 text-sm",
            c.conversation_id === conversationId ? "bg-skyTint" : "hover:bg-canvas")}>
            {editing === c.conversation_id ? (
              <form className="flex-1" onSubmit={(e) => { e.preventDefault();
                rename.mutate({ id: c.conversation_id, title }); setEditing(null); }}>
                <input autoFocus className="w-full rounded border border-line px-1" value={title}
                  onChange={(e) => setTitle(e.target.value)} onBlur={() => setEditing(null)} aria-label="New title" />
              </form>
            ) : (
              <NavLink to={`/c/${c.conversation_id}`} className="flex-1 truncate text-body">{c.title}</NavLink>
            )}
            <button className="ml-1 hidden text-faint hover:text-ink group-hover:inline" aria-label={`Rename ${c.title}`}
              onClick={() => { setEditing(c.conversation_id); setTitle(c.title); }}>✎</button>
            <button className="ml-1 hidden text-faint hover:text-bad group-hover:inline" aria-label={`Delete ${c.title}`}
              onClick={() => onDelete(c.conversation_id, c.title)}>🗑</button>
          </div>))}
      </nav>
      <div className="border-t border-line p-3 text-sm">
        <NavLink to="/feedback" className="block rounded px-2 py-1 hover:bg-canvas">My feedback</NavLink>
        <NavLink to="/agentops" className="block rounded px-2 py-1 hover:bg-canvas">AgentOps</NavLink>
        <div className="mt-2 flex items-center justify-between px-2">
          <span className="truncate text-muted">{user.first_name} {user.last_name}</span>
          <Button variant="ghost" onClick={onSignOut}>Sign out</Button>
        </div>
      </div>
    </aside>
  );
}
