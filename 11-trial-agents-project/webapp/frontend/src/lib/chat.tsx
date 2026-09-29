// The turn in flight, held ABOVE the routes.
//
//   send(text, conversationId?)
//     └─ streamChat ── conversation ──► new conversation? navigate to /c/<id>
//                  ── progress ──────► elapsed seconds (the progress indicator)
//                  ── answer | error ► refetch messages; pending cleared
//
// Why above the routes: the first question of a new conversation creates it
// mid-stream and the URL changes from / to /c/<id>. State inside the screen
// could be lost to a remount; here it survives the navigation.
import { createContext, useContext, useState, type ReactNode } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { streamChat } from "../api/sse";

type Pending = { conversationId: string | null; question: string; elapsed: number;
                 userMessageId?: string; error?: string };
type Chat = { pending: Pending | null; send: (text: string, conversationId: string | null) => Promise<void> };

const ChatContext = createContext<Chat>({ pending: null, send: async () => {} });
export const useChat = () => useContext(ChatContext);

export function ChatProvider({ children }: { children: ReactNode }) {
  const [pending, setPending] = useState<Pending | null>(null);
  const qc = useQueryClient();
  const navigate = useNavigate();

  async function send(text: string, conversationId: string | null) {
    setPending({ conversationId, question: text, elapsed: 0 });
    try {
      await streamChat({ text, conversation_id: conversationId }, (e) => {
        if (e.event === "conversation") {
          setPending((p) => p && { ...p, conversationId: e.data.conversation_id,
                                          userMessageId: e.data.user_message.message_id });
          if (!conversationId) navigate(`/c/${e.data.conversation_id}`, { replace: true });
          qc.invalidateQueries({ queryKey: ["conversations"] });
        } else if (e.event === "progress") {
          setPending((p) => p && { ...p, elapsed: e.data.elapsed_s });
        } else {
          qc.invalidateQueries({ queryKey: ["messages", e.data.message.conversation_id] });
          qc.invalidateQueries({ queryKey: ["conversations"] });   // newest-activity order changed
          qc.invalidateQueries({ queryKey: ["agentops"] });
        }
      });
      setPending(null);
    } catch (err) {
      setPending((p) => p && { ...p, error: err instanceof Error ? err.message : "The request failed." });
    }
  }
  return <ChatContext.Provider value={{ pending, send }}>{children}</ChatContext.Provider>;
}
