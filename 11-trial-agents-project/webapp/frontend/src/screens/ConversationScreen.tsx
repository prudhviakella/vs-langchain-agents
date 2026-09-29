// One conversation: its questions and answers, the turn in progress, and the composer.
//   /               a new conversation (created by the first question)
//   /c/:conversationId
import { useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { useMessages } from "../api/hooks";
import type { Message } from "../api/types";
import { AnswerDetails } from "../components/AnswerDetails";
import { AnswerText } from "../components/AnswerText";
import { FeedbackControls } from "../components/FeedbackControls";
import { Button, Card, Spinner } from "../components/ui";
import { useChat } from "../lib/chat";

const EXAMPLES = ["Which trials does Novo Nordisk sponsor?",
  "What are the exclusion criteria of the IMbrave150 trial?",
  "Remember that I focus on phase 3 trials."];

export function ConversationScreen() {
  const { conversationId } = useParams();
  const { data: messages = [], isLoading } = useMessages(conversationId);
  const { pending, send } = useChat();
  const [text, setText] = useState("");
  const bottom = useRef<HTMLDivElement>(null);
  // The turn in flight belongs to this screen when its conversation is this one
  // (or both are still new).
  const mine = pending && (pending.conversationId ?? null) === (conversationId ?? null) ? pending : null;
  const busy = !!pending && !pending.error;

  useEffect(() => { bottom.current?.scrollIntoView?.({ behavior: "smooth" }); }, [messages.length, mine?.elapsed]);

  const submit = (value: string) => {
    if (!value.trim() || busy) return;
    setText("");
    void send(value.trim(), conversationId ?? null);
  };

  return (
    <div className="flex h-screen flex-1 flex-col">
      <div className="flex-1 overflow-y-auto px-6 py-4">
        <div className="mx-auto max-w-3xl space-y-4">
          {!conversationId && !mine && (
            <div className="pt-16 text-center">
              <h2 className="text-lg font-semibold text-ink">Ask about the 20 clinical trials</h2>
              <p className="mt-1 text-sm text-muted">Sponsors, sites and phases from the registry; eligibility,
                endpoints and safety from the protocols.</p>
              <div className="mt-4 flex flex-wrap justify-center gap-2">
                {EXAMPLES.map((q) => <Button key={q} variant="ghost" className="border border-line"
                  onClick={() => submit(q)}>{q}</Button>)}
              </div>
            </div>)}
          {isLoading && conversationId && <Spinner />}
          {messages.map((m) => <Bubble key={m.message_id} message={m} />)}
          {mine && (
            <>
              {!messages.some((m) => m.message_id === mine.userMessageId) &&
                <div className="ml-auto max-w-[80%] rounded-xl2 bg-primary px-4 py-2 text-sm text-white">{mine.question}</div>}
              <Card>
                {mine.error ? <p className="text-sm text-bad" role="alert">{mine.error}</p> :
                  <p className="flex items-center gap-2 text-sm text-muted"><Spinner />
                    The agents are working… {mine.elapsed > 0 && `${mine.elapsed} s`}</p>}
              </Card>
            </>)}
          <div ref={bottom} />
        </div>
      </div>
      <form className="border-t border-line bg-white px-6 py-3" onSubmit={(e) => { e.preventDefault(); submit(text); }}>
        <div className="mx-auto flex max-w-3xl gap-2">
          <textarea className="flex-1 resize-none rounded-lg border border-line px-3 py-2 text-sm outline-none focus:border-primary"
            rows={2} placeholder="Ask a question…" aria-label="Question" value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(text); } }} />
          <Button type="submit" disabled={busy || !text.trim()}>Send</Button>
        </div>
      </form>
    </div>
  );
}

function Bubble({ message }: { message: Message }) {
  if (message.role === "user")
    return <div className="ml-auto max-w-[80%] rounded-xl2 bg-primary px-4 py-2 text-sm text-white">{message.text}</div>;
  return (
    <Card>
      {message.status === "error"
        ? <p className="text-sm text-bad">{message.text}</p>
        : <AnswerText text={message.text} />}
      {message.details && message.status === "complete" && <AnswerDetails details={message.details} />}
      {message.status === "complete" && <FeedbackControls message={message} />}
    </Card>
  );
}
