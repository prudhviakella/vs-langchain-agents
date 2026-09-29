// Every rating I have given, newest first, each linking back to its conversation.
import { Link } from "react-router-dom";
import { useMyFeedback } from "../api/hooks";
import { Badge, Card, Spinner } from "../components/ui";
import { fmtWhen } from "../lib/format";

export function MyFeedbackScreen() {
  const { data = [], isLoading } = useMyFeedback();
  return (
    <div className="h-screen flex-1 overflow-y-auto px-6 py-4">
      <div className="mx-auto max-w-3xl space-y-3">
        <h1 className="text-lg font-semibold text-ink">My feedback</h1>
        {isLoading && <Spinner />}
        {!isLoading && data.length === 0 && <p className="text-sm text-muted">You have not rated any answers yet.</p>}
        {data.map((f) => (
          <Card key={`${f.conversation_id}-${f.message_id}`}>
            <div className="flex items-center gap-2 text-xs text-muted">
              <Badge tone={f.rating === "up" ? "ok" : "bad"}>{f.rating === "up" ? "Helpful" : "Not helpful"}</Badge>
              <span>{fmtWhen(f.created_at)}</span>
              <Link to={`/c/${f.conversation_id}`} className="ml-auto text-primary hover:underline">Open conversation</Link>
            </div>
            <p className="mt-2 text-sm font-medium text-ink">{f.question}</p>
            <p className="mt-1 text-sm text-muted">{f.answer_snippet}…</p>
            {f.comment && <p className="mt-2 text-sm">“{f.comment}”</p>}
          </Card>))}
      </div>
    </div>
  );
}
