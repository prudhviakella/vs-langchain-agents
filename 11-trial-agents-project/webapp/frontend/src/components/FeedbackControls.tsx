// Thumbs up / down on one answer, with an optional comment. Resubmitting replaces.
import { useState } from "react";
import { useGiveFeedback } from "../api/hooks";
import type { Message } from "../api/types";
import { Button, cx } from "./ui";

export function FeedbackControls({ message }: { message: Message }) {
  const give = useGiveFeedback();
  const current = message.feedback;
  const [rating, setRating] = useState<"up" | "down" | null>(null);
  const [comment, setComment] = useState("");

  const send = (r: "up" | "down", text: string) =>
    give.mutate({ conversation_id: message.conversation_id, message_id: message.message_id,
                  rating: r, comment: text }, { onSuccess: () => { setRating(null); setComment(""); } });

  return (
    <div className="mt-2 text-xs">
      <div className="flex items-center gap-1">
        {(["up", "down"] as const).map((r) => (
          <button key={r} aria-label={r === "up" ? "Helpful" : "Not helpful"} onClick={() => setRating(r)}
            className={cx("rounded px-1.5 py-0.5 hover:bg-canvas", current?.rating === r && "bg-skyTint")}>
            {r === "up" ? "👍" : "👎"}</button>))}
        {current && <span className="text-faint">You rated this {current.rating === "up" ? "helpful" : "not helpful"}.</span>}
      </div>
      {rating && (
        <div className="mt-1 flex gap-2">
          <input className="flex-1 rounded border border-line px-2 py-1" placeholder="Optional comment"
            value={comment} onChange={(e) => setComment(e.target.value)} />
          <Button onClick={() => send(rating, comment)} disabled={give.isPending}>Send</Button>
          <Button variant="ghost" onClick={() => setRating(null)}>Cancel</Button>
        </div>)}
    </div>
  );
}
