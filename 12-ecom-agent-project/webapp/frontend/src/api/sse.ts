// Chat streams: POST /chat/message (and GET /chat/stream/{id} to resume) answer with
// Server-Sent Events. EventSource cannot POST, so the body is read as a stream and the
// "event:" / "data:" frames are parsed here (ACT's parser). ": ping" keepalive frames
// carry no data and are skipped.
import { API_PREFIX } from "../lib/config";

export type SseEvent = { event: string; data: any };

export async function streamChat(
  body: { text: string; conversation_id?: string },
  onEvent: (e: SseEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const res = await fetch(API_PREFIX + "/chat/message", {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  });
  await consume(res, onEvent);
}

export async function resumeChat(interactionId: string, onEvent: (e: SseEvent) => void, signal?: AbortSignal) {
  const res = await fetch(API_PREFIX + `/chat/stream/${interactionId}`, {
    credentials: "same-origin",
    headers: { Accept: "text/event-stream" },
    signal,
  });
  await consume(res, onEvent);
}

export async function stopTurn(interactionId: string): Promise<void> {
  await fetch(API_PREFIX + `/chat/${encodeURIComponent(interactionId)}/stop`, {
    method: "POST",
    credentials: "same-origin",
  }).catch(() => undefined);
}

async function consume(res: Response, onEvent: (e: SseEvent) => void): Promise<void> {
  if (!res.ok || !res.body) {
    const t = await res.text().catch(() => "");
    throw new Error(`chat stream failed (${res.status}): ${t}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const e = parseFrame(buf.slice(0, idx));
      buf = buf.slice(idx + 2);
      if (e) onEvent(e);
    }
  }
}

export function parseFrame(frame: string): SseEvent | null {
  let event = "message";
  const data: string[] = [];
  for (const raw of frame.split("\n")) {
    const line = raw.trimEnd();
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trim());
  }
  if (data.length === 0) return null;
  const joined = data.join("\n");
  try {
    return { event, data: JSON.parse(joined) };
  } catch {
    return { event, data: joined };
  }
}
