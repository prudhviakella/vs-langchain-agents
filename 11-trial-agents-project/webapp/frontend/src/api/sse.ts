// One chat turn: POST /api/chat, read the Server-Sent Events off the response
// body. fetch, not EventSource: EventSource cannot send a POST body. (The
// reference's streamChat/consumeSseResponse, with the cookie instead of a token.)
//
//   conversation  {conversation_id, user_message}   at once
//   progress      {elapsed_s}                       every 5 s while the agents work
//   answer        {message}                         the saved answer
//   error         {message}                         the saved failure

import { ApiError } from "./client";
import type { Message } from "./types";

export type ChatEvent =
  | { event: "conversation"; data: { conversation_id: string; user_message: Message } }
  | { event: "progress"; data: { elapsed_s: number } }
  | { event: "answer" | "error"; data: { message: Message } };

export function parseEvents(buffer: string): { events: ChatEvent[]; rest: string } {
  const blocks = buffer.split("\n\n");
  const rest = blocks.pop() ?? "";
  const events: ChatEvent[] = [];
  for (const block of blocks) {
    let name = "message", data = "";
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) name = line.slice(6).trim();
      else if (line.startsWith("data:")) data += line.slice(5).trim();
    }
    if (data) events.push({ event: name, data: JSON.parse(data) } as ChatEvent);
  }
  return { events, rest };
}

export async function streamChat(body: { text: string; conversation_id?: string | null },
                                 onEvent: (e: ChatEvent) => void, signal?: AbortSignal): Promise<void> {
  const res = await fetch("/api/chat", {
    method: "POST", credentials: "same-origin", signal,
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
  });
  if (!res.ok || !res.body) {
    const payload = await res.json().catch(() => null);
    throw new ApiError(res.status, payload?.detail ?? payload);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parsed = parseEvents(buffer);
    buffer = parsed.rest;
    parsed.events.forEach(onEvent);
  }
}
