// Frontend behaviour against a mocked backend (fetch), in jsdom.
//   stream parser     frames split across network chunks
//   login             unknown username -> first/last name -> account created
//   answer panel      agents + rationale, the Cypher, citations best-first, trace link
//   answer text       markdown-light rendering; HTML in an answer stays text
//   full turn         question -> progress -> answer, then the saved answer shows
import { QueryClient } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { App } from "../App";
import { parseEvents } from "../api/sse";
import type { AnswerDetails as Details, Message } from "../api/types";
import { AnswerDetails } from "../components/AnswerDetails";
import { AnswerText } from "../components/AnswerText";
import { LoginScreen } from "../screens/LoginScreen";

const json = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

const DETAILS: Details = {
  agents: [{ agent: "trial_graph", question: "Resolve IMbrave150", rationale: "resolve the trial to its document",
             result_shape: "table", succeeded: true, query: "MATCH (t:Trial) RETURN t.nctId", stats: null },
           { agent: "trial_search", question: "exclusion criteria", rationale: "protocol text",
             result_shape: "passages", succeeded: true, query: null, stats: { search_calls: 2 },
             searches: [{ query: "exclusion criteria", doc_id: "nct03434379-hepatocellular-atezo-bev",
                          content_type: null, top_k: 8, candidates: 40, results: 8, reranked: true, succeeded: true },
                        { query: "dose table", doc_id: null, content_type: "table_summary", top_k: 8,
                          candidates: 0, results: 0, reranked: null, succeeded: false }] }],
  tools: [{ tool: "recall_facts", args: { query: "format" }, succeeded: true, result: "", items: 1 }],
  citations: [{ n: 1, doc_id: "nct03434379", page: 56, headings: ["4.1.2"], snippet: "best passage", rerank_score: 0.9, origin: "search" },
              { n: 2, doc_id: "nct03434379", page: 54, headings: [], snippet: "weaker passage", rerank_score: 0.4, origin: "search" }],
  artifact: null, memory: [{ kind: "semantic", text: "Prefers tables.", created_at: "2026-09-20T10:00:00Z", score: 0.93 }],
  decision: { resolved_question: "Exclusion criteria of IMbrave150 (NCT03434379)?" },
  usage: { input_tokens: 12000, cached_input_tokens: 10000, output_tokens: 500, total_tokens: 12500, llm_calls: 4, model_id: "gpt-6-sol" },
  cost_usd: 0.029, latency_ms: 28000, history_turns: 2, trace_id: "abc",
  trace_url: "https://us-east-1.console.aws.amazon.com/cloudwatch/home#xray:traces/1-abc",
};

afterEach(() => vi.restoreAllMocks());

test("SSE frames split across network chunks are reassembled", () => {
  const first = parseEvents('event: progress\ndata: {"elapsed_s": 5}\n\nevent: answer\ndata: {"mess');
  expect(first.events).toEqual([{ event: "progress", data: { elapsed_s: 5 } }]);
  const second = parseEvents(first.rest + 'age": {"text": "done"}}\n\n');
  expect(second.events).toEqual([{ event: "answer", data: { message: { text: "done" } } }]);
});

test("an unknown username is asked for first and last name, then created", async () => {
  const bodies: unknown[] = [];
  vi.spyOn(globalThis, "fetch").mockImplementation(async (_url, init) => {
    const body = JSON.parse(String(init?.body));
    bodies.push(body);
    return body.first_name
      ? json(200, { username: "new.user", first_name: "New", last_name: "User", is_admin: false })
      : json(404, { detail: { code: "new_user", message: "new username" } });
  });
  const signedIn = vi.fn();
  render(<LoginScreen onSignedIn={signedIn} />);
  await userEvent.type(screen.getByLabelText("Username"), "new.user");
  await userEvent.type(screen.getByLabelText("Password"), "longenough1");
  await userEvent.click(screen.getByRole("button", { name: "Sign in" }));

  await userEvent.type(await screen.findByLabelText("First name"), "New");
  await userEvent.type(screen.getByLabelText("Last name"), "User");
  await userEvent.click(screen.getByRole("button", { name: "Create account" }));
  await waitFor(() => expect(signedIn).toHaveBeenCalledWith(expect.objectContaining({ username: "new.user" })));
  expect(bodies[1]).toMatchObject({ username: "new.user", first_name: "New", last_name: "User" });
});

test("a wrong password says so", async () => {
  vi.spyOn(globalThis, "fetch").mockResolvedValue(json(401, { detail: "wrong username or password" }));
  render(<LoginScreen onSignedIn={vi.fn()} />);
  await userEvent.type(screen.getByLabelText("Username"), "prudhvi");
  await userEvent.type(screen.getByLabelText("Password"), "wrong-pass");
  await userEvent.click(screen.getByRole("button", { name: "Sign in" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Wrong username or password.");
});

test("the answer panel shows agents with rationale, the query, citations and the trace", async () => {
  render(<AnswerDetails details={DETAILS} />);
  expect(screen.getByText("Registry graph")).toBeInTheDocument();
  await userEvent.click(screen.getByText("Details"));
  expect(screen.getByText("resolve the trial to its document")).toBeInTheDocument();
  expect(screen.getByText(/Exclusion criteria of IMbrave150/)).toBeInTheDocument();

  await userEvent.click(screen.getByRole("tab", { name: /Queries/ }));
  expect(screen.getByText("MATCH (t:Trial) RETURN t.nctId")).toBeInTheDocument();
  expect(screen.getByText("search calls: 2")).toBeInTheDocument();
  expect(screen.getByText("exclusion criteria")).toBeInTheDocument();
  expect(screen.getByText("in nct03434379-hepatocellular-atezo-bev")).toBeInTheDocument();
  expect(screen.getByText("40 candidates → 8 kept")).toBeInTheDocument();
  expect(screen.getByText("re-ranked")).toBeInTheDocument();
  expect(screen.getByText("table_summary only")).toBeInTheDocument();
  expect(screen.getByText("failed")).toBeInTheDocument();

  await userEvent.click(screen.getByRole("tab", { name: /Citations/ }));
  const items = screen.getAllByRole("listitem");
  expect(items[0]).toHaveTextContent("best passage");
  expect(items[0]).toHaveTextContent("relevance 0.90");

  await userEvent.click(screen.getByRole("tab", { name: /Memory/ }));
  expect(screen.getByText(/Prefers tables\./)).toBeInTheDocument();

  await userEvent.click(screen.getByRole("tab", { name: /Usage/ }));
  expect(screen.getByRole("link", { name: /Open the trace/ })).toHaveAttribute("href", DETAILS.trace_url);
});

test("answer text renders bold and bullets, and HTML stays text", () => {
  const { container } = render(<AnswerText text={"**Lead** line\n- one\n- two\n<img src=x onerror=alert(1)>"} />);
  expect(container.querySelector("strong")).toHaveTextContent("Lead");
  expect(container.querySelectorAll("li")).toHaveLength(2);
  expect(container.querySelector("img")).toBeNull();
  expect(container).toHaveTextContent("<img src=x onerror=alert(1)>");
});

test("a full turn: question, progress, then the saved answer with feedback controls", async () => {
  const conv = { conversation_id: "c1", title: "IMbrave150?", created_at: "2026-09-27T10:00:00Z",
                 updated_at: "2026-09-27T10:00:00Z", message_count: 2 };
  const question: Message = { message_id: "u1", conversation_id: "c1", role: "user", text: "IMbrave150?",
                              created_at: "2026-09-27T10:00:00Z", status: "complete", details: null, feedback: null };
  const answer: Message = { ...question, message_id: "a1", role: "assistant", text: "**Excluded:** hepatic encephalopathy",
                            details: DETAILS };
  let started = false, saved = false;
  vi.spyOn(globalThis, "fetch").mockImplementation(async (url) => {
    const path = String(url);
    if (path === "/api/auth/me") return json(200, { username: "prudhvi", first_name: "Prudhvi", last_name: "A", is_admin: true });
    // the backend creates the conversation as the chat request starts
    if (path.startsWith("/api/conversations?")) return json(200, started ? [conv] : []);
    if (path === "/api/conversations/c1/messages") return json(200, saved ? [question, answer] : [question]);
    if (path === "/api/chat") {
      started = true;
      // Like the backend: the answer is saved only when its event is sent, so
      // the screen can show it only by refetching after that event.
      const send = (c: ReadableStreamDefaultController, f: string) => c.enqueue(new TextEncoder().encode(f));
      const stream = new ReadableStream({ async start(c) {
        send(c, `event: conversation\ndata: ${JSON.stringify({ conversation_id: "c1", user_message: question })}\n\n`);
        send(c, `event: progress\ndata: {"elapsed_s": 5}\n\n`);
        await new Promise((r) => setTimeout(r, 150));          // the agents at work
        saved = true;
        send(c, `event: answer\ndata: ${JSON.stringify({ message: answer })}\n\n`);
        c.close();
      } });
      return new Response(stream, { status: 200, headers: { "Content-Type": "text/event-stream" } });
    }
    return json(404, { detail: "unexpected " + path });
  });

  render(<App client={new QueryClient({ defaultOptions: { queries: { retry: false } } })} />);
  const box = await screen.findByLabelText("Question");
  fireEvent.change(box, { target: { value: "IMbrave150?" } });
  fireEvent.submit(box.closest("form")!);

  expect(await screen.findByText(/The agents are working/)).toBeInTheDocument();
  expect(await screen.findByText("hepatic encephalopathy", { exact: false })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Helpful" })).toBeInTheDocument();
  expect(await screen.findByText("IMbrave150?", { selector: "a" })).toBeInTheDocument();   // sidebar
});


test("a markdown table in an answer renders as a table — the PIONEER 4 / STEP 1 comparison", () => {
  const text = [
    "The key difference is the **control group**.",
    "| | PIONEER 4 (NCT02863419) | STEP 1 (NCT03548935) |",
    "|---|---|---|",
    "| **Population** | Type 2 diabetes | BMI ≥ 30 kg/m² |",
    "| **Design** | 52-week, **2:2:1** | 68-week, 2:1 |",
    "Their endpoints should not be compared directly.",
  ].join("\n");
  const { container } = render(<AnswerText text={text} />);
  const table = container.querySelector("table")!;
  expect(table).not.toBeNull();
  expect(table.querySelectorAll("thead th")).toHaveLength(3);
  expect(table.querySelectorAll("tbody tr")).toHaveLength(2);           // separator row dropped
  expect(table.querySelector("tbody td strong")).toHaveTextContent("Population");
  expect(container).not.toHaveTextContent("|---|");
  expect(container.querySelectorAll("p")).toHaveLength(2);              // text before and after
});

test("pipes without a separator row stay text", () => {
  const { container } = render(<AnswerText text={"Ratio | 2:1 | allocation"} />);
  expect(container.querySelector("table")).toBeNull();
});
