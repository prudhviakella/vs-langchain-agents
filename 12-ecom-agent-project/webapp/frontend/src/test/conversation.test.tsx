// Frontend: the SSE parser, and a saved answer showing all five parts — summary, reasoning,
// chart, table, citation — with nothing from the model rendered as HTML.
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, describe, expect, it, vi } from "vitest";
import { parseFrame } from "../api/sse";
import type { Message } from "../api/types";
import { AnswerText, AssistantTurn, formatSql, Reasoning, Sources } from "../screens/ConversationScreen";
import { ArtifactRenderer } from "../components/artifacts/ArtifactRenderer";

vi.mock("plotly.js-dist-min", () => ({ default: { newPlot: vi.fn(), purge: vi.fn(), Plots: { resize: vi.fn() } } }));

describe("SSE frames", () => {
  it("parses a named event with JSON data", () => {
    expect(parseFrame('event: token\ndata: {"text":"Hi"}')).toEqual({ event: "token", data: { text: "Hi" } });
  });
  it("ignores keepalive comments", () => {
    expect(parseFrame(": ping")).toBeNull();
  });
});

const answer: Message = {
  message_id: "m1", conversation_id: "c1", interaction_id: "i1", role: "assistant", msg_kind: "answer",
  text: "**OnTrac** delivers on time most often: **41.6%**.\n\n- FedEx 40.5%\n- UPS 39.8%",
  reasoning: "Plan: compare carriers.\nSQL (nlq):\n```sql\nSELECT carrier FROM ecom.shipments\n```\n→ 5 rows\n",
  artifacts: {
    table: { columns: ["carrier", "on_time_rate_pct"], rows: [["OnTrac", 41.6], ["FedEx", 40.5]] },
    charts: [{ kind: "chart", chart_type: "bar", insight: "OnTrac leads",
               spec: { data: [{ type: "bar", x: ["OnTrac", "FedEx"], y: [41.6, 40.5] }], layout: { title: { text: "On time" } } } }],
  },
  sources: [{ agent: "nlq", store: "PostgreSQL ecom", query_language: "SQL", query: "SELECT carrier FROM ecom.shipments",
              question: "q", row_count: 5, sources: ["ecom.shipments"], filters: [] }],
  entities: [], created_at_ns: 1, status: "complete",
};

const wrap = (ui: React.ReactNode) =>
  render(<QueryClientProvider client={new QueryClient()}>{ui}</QueryClientProvider>);

describe("a saved answer", () => {
  it("summary: bold and list rendered as elements", () => {
    wrap(<AnswerText text={answer.text} />);
    expect(screen.getByText("OnTrac").tagName).toBe("STRONG");
    expect(screen.getByText("FedEx 40.5%").tagName).toBe("LI");
  });

  it("reasoning: collapsed, then the SQL as a code block", () => {
    wrap(<Reasoning text={answer.reasoning!} />);
    expect(screen.queryByText(/SELECT carrier/)).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /Reasoning/ }));
    expect(screen.getByText("SELECT carrier FROM ecom.shipments").tagName).toBe("PRE");
    expect(screen.getByText(/Plan: compare carriers/)).toBeInTheDocument();
  });

  it("table and chart render from the stored artifacts", () => {
    const { container } = wrap(<ArtifactRenderer artifacts={answer.artifacts} />);
    expect(screen.getAllByText("OnTrac").length).toBeGreaterThan(0);       // table cell
    expect(container.textContent).toMatch(/On time|Chart|OnTrac leads/);     // chart block
  });

  it("citation: tables and rows, the query one click deeper", () => {
    wrap(<Sources sources={answer.sources} />);
    fireEvent.click(screen.getByRole("button", { name: /1 source/ }));
    expect(screen.getByText("ecom.shipments")).toBeInTheDocument();
    expect(screen.getByText("5 rows")).toBeInTheDocument();
    expect(screen.queryByText("SELECT carrier FROM ecom.shipments")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /show SQL/ }));
    expect(screen.getByText("SELECT carrier FROM ecom.shipments")).toBeInTheDocument();
  });

  it("model text is never HTML", () => {
    const { container } = wrap(<AnswerText text={'<img src=x onerror="alert(1)"> **ok**'} />);
    expect(container.querySelector("img")).toBeNull();
    expect(container.textContent).toContain('<img src=x onerror="alert(1)">');
  });
});


describe("reasoning SQL", () => {
  it("one clause per line; quoted keywords stay put", () => {
    expect(formatSql("SELECT a FROM ecom.orders o JOIN ecom.order_items i ON i.order_id = o.order_id WHERE note = 'x from y' GROUP BY a ORDER BY a LIMIT 5"))
      .toBe("SELECT a\nFROM ecom.orders o\nJOIN ecom.order_items i ON i.order_id = o.order_id\nWHERE note = 'x from y'\nGROUP BY a\nORDER BY a\nLIMIT 5");
  });
});

describe("feedback", () => {
  afterEach(() => vi.unstubAllGlobals());
  it("👎 saves the vote at once, then asks what was wrong and sends the comment", async () => {
    const calls: any[] = [];
    vi.stubGlobal("fetch", vi.fn(async (_url: string, init?: RequestInit) => {
      calls.push(JSON.parse(String(init?.body ?? "{}")));
      return new Response(null, { status: 204 });
    }));
    wrap(<AssistantTurn m={answer} onOption={() => undefined} />);
    fireEvent.click(screen.getByTitle("Not helpful"));
    await waitFor(() => expect(calls[0]).toEqual({ conversation_id: "c1", message_id: "m1", vote: "down" }));
    const box = screen.getByPlaceholderText("What was wrong or missing?");
    fireEvent.change(box, { target: { value: "wrong period — I meant 2024" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(calls[1]).toEqual({ conversation_id: "c1", message_id: "m1", vote: "down",
                                                  comment: "wrong period — I meant 2024" }));
    await waitFor(() => expect(screen.getByText(/Your comment: “wrong period — I meant 2024”/)).toBeInTheDocument());
  });
});
