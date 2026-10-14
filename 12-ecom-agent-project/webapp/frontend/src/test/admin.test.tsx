// Admin console: usage KPIs (and "—" without a price), the feedback hub, a KB record's SQL.
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { FeedbackHub, KnowledgeBase, Usage } from "../screens/AdminScreen";

vi.mock("plotly.js-dist-min", () => ({ default: { newPlot: vi.fn(), purge: vi.fn(), Plots: { resize: vi.fn() } } }));

function mockApi(routes: Record<string, unknown>) {
  vi.stubGlobal("fetch", vi.fn(async (url: string) => {
    const key = Object.keys(routes).find((k) => url.includes(k))!;
    return new Response(JSON.stringify(routes[key]), { status: 200, headers: { "Content-Type": "application/json" } });
  }));
}

const wrap = (ui: React.ReactNode) =>
  render(<QueryClientProvider client={new QueryClient()}><MemoryRouter>{ui}</MemoryRouter></QueryClientProvider>);

afterEach(() => vi.unstubAllGlobals());

describe("admin", () => {
  it("usage: KPIs, and — for cost without a price", async () => {
    mockApi({ "/admin/usage": {
      days: 7, priced: false, model: "gpt-6-sol",
      totals: { interactions: 12, users: 3, failed: 1, total_tokens: 241200, input_tokens: 228000, output_tokens: 13200,
                cost_usd: null, avg_latency_ms: 30500, p95_latency_ms: 52000 },
      by_day: [{ day: "2026-10-05", interactions: 12, tokens: 241200, cost_usd: null }],
      by_agent: [{ agent: "nlq", calls: 12, input_tokens: 216000, output_tokens: 12000, total_tokens: 228000, cost_usd: null }],
      by_user: [{ user_id: "prudhvi", interactions: 9, tokens: 180900, cost_usd: null }] } });
    wrap(<Usage />);
    await waitFor(() => expect(screen.getByText("241,200")).toBeInTheDocument());
    expect(screen.getByText("30.5 s / 52.0 s")).toBeInTheDocument();
    expect(screen.getByText(/no price is set for gpt-6-sol/)).toBeInTheDocument();
    expect(screen.getByText("nlq")).toBeInTheDocument();
    expect(screen.getByText("prudhvi")).toBeInTheDocument();
  });

  it("feedback: vote, question, answer, comment, link", async () => {
    mockApi({ "/admin/feedback": { likes: 0, dislikes: 1, comments: 1, items: [{
      conversation_id: "c9", message_id: "m1", user_id: "prudhvi", user_name: "Prudhvi A", vote: "down",
      comment: "wrong period", at_ns: 1_790_000_000_000_000_000, question: "Revenue last month?", answer: "Revenue was $5.1M." }] } });
    wrap(<FeedbackHub />);
    await waitFor(() => expect(screen.getByText("Revenue last month?")).toBeInTheDocument());
    expect(screen.getByText("Revenue was $5.1M.")).toBeInTheDocument();
    expect(screen.getByText("“wrong period”")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /open/ })).toHaveAttribute("href", "/conversation/c9");
  });

  it("knowledge base: an example expands to its SQL", async () => {
    mockApi({ "/admin/kb": { total: 1, counts: { "nlq-schema": 0, "nlq-examples": 1, common: 0 }, records: [{
      id: "ex1", namespace: "nlq-examples", type: "example", title: "Revenue by month", text: "",
      metadata: { query: "SELECT date_trunc('month', order_date) FROM ecom.orders" } }] } });
    wrap(<KnowledgeBase />);
    await waitFor(() => expect(screen.getByText("Revenue by month")).toBeInTheDocument());
    expect(screen.queryByText(/SELECT date_trunc/)).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /Revenue by month/ }));
    expect(screen.getAllByText(/SELECT date_trunc/).length).toBeGreaterThan(0);
  });
});
