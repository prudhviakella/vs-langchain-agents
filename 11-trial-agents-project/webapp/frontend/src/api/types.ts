// Mirrors webapp/backend/app/models.py — the contract with the backend.

export type User = { username: string; first_name: string; last_name: string; is_admin: boolean };

export type Conversation = {
  conversation_id: string; title: string; created_at: string; updated_at: string;
  message_count: number;
};

export type SearchRun = {
  query: string; doc_id: string | null; content_type: string | null; top_k: number | null;
  candidates: number; results: number; reranked: boolean | null; succeeded: boolean;
};
export type AgentStep = {
  agent: string; question: string; rationale: string; result_shape: string;
  succeeded: boolean; query: string | null; stats: Record<string, number> | null;
  searches?: SearchRun[] | null;          // absent on answers saved before searches were recorded
};
export type ToolStep = {
  tool: string; args: Record<string, unknown>; succeeded: boolean; result: string; items: number;
};
export type Citation = {
  n: number; doc_id: string; page: number | null; headings: string[]; snippet: string;
  rerank_score: number | null; origin: string;
};
export type GraphNode = { element_id: string; labels: string[]; properties: Record<string, unknown> };
export type GraphRel = { element_id: string; type: string; start: string; end: string };
export type Artifact =
  | { kind: "table"; agent: string; columns: string[]; rows: unknown[][]; total_rows: number }
  | { kind: "graph"; agent: string; nodes: GraphNode[]; relationships: GraphRel[]; total_nodes: number };

export type AnswerDetails = {
  agents: AgentStep[]; tools: ToolStep[]; citations: Citation[]; artifact: Artifact | null;
  memory: { kind: string; text: string; created_at: string; score: number | null }[];
  decision: { answerable?: boolean; entities?: string[]; note?: string;
              resolved_question?: string; from_conversation?: boolean };
  usage: { input_tokens?: number; cached_input_tokens?: number; output_tokens?: number;
           total_tokens?: number; llm_calls?: number; model_id?: string };
  cost_usd: number; latency_ms: number; history_turns: number; trace_id: string; trace_url: string;
};

export type Feedback = {
  conversation_id: string; message_id: string; username: string; rating: "up" | "down";
  comment: string; question: string; answer_snippet: string; created_at: string;
};

export type Message = {
  message_id: string; conversation_id: string; role: "user" | "assistant"; text: string;
  created_at: string; status: "complete" | "error"; details: AnswerDetails | null;
  feedback: Feedback | null;
};

export type Interaction = {
  conversation_id: string; message_id: string; username: string; question: string;
  status: string; created_at: string; agents: string[]; tools: string[];
  input_tokens: number; cached_input_tokens: number; output_tokens: number; total_tokens: number;
  llm_calls: number; cost_usd: number; latency_ms: number; trace_id: string; trace_url: string;
};

export type UsageSummary = {
  days: number; scope: "mine" | "everyone"; interactions: number; errors: number;
  input_tokens: number; cached_input_tokens: number; output_tokens: number; total_tokens: number;
  cost_usd: number; latency_p50_ms: number; latency_p95_ms: number;
  agents: Record<string, number>; tools: Record<string, number>;
  feedback_up: number; feedback_down: number;
};
