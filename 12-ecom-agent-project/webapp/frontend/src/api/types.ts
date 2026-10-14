// API types — field for field the backend's models (webapp/backend/app/models.py).

export type Vote = "up" | "down";
export type Flag = "Crit" | "High" | "Med" | "Low";
export type MessageRole = "user" | "assistant";
export type MsgKind = "answer" | "reasoning" | "clarification" | "note";

export interface Analyst {
  user_id: string;
  name: string;
  first_name?: string | null;
  last_name?: string | null;
  short_name: string;
  role: "Analyst" | "Admin";
}

export interface ConversationSummary {
  conversation_id: string;
  title: string;
  status: "Active" | "Archived";
  owner_id: string;
  last_activity_ns: number;
}

export interface ConversationPage {
  conversations: ConversationSummary[];
}

export interface TableArtifact {
  columns: string[];
  rows: (string | number)[][];
  row_count?: number;
  truncated?: boolean;
}

export interface ChartArtifact {
  kind?: "chart";
  chart_type?: string;
  insight?: string;
  source?: string;
  spec: { data: Record<string, unknown>[]; layout?: Record<string, unknown> };
}

export interface Artifacts {
  table?: TableArtifact;
  charts?: ChartArtifact[];
}

/** One query the answer rests on — the citation. */
export interface SourceRecord {
  call_id?: string;
  agent: string;
  store: string;
  query_language: string;
  query: string;
  question: string;
  row_count: number | null;
  sources: string[];
  filters: string[];
}

export interface Entity {
  eid: string;
  type: string;
  name: string;
}

export interface MessageFeedback {
  message_id: string;
  likes: number;
  dislikes: number;
  my_vote?: Vote | null;
  my_comment?: string | null;
}

export interface Message {
  message_id: string;
  conversation_id: string;
  interaction_id: string;
  role: MessageRole;
  msg_kind: MsgKind;
  text: string;
  reasoning?: string | null;
  artifacts?: Artifacts | null;
  entities: Entity[];
  options?: string[] | null;
  status?: string | null;
  sources?: SourceRecord[] | null;
  created_at_ns: number;
  feedback?: MessageFeedback | null;
  truncation_note?: string | null;
  latency_ms?: number | null;
}

export interface Page<T> {
  items: T[];
}

export interface LatestInteraction {
  interaction_id: string | null;
  message_id: string | null;
  status: "streaming" | "complete" | "failed" | "stopped" | null;
  is_active: boolean;
}
