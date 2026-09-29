export const fmtInt = (n: number | undefined) => (n ?? 0).toLocaleString("en-US");
export const fmtUsd = (n: number | undefined) => `$${(n ?? 0).toFixed(4)}`;
export const fmtMs = (ms: number | undefined) =>
  !ms ? "—" : ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`;
export const fmtWhen = (iso: string) =>
  new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
export const AGENT_LABEL: Record<string, string> = {
  trial_graph: "Registry graph", trial_search: "Protocol search",
};
export const TOOL_LABEL: Record<string, string> = {
  remember_fact: "Remembered a fact", recall_facts: "Recalled facts",
  record_episode: "Recorded this interaction", recall_episodes: "Recalled past interactions",
};
