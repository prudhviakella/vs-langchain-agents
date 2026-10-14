export function fmtDate(ns?: number | null): string {
  if (!ns) return "";
  return new Date(Math.floor(ns / 1e6)).toLocaleString(undefined, {
    month: "long", day: "numeric", year: "numeric", hour: "2-digit", minute: "2-digit",
  });
}
// Fixed to "en-US" (rather than the browser's locale) plus an explicit
// "date • 24h-time" join, so this always reads e.g. "June 1, 2026 • 14:32"
// instead of drifting per-viewer locale (en-GB would otherwise render
// "1 June 2026 at 14:32"). Separate from fmtDate so existing callers of that
// one keep their locale-dependent formatting.
export function fmtDateTime(ns?: number | null): string {
  if (!ns) return "";
  const d = new Date(Math.floor(ns / 1e6));
  const date = d.toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric" });
  const time = d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
  return `${date} • ${time}`;
}
export function fmtDateShort(ns?: number | null): string {
  if (!ns) return "";
  return new Date(Math.floor(ns / 1e6)).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}
export function initials(name?: string | null): string {
  if (!name) return "?";
  const p = name.trim().split(/\s+/);
  return (((p[0]?.[0] ?? "") + (p[1]?.[0] ?? "")).toUpperCase()) || name[0]!.toUpperCase();
}
export function num(n?: number | null): string { return n == null ? "" : n.toLocaleString(); }
export function bytes(n?: number | null): string {
  if (!n) return "";
  if (n < 1024) return `${n} B`;
  if (n < 1048576) return `${Math.round(n / 1024)} KB`;
  return `${(n / 1048576).toFixed(1)} MB`;
}
