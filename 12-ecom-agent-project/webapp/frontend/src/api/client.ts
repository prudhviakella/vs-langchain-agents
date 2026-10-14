// JSON calls to the backend. The session is an HttpOnly cookie set by /auth/login,
// so there is no token handling here — the browser sends the cookie itself.
import { API_PREFIX } from "../lib/config";

export class ApiError extends Error {
  status: number;
  detail: unknown;
  constructor(status: number, detail: unknown) {
    super(
      typeof detail === "string"
        ? detail
        : (detail as { message?: string })?.message ?? `Request failed (${status})`
    );
    this.status = status;
    this.detail = detail;
  }
}

type Opts = { method?: string; body?: unknown; query?: Record<string, string | undefined> };

export async function api<T>(path: string, opts: Opts = {}): Promise<T> {
  const qs = opts.query
    ? "?" + Object.entries(opts.query).filter(([, v]) => v).map(([k, v]) => `${k}=${encodeURIComponent(v!)}`).join("&")
    : "";
  const res = await fetch(API_PREFIX + path + qs, {
    method: opts.method ?? "GET",
    credentials: "same-origin",
    headers: opts.body !== undefined ? { "Content-Type": "application/json" } : undefined,
    body: opts.body !== undefined ? JSON.stringify(opts.body) : undefined,
  });
  if (res.status === 204) return undefined as T;
  const data = await res.json().catch(() => null);
  if (!res.ok) throw new ApiError(res.status, data?.detail ?? data);
  return data as T;
}
