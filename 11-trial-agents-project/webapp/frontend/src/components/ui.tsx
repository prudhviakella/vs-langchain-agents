// Small primitives in the reference app's style (card, hairline borders, primary navy).
import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode } from "react";

export const cx = (...parts: (string | false | null | undefined)[]) => parts.filter(Boolean).join(" ");

export function Button({ variant = "primary", className, ...props }:
  ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "ghost" | "danger" }) {
  return (
    <button {...props} className={cx(
      "inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium transition disabled:opacity-50",
      variant === "primary" && "bg-primary text-white hover:bg-primary-dark",
      variant === "ghost" && "text-body hover:bg-canvas",
      variant === "danger" && "text-bad hover:bg-red-50",
      className)} />
  );
}

export function Input(props: InputHTMLAttributes<HTMLInputElement>) {
  return <input {...props} className={cx(
    "w-full rounded-lg border border-line bg-white px-3 py-2 text-sm text-ink outline-none focus:border-primary",
    props.className)} />;
}

export function Card({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cx("rounded-xl2 border border-line bg-card p-4 shadow-card", className)}>{children}</div>;
}

export function Badge({ children, tone = "neutral" }: { children: ReactNode; tone?: "neutral" | "ok" | "bad" | "info" }) {
  return <span className={cx("inline-block rounded-full px-2 py-0.5 text-xs font-medium",
    tone === "neutral" && "bg-canvas text-muted", tone === "ok" && "bg-green-50 text-ok",
    tone === "bad" && "bg-red-50 text-bad", tone === "info" && "bg-skyTint text-primary")}>{children}</span>;
}

export function Spinner() {
  return <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-line border-t-primary" aria-label="working" />;
}

export function Tabs<T extends string>({ tabs, active, onChange }:
  { tabs: { id: T; label: string; count?: number }[]; active: T; onChange: (id: T) => void }) {
  return (
    <div className="flex gap-1 border-b border-line" role="tablist">
      {tabs.map((t) => (
        <button key={t.id} role="tab" aria-selected={active === t.id} onClick={() => onChange(t.id)}
          className={cx("px-3 py-1.5 text-sm", active === t.id
            ? "border-b-2 border-primary font-medium text-ink" : "text-muted hover:text-ink")}>
          {t.label}{t.count !== undefined && <span className="ml-1 text-faint">{t.count}</span>}
        </button>
      ))}
    </div>
  );
}
