import type { ButtonHTMLAttributes, ReactNode, CSSProperties } from "react";
import { useEffect, useRef, useState } from "react";
import type { Flag } from "../../api/types";
import { ApiError } from "../../api/client";

export function cx(...c: (string | false | null | undefined)[]) {
  return c.filter(Boolean).join(" ");
}

// ---- Button -----------------------------------------------------------------
type Variant = "primary" | "subtle" | "ghost" | "danger";
const variants: Record<Variant, string> = {
  primary:
    "bg-brand text-white shadow-sm hover:bg-brand-600 hover:shadow-md disabled:opacity-50",
  subtle:
    "bg-white text-ink border border-line shadow-xs hover:bg-canvas hover:border-[#D8DBDF] disabled:opacity-50",
  ghost: "text-muted hover:text-ink hover:bg-canvas disabled:opacity-50",
  danger: "text-brand hover:bg-brand-50 disabled:opacity-50",
};
export function Button({
  variant = "subtle",
  className,
  children,
  ...p
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
  return (
    <button
      {...p}
      className={cx(
        "inline-flex items-center justify-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium transition-all duration-150 ease-premium active:scale-[0.98] focus:outline-hidden focus-visible:ring-2 focus-visible:ring-brand/40 focus-visible:ring-offset-1 focus-visible:ring-offset-white disabled:pointer-events-none",
        variants[variant],
        className
      )}
    >
      {children}
    </button>
  );
}

// ---- Badges -----------------------------------------------------------------
export function Label({ children }: { children: ReactNode }) {
  return (
    <span className="inline-block rounded-sm bg-canvas px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wider text-muted ring-1 ring-inset ring-line">
      {children}
    </span>
  );
}
const flagStyle: Record<Flag, string> = {
  Crit: "bg-flag-critbg text-flag-crit ring-flag-crit/20",
  High: "bg-flag-highbg text-flag-high ring-flag-high/20",
  Med: "bg-flag-medbg text-flag-med ring-flag-med/20",
  Low: "bg-flag-lowbg text-flag-low ring-flag-low/20",
};
export function FlagBadge({ flag }: { flag?: Flag | null }) {
  if (!flag) return null;
  return (
    <span
      className={cx(
        "inline-flex items-center rounded-md px-2 py-0.5 text-xs font-semibold ring-1 ring-inset",
        flagStyle[flag]
      )}
    >
      {flag}
    </span>
  );
}

// ---- Avatar -----------------------------------------------------------------
export function Avatar({
  name,
  size = 26,
}: {
  name?: string | null;
  size?: number;
}) {
  const init = (
    (name?.trim()?.[0] ?? "?") + (name?.trim()?.split(/\s+/)?.[1]?.[0] ?? "")
  ).toUpperCase();
  return (
    <span
      className="inline-flex items-center justify-center rounded-full font-semibold text-white ring-1 ring-inset ring-white/20 shadow-xs"
      style={{
        width: size,
        height: size,
        fontSize: size * 0.4,
        background: "#1E3A5F",
      }}
    >
      {init}
    </span>
  );
}

// ---- Card -------------------------------------------------------------------
export function Card({
  className,
  children,
  hover,
  onClick,
  style,
}: {
  className?: string;
  children: ReactNode;
  hover?: boolean;
  onClick?: () => void;
  style?: CSSProperties;
}) {
  return (
    <div
      onClick={onClick}
      style={style}
      className={cx(
        "rounded-xl2 border border-line bg-card shadow-card",
        hover && "card-interactive",
        className
      )}
    >
      {children}
    </div>
  );
}

// ---- States -----------------------------------------------------------------
export function Spinner({ className }: { className?: string }) {
  return (
    <span
      className={cx(
        "inline-block animate-spin rounded-full border-2 border-line border-t-brand",
        className
      )}
      style={{ width: 16, height: 16 }}
    />
  );
}
export function Loading({ label = "Loading" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 p-6 text-sm text-muted">
      <Spinner /> {label}…
    </div>
  );
}
export function EmptyState({
  title,
  hint,
  action,
  icon,
}: {
  title: string;
  hint?: string;
  action?: ReactNode;
  icon?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-xl2 border border-dashed border-line bg-white/50 p-10 text-center">
      {icon && (
        <span className="relative mb-1 flex h-16 w-16 items-center justify-center rounded-full bg-brand-50 text-brand">
          {icon}
        </span>
      )}
      <p className="font-semibold text-ink">{title}</p>
      {hint && <p className="max-w-md text-sm text-muted">{hint}</p>}
      {action && <div className="mt-1">{action}</div>}
    </div>
  );
}
export function ErrorState({ error }: { error: unknown }) {
  const msg =
    error instanceof ApiError
      ? `${error.message}`
      : error instanceof Error
      ? error.message
      : "Something went wrong";
  return (
    <div className="flex items-start gap-2.5 rounded-xl2 border border-flag-crit/20 bg-flag-critbg/50 p-4 text-sm text-flag-crit">
      <svg
        className="mt-0.5 h-4 w-4 shrink-0"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth={2}
      >
        <circle cx="12" cy="12" r="9" />
        <path strokeLinecap="round" d="M12 8v5M12 16h.01" />
      </svg>
      <span>{msg}</span>
    </div>
  );
}

// ---- Modal ------------------------------------------------------------------
export function Modal({
  open,
  onClose,
  title,
  children,
  width = 480,
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
  width?: number;
}) {
  useEffect(() => {
    if (!open) return;
    const h = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-ink/40 p-4 pt-24 backdrop-blur-[2px] animate-fade-in"
      onClick={onClose}
    >
      <div
        className="w-full rounded-xl2 border border-line bg-card shadow-lg animate-fade-in-up"
        style={{ maxWidth: width }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-line px-5 py-3.5">
          <h3 className="text-base font-semibold text-ink">{title}</h3>
          <button
            onClick={onClose}
            className="rounded-md p-1 text-muted transition-colors hover:bg-canvas hover:text-ink focus:outline-hidden focus-visible:ring-2 focus-visible:ring-brand/40"
            aria-label="Close"
          >
            ✕
          </button>
        </div>
        <div className="p-5">{children}</div>
      </div>
    </div>
  );
}

// ---- Field ------------------------------------------------------------------
export function Field({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-xs font-semibold uppercase tracking-wide text-muted">
        {label}
      </span>
      {children}
    </label>
  );
}
export const inputCls =
  "w-full rounded-lg border border-line bg-white px-3 py-2 text-sm text-ink shadow-xs transition duration-150 ease-premium placeholder:text-faint hover:border-[#D8DBDF] focus:border-brand/50 focus:outline-hidden focus:ring-2 focus:ring-brand/20";

// ---- Menu (dropdown) --------------------------------------------------------
export function Menu({
  button,
  align = "right",
  children,
}: {
  button: ReactNode;
  align?: "left" | "right";
  children: (close: () => void) => ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const h = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node))
        setOpen(false);
    };
    window.addEventListener("mousedown", h);
    return () => window.removeEventListener("mousedown", h);
  }, [open]);
  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        className="rounded-lg px-2 py-1.5 text-muted transition-colors hover:bg-canvas hover:text-ink focus:outline-hidden focus-visible:ring-2 focus-visible:ring-brand/40"
      >
        {button}
      </button>
      {open && (
        <div
          className={cx(
            "absolute z-30 mt-1 min-w-[168px] overflow-hidden rounded-xl2 border border-line bg-white py-1 shadow-pop animate-scale-in",
            align === "right"
              ? "right-0 origin-top-right"
              : "left-0 origin-top-left"
          )}
        >
          {children(() => setOpen(false))}
        </div>
      )}
    </div>
  );
}
export function MenuItem({
  onClick,
  danger,
  children,
}: {
  onClick: () => void;
  danger?: boolean;
  children: ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={cx(
        "block w-full px-3 py-1.5 text-left text-sm transition-colors hover:bg-canvas",
        danger ? "text-brand hover:bg-brand-50" : "text-body"
      )}
    >
      {children}
    </button>
  );
}
