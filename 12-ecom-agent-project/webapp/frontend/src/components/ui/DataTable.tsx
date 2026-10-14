import { useEffect, useMemo, useState, type ReactNode } from "react";
import { cx } from "./primitives";

// ---------------------------------------------------------------------------
// DataTable — a dependency-free, premium data grid: sortable columns, a global
// search box, and pagination. Columns declare how to render, sort, and filter a
// row, so it works for simple string/number grids *and* rows with inline
// controls (selects, inputs, buttons). Styling follows the Gilead tokens.
// ---------------------------------------------------------------------------

export type Align = "left" | "right" | "center";
type SortDir = "asc" | "desc";

export interface Column<T> {
  key: string;
  header: ReactNode;
  align?: Align;                                       // cell + header alignment
  sortable?: boolean;                                  // default true
  sortValue?: (row: T) => string | number | null | undefined;
  filterText?: (row: T) => string;                     // contributes to global search
  render: (row: T) => ReactNode;
  headerClassName?: string;
  cellClassName?: string;
  width?: string | number;
}

export interface DataTableProps<T> {
  columns: Column<T>[];
  rows: T[];
  rowKey: (row: T, index: number) => string | number;
  pageSize?: number;                                   // default 10
  pageSizeOptions?: number[];                          // default [10, 25, 50]
  searchable?: boolean;                                // default: auto (rows > 8)
  searchPlaceholder?: string;
  initialSort?: { key: string; dir: SortDir };
  toolbar?: ReactNode;                                 // extra left-side controls (filters)
  emptyText?: string;
  minWidth?: number;
  className?: string;
}

const alignCls: Record<Align, string> = { left: "text-left", right: "text-right", center: "text-center" };

function SortCaret({ state }: { state: SortDir | "none" }) {
  return (
    <span className="ml-1 inline-flex flex-col leading-none align-middle">
      <svg width="7" height="4" viewBox="0 0 7 4" className={state === "asc" ? "text-brand" : "text-faint/50"}><path d="M3.5 0 7 4H0z" fill="currentColor" /></svg>
      <svg width="7" height="4" viewBox="0 0 7 4" className={state === "desc" ? "text-brand" : "text-faint/50"}><path d="M3.5 4 0 0h7z" fill="currentColor" /></svg>
    </span>
  );
}

function PageButton({ disabled, onClick, label, children }: { disabled?: boolean; onClick: () => void; label: string; children: ReactNode }) {
  return (
    <button type="button" onClick={onClick} disabled={disabled} aria-label={label}
      className="inline-flex h-7 min-w-7 items-center justify-center rounded-md border border-line bg-white px-1.5 text-muted shadow-xs transition-all duration-150 ease-premium hover:border-[#D8DBDF] hover:text-ink disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:border-line disabled:hover:text-muted focus:outline-hidden focus-visible:ring-2 focus-visible:ring-brand/40">
      {children}
    </button>
  );
}

export function DataTable<T>({
  columns, rows, rowKey,
  pageSize = 10, pageSizeOptions = [10, 25, 50],
  searchable, searchPlaceholder = "Search…",
  initialSort, toolbar, emptyText = "No results", minWidth, className,
}: DataTableProps<T>) {
  const [query, setQuery] = useState("");
  const [sortKey, setSortKey] = useState<string | null>(initialSort?.key ?? null);
  const [sortDir, setSortDir] = useState<SortDir>(initialSort?.dir ?? "asc");
  const [page, setPage] = useState(0);
  const [size, setSize] = useState(pageSize);

  const canSearch = columns.some((c) => !!c.filterText);
  const showSearch = canSearch && (searchable ?? rows.length > 5);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return rows;
    return rows.filter((r) =>
      columns.some((c) => (c.filterText ? c.filterText(r).toLowerCase().includes(q) : false)),
    );
  }, [rows, query, columns]);

  const sorted = useMemo(() => {
    if (!sortKey) return filtered;
    const col = columns.find((c) => c.key === sortKey);
    if (!col?.sortValue) return filtered;
    const dir = sortDir === "asc" ? 1 : -1;
    const val = col.sortValue;
    return [...filtered].sort((a, b) => {
      const av = val(a), bv = val(b);
      const an = av == null || av === "", bn = bv == null || bv === "";
      if (an && bn) return 0;
      if (an) return 1;            // empties last, regardless of direction
      if (bn) return -1;
      if (typeof av === "number" && typeof bv === "number") return (av - bv) * dir;
      return String(av).localeCompare(String(bv), undefined, { numeric: true, sensitivity: "base" }) * dir;
    });
  }, [filtered, sortKey, sortDir, columns]);

  useEffect(() => { setPage(0); }, [query, size, rows.length, sortKey, sortDir]);

  const total = sorted.length;
  const pageCount = Math.max(1, Math.ceil(total / size));
  const current = Math.min(page, pageCount - 1);
  const showPagination = total > 0;
  const start = current * size;
  const paged = sorted.slice(start, start + size);

  const toggleSort = (col: Column<T>) => {
    if (col.sortable === false || !col.sortValue) return;
    if (sortKey !== col.key) { setSortKey(col.key); setSortDir("asc"); return; }
    if (sortDir === "asc") { setSortDir("desc"); return; }
    setSortKey(null);           // asc -> desc -> off
  };

  return (
    <div className={cx("overflow-hidden rounded-lg border border-line bg-card shadow-sm", className)}>
      {(showSearch || toolbar) && (
        <div className="flex flex-wrap items-center gap-2 border-b border-line px-3 py-2.5">
          {toolbar && <div className="flex flex-wrap items-center gap-2">{toolbar}</div>}
          {showSearch && (
            <label className="relative ml-auto flex items-center">
              <svg className="pointer-events-none absolute left-2.5 h-3.5 w-3.5 text-faint" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}><circle cx="11" cy="11" r="7" /><path d="m21 21-4-4" /></svg>
              <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder={searchPlaceholder}
                className="w-52 max-w-full rounded-lg border border-line bg-white py-1.5 pl-8 pr-3 text-sm text-ink shadow-xs transition placeholder:text-faint hover:border-[#D8DBDF] focus:border-brand/50 focus:outline-hidden focus:ring-2 focus:ring-brand/20" />
            </label>
          )}
        </div>
      )}

      <div className="overflow-x-auto">
        <table className="w-full text-sm" style={minWidth ? { minWidth } : undefined}>
          <thead>
            <tr className="border-b border-line bg-canvas/60">
              {columns.map((c) => {
                const active = sortKey === c.key;
                const sortable = c.sortable !== false && !!c.sortValue;
                return (
                  <th key={c.key} style={c.width ? { width: c.width } : undefined}
                    className={cx("px-3 py-2.5 text-[11px] font-semibold uppercase tracking-wider text-muted",
                      alignCls[c.align ?? "left"], sortable && "cursor-pointer select-none transition-colors hover:text-ink", c.headerClassName)}
                    onClick={sortable ? () => toggleSort(c) : undefined}
                    aria-sort={active ? (sortDir === "asc" ? "ascending" : "descending") : undefined}>
                    <span className={cx("inline-flex items-center", c.align === "right" && "flex-row-reverse")}>
                      {c.header}
                      {sortable && <SortCaret state={active ? sortDir : "none"} />}
                    </span>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {paged.length === 0 ? (
              <tr><td colSpan={columns.length} className="px-3 py-10 text-center text-sm text-muted">{emptyText}</td></tr>
            ) : (
              paged.map((row, i) => (
                <tr key={rowKey(row, start + i)} className="row-interactive border-b border-line/60 last:border-0">
                  {columns.map((c) => (
                    <td key={c.key} className={cx("px-3 py-2.5 align-middle text-body", alignCls[c.align ?? "left"], c.cellClassName)}>
                      {c.render(row)}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {showPagination && (
        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-line px-3 py-2.5">
          <p className="text-xs text-muted">
            Showing <span className="font-medium text-body">{total === 0 ? 0 : start + 1}</span>–<span className="font-medium text-body">{Math.min(start + size, total)}</span> of <span className="font-medium text-body">{total}</span>
          </p>
          <div className="flex items-center gap-3">
            <label className="hidden items-center gap-1.5 text-xs text-muted sm:flex">
              Rows
              <select value={size} onChange={(e) => setSize(Number(e.target.value))}
                className="cursor-pointer rounded-md border border-line bg-white px-1.5 py-1 text-xs text-ink shadow-xs focus:border-brand/50 focus:outline-hidden focus:ring-2 focus:ring-brand/20">
                {pageSizeOptions.map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </label>
            <div className="flex items-center gap-1">
              <PageButton label="First page" disabled={current === 0} onClick={() => setPage(0)}>
                <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}><path d="m17 6-6 6 6 6M11 6l-6 6 6 6" /></svg>
              </PageButton>
              <PageButton label="Previous page" disabled={current === 0} onClick={() => setPage(current - 1)}>
                <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}><path d="m14 6-6 6 6 6" /></svg>
              </PageButton>
              <span className="px-1 text-xs text-muted">Page <span className="font-medium text-body">{current + 1}</span> / {pageCount}</span>
              <PageButton label="Next page" disabled={current >= pageCount - 1} onClick={() => setPage(current + 1)}>
                <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}><path d="m10 6 6 6-6 6" /></svg>
              </PageButton>
              <PageButton label="Last page" disabled={current >= pageCount - 1} onClick={() => setPage(pageCount - 1)}>
                <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}><path d="m7 6 6 6-6 6M13 6l6 6-6 6" /></svg>
              </PageButton>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
