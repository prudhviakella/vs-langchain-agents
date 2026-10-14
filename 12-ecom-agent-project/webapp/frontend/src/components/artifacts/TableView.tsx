import { useState } from "react";
import type { TableArtifact, Flag } from "../../api/types";
import { FlagBadge } from "../ui/primitives";
import { DataTable, type Column } from "../ui/DataTable";
import { FullscreenModal, IconButton, ExpandIcon, DownloadIcon } from "./FullscreenModal";

// Assistant result tables are DYNAMIC — columns depend on the user's question, so
// there is no fixed schema and no guaranteed "flag" column. We render whatever
// columns arrive. The only special-casing is risk badges, and it is *column-aware*:
// a column is treated as a flag column only when its header looks like one AND all
// its values are the known tiers — so an ordinary "Priority: High" cell stays plain.
const FLAG_VALUES = new Set<Flag>(["Crit", "High", "Med", "Low"]);
const FLAG_HEADER = /^(flag|flags|severity|risk|risk[ _]?level|tier|priority)$/i;

// ---------------------------------------------------------------------------
//  Column headers are DATABASE IDENTIFIERS. Analysts are not.
// ---------------------------------------------------------------------------
// A result table arrives with the warehouse's own column names — SOURCE_TABLE,
// NEO4J_ADDRESS, ST_CD, CLAIM_COUNT. An investigator reading a lead should not
// have to decode `ST_CD`, and `NEO4J_ADDRESS` is worse than opaque: it names the
// platform's graph database in a document that may end up in a case file.
//
// This is presentation only. The underlying column name is unchanged, still
// exported, still in the cited query — so nothing about provenance is lost. What
// changes is what a human reads.
//
// Deliberately a small, literal map plus generic tidying, NOT clever inference.
// A wrong guess at a column's meaning is worse than a raw name: a mislabelled
// column silently changes what the reader thinks the number IS. Anything not
// recognised is only de-underscored and title-cased, which cannot mislead.
//
// The better long-term source is the KB itself — every column carries
// `analyst_synonyms`, and the first synonym is a human label written by the BA.
// Threading those through to the artifact would make this map unnecessary.
const HEADER_LABELS: Record<string, string> = {
  st_cd: "State",
  st: "State",
  npi: "NPI",
  acnt_nm: "Pharmacy",
  acnt_eid: "Pharmacy ID",
  pres_nm: "Prescriber",
  pres_npi: "Prescriber NPI",
  src_pres_npi: "Prescriber NPI",
  pharmacy_npi: "Pharmacy NPI",
  claim_count: "Claims",
  presn_num: "Prescription #",
  brd_nm: "Brand",
  ofr_val_amt: "Offer Value",
  page_rank: "Network Proximity",
  anomaly_score: "Anomaly Score",
  fraud_score: "Fraud Score",
  node_id: "Entity ID",
  member_id: "Member ID",
  // Internal plumbing an analyst should never be asked to interpret.
  source_table: "Source",
  neo4j_address: "Address",
  neo4j_name: "Name",
  neo4j_id: "Entity ID",
  pt_cycle_id: "Data Cycle",
  cycle_id: "Data Cycle",
  run_id: "Model Run",
};

export function humanizeHeader(raw: unknown): string {
  const s = String(raw ?? "").trim();
  if (!s) return "";
  const mapped = HEADER_LABELS[s.toLowerCase()];
  if (mapped) return mapped;
  // Unrecognised: strip an internal prefix, de-underscore, title-case. Never
  // guess at meaning — this can only make a name readable, not wrong.
  return s
    .replace(/^(neo4j|src|rpt|agg|pt)_/i, "")
    .replace(/[_\s]+/g, " ")
    .trim()
    .replace(/\b\w/g, (c) => c.toUpperCase());
}


// ---------------------------------------------------------------------------
//  CSV export
// ---------------------------------------------------------------------------
// Exports the ORIGINAL column names, not the humanised headers. What is on
// screen is for reading; what leaves the platform goes into a spreadsheet, an
// email, or a case file, where `st_cd` is the name that will still match the
// warehouse and the cited query months later. Renaming it to "State" on the way
// out would break that link precisely when someone needs it.
//
// Exports EVERY row, not the current page. A file called
// `<something>.csv` that silently contains ten of forty-seven rows is the
// reassuring-direction failure: it looks complete and is not.
export function toCsv(columns: unknown[], rows: (string | number)[][]): string {
  const cell = (v: unknown) => {
    if (v == null) return "";
    const s = String(v);
    // Quote when the value contains a delimiter, a quote, or a newline; double
    // any embedded quotes. Excel and Sheets both read this correctly.
    return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [columns.map(cell).join(","), ...rows.map((r) => r.map(cell).join(","))]
    .join("\r\n");                                  // CRLF — RFC 4180, and what Excel expects
}

// Shared by every CSV-export button in the artifact tray (results tables,
// the entity panel, ...) \u2014 one BOM/filename/anchor-click implementation
// rather than each caller reinventing it slightly differently.
export function downloadCsv(
  columns: unknown[],
  rows: (string | number)[][],
  filenamePrefix = "act-table",
) {
  // A BOM, so Excel opens UTF-8 correctly. Without it, pharmacy names with
  // accents arrive mojibaked in the one place an analyst is most likely to
  // paste them into a document that leaves the building.
  const blob = new Blob(["\uFEFF" + toCsv(columns, rows)],
                        { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${filenamePrefix}-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.csv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

type Row = (string | number)[];

export function TableView({ table }: { table: TableArtifact }) {
  const columns = Array.isArray(table?.columns) ? table.columns : [];
  const rows = Array.isArray(table?.rows) ? table.rows : [];
  const [full, setFull] = useState(false);
  if (columns.length === 0) return null;

  const cols: Column<Row>[] = columns.map((label, i) => {
    const values = rows.map((r) => r[i]);
    const numeric = values.length > 0 && values.every((v) => v == null || typeof v === "number");
    const isFlagCol =
      FLAG_HEADER.test(String(label).trim()) &&
      values.length > 0 &&
      values.every((v) => v == null || FLAG_VALUES.has(String(v) as Flag));

    return {
      key: String(i),
      header: humanizeHeader(label),
      align: i === 0 ? "left" : numeric ? "right" : "left",
      sortValue: (r) => (typeof r[i] === "number" ? (r[i] as number) : r[i] == null ? "" : String(r[i])),
      filterText: (r) => (r[i] == null ? "" : String(r[i])),
      render: (r) => {
        const cell = r[i];
        if (cell == null || cell === "") return <span className="text-faint">—</span>;
        if (isFlagCol) return <FlagBadge flag={String(cell) as Flag} />;
        if (typeof cell === "number") return <span className="tabular-nums">{cell.toLocaleString()}</span>;
        return <span className={i === 0 ? "font-medium text-ink" : undefined}>{String(cell)}</span>;
      },
    };
  });

  // Expanded view shows more rows per page as well as more width. A wider
  // table that still pages ten at a time has only solved half the problem.
  // `renderTable`, not `table` — the component's own prop is named `table`.
  const renderTable = (expanded: boolean) => (
    <DataTable columns={cols} rows={rows} rowKey={(_, i) => i}
               pageSize={expanded ? 50 : 10} searchable
               searchPlaceholder="Filter rows…" />
  );

  return (
    <>
      <div className="flex items-center justify-end gap-1.5 pb-1.5">
        <span className="mr-auto text-[11px] text-faint">
          {rows.length.toLocaleString()} row{rows.length === 1 ? "" : "s"}
        </span>
        <IconButton label="Download CSV" onClick={() => downloadCsv(columns, rows)}>
          <DownloadIcon />
        </IconButton>
        <IconButton label="Expand table" onClick={() => setFull(true)}>
          <ExpandIcon />
        </IconButton>
      </div>

      {renderTable(false)}

      {full && (
        <FullscreenModal title={`Results — ${rows.length.toLocaleString()} rows`}
                         onClose={() => setFull(false)}>
          <div className="flex justify-end pb-2">
            <IconButton label="Download CSV" onClick={() => downloadCsv(columns, rows)}>
              <DownloadIcon />
            </IconButton>
          </div>
          {renderTable(true)}
        </FullscreenModal>
      )}
    </>
  );
}
