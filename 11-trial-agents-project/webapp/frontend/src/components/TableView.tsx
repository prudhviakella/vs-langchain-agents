import type { Artifact } from "../api/types";

export function TableView({ artifact }: { artifact: Extract<Artifact, { kind: "table" }> }) {
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-left text-sm">
        <thead><tr>{artifact.columns.map((c) => (
          <th key={c} className="border-b border-line px-2 py-1.5 font-medium text-ink">{c}</th>))}</tr></thead>
        <tbody>{artifact.rows.map((row, i) => (
          <tr key={i} className="odd:bg-canvas/60">{row.map((v, j) => (
            <td key={j} className="border-b border-line px-2 py-1 align-top">{v === null ? "—" : String(v)}</td>))}</tr>))}
        </tbody>
      </table>
      {artifact.total_rows > artifact.rows.length &&
        <p className="mt-1 text-xs text-faint">Showing {artifact.rows.length} of {artifact.total_rows} rows.</p>}
    </div>
  );
}
