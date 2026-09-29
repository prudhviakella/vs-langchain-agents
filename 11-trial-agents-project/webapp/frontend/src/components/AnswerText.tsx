// The composer writes light markdown: **bold**, "- " bullets, and | tables |.
// Everything is rendered as React elements, never as HTML, so answer text
// cannot inject markup.
//
//   | | PIONEER 4 | STEP 1 |        header row      -> <thead>
//   |---|---|---|                    separator row   -> dropped
//   | **Population** | … | … |       body rows       -> <tbody>
import { Fragment } from "react";

function inline(text: string) {
  return text.split(/(\*\*[^*]+\*\*|\*[^*\s][^*]*\*)/g).map((part, i) =>
    part.startsWith("**") && part.endsWith("**")
      ? <strong key={i} className="font-semibold text-ink">{part.slice(2, -2)}</strong>
      : part.length > 2 && part.startsWith("*") && part.endsWith("*")
        ? <em key={i}>{part.slice(1, -1)}</em>
        : <Fragment key={i}>{part}</Fragment>);
}

const isTableRow = (line: string) => /^\s*\|.*\|\s*$/.test(line);
const isSeparator = (line: string) => /^\s*\|(\s*:?-{3,}:?\s*\|)+\s*$/.test(line);
const cells = (line: string) => line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());

function Table({ lines }: { lines: string[] }) {
  const [head, ...rest] = lines;
  const body = rest.filter((l) => !isSeparator(l));
  return (
    <div className="my-2 overflow-x-auto">
      <table className="min-w-full border-collapse text-left text-sm">
        <thead><tr>{cells(head).map((c, i) => (
          <th key={i} className="border-b border-line bg-canvas px-2 py-1.5 align-bottom font-semibold text-ink">{inline(c)}</th>))}</tr></thead>
        <tbody>{body.map((row, r) => (
          <tr key={r}>{cells(row).map((c, i) => (
            <td key={i} className="border-b border-line px-2 py-1.5 align-top">{inline(c)}</td>))}</tr>))}</tbody>
      </table>
    </div>
  );
}

export function AnswerText({ text }: { text: string }) {
  const blocks: JSX.Element[] = [];
  const lines = text.split("\n");
  let bullets: string[] = [];
  const flush = () => {
    if (bullets.length) blocks.push(<ul key={blocks.length} className="my-1 list-disc space-y-1 pl-5">
      {bullets.map((b, i) => <li key={i}>{inline(b)}</li>)}</ul>);
    bullets = [];
  };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    // A table is a header row followed at once by a separator row.
    if (isTableRow(line) && i + 1 < lines.length && isSeparator(lines[i + 1])) {
      flush();
      const table: string[] = [];
      while (i < lines.length && isTableRow(lines[i])) table.push(lines[i++]);
      i--;
      blocks.push(<Table key={blocks.length} lines={table} />);
      continue;
    }
    if (/^\s*[-*] /.test(line)) { bullets.push(line.replace(/^\s*[-*] /, "")); continue; }
    flush();
    if (line.trim()) blocks.push(<p key={blocks.length} className="my-1">{inline(line)}</p>);
  }
  flush();
  return <div className="text-sm leading-relaxed">{blocks}</div>;
}
