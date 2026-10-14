// Table + charts of one answer (ACT's ArtifactRenderer, trimmed: no graphs, no exports).
//
//   artifacts.table   ─► TableView   (sort, full screen, CSV download)
//   artifacts.charts  ─► ChartView   one Plotly figure each; several → a tab strip
//
// Each chart sits in its own ErrorBoundary: one malformed figure must not take down
// its siblings or the answer text.
import type { Artifacts, ChartArtifact } from "../../api/types";
import { ErrorBoundary } from "../ErrorBoundary";
import { ArtifactTabs } from "./ArtifactTabs";
import { ChartView, chartTitle } from "./ChartView";
import { LazyArtifact } from "./LazyArtifact";
import { TableView } from "./TableView";

export function ArtifactRenderer({ artifacts }: { artifacts?: Artifacts | null }) {
  if (!artifacts) return null;
  const charts: ChartArtifact[] = Array.isArray(artifacts.charts) ? artifacts.charts : [];
  const table = artifacts.table;
  if (!table && charts.length === 0) return null;
  const one = (c: ChartArtifact) => (
    <ErrorBoundary>
      <LazyArtifact label="Chart">
        <ChartView chart={c} />
      </LazyArtifact>
    </ErrorBoundary>
  );
  return (
    <div className="mt-4 space-y-4">
      {charts.length === 1 && one(charts[0])}
      {charts.length > 1 && (
        <ArtifactTabs items={charts} label="Chart" itemKey={(_, i) => i}
                      titleFor={(c, i) => chartTitle(c, `Chart ${i + 1}`)} renderItem={(c) => one(c)} />
      )}
      {table && (
        <ErrorBoundary>
          <TableView table={table} />
        </ErrorBoundary>
      )}
    </div>
  );
}
