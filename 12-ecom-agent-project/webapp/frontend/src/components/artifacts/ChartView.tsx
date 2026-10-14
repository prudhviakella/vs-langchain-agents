import { lazy, Suspense, useState } from "react";
import type { ChartArtifact } from "../../api/types";
import { FullscreenModal, IconButton, ExpandIcon } from "./FullscreenModal";

const PlotlyChart = lazy(() => import("./PlotlyChart"));

function hasData(c: ChartArtifact | undefined): boolean {
  return !!c?.spec && Array.isArray(c.spec.data) && c.spec.data.length > 0;
}
export function chartTitle(c: ChartArtifact, fallback = "Chart"): string {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const t = (c.spec?.layout as any)?.title;
  return (typeof t === "string" ? t : t?.text) || fallback;
}

export function ChartView({ chart }: { chart: ChartArtifact }) {
  const [full, setFull] = useState(false);
  if (!hasData(chart)) {
    return (
      <div className="rounded-lg border border-line bg-canvas/50 p-4 text-sm text-muted">
        Chart data wasn’t in a recognized format, so it can’t be plotted.
      </div>
    );
  }
  return (
    <div className="group relative">
      {/* top-left so it doesn't collide with Plotly's mode bar (top-right) */}
      <div className="absolute left-2 top-2 z-10 opacity-0 transition group-hover:opacity-100">
        <IconButton label="Expand chart" onClick={() => setFull(true)}><ExpandIcon /></IconButton>
      </div>
      <Suspense fallback={<div className="h-72 w-full animate-pulse rounded-lg bg-canvas" />}>
        <PlotlyChart chart={chart} />
      </Suspense>
      {full && (
        <FullscreenModal title={chartTitle(chart)} onClose={() => setFull(false)}>
          <Suspense fallback={<div className="h-full w-full animate-pulse rounded-lg bg-canvas" />}>
            <PlotlyChart chart={chart} className="h-[72vh] w-full" />
          </Suspense>
        </FullscreenModal>
      )}
    </div>
  );
}
