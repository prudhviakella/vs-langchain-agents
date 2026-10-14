import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";
import type { ChartArtifact } from "../../api/types";

// Pass the backend's Plotly figure spec straight through. The mode bar is on, so
// every chart can be downloaded as PNG, zoomed, panned, and autoscaled natively.
export default function PlotlyChart({
  chart,
  className,
}: {
  chart: ChartArtifact;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const layout = {
      autosize: true,
      margin: { l: 56, r: 18, t: 40, b: 44 },
      font: {
        family:
          '"proxima-nova", "Proxima Nova", "Plus Jakarta Sans", Inter, sans-serif',
        size: 12,
        color: "#3A3F45",
      },
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      xaxis: { automargin: true, gridcolor: "#EEF0F2" },
      yaxis: {
        automargin: true,
        gridcolor: "#EEF0F2",
        zerolinecolor: "#E6E8EB",
      },
      ...(chart.spec?.layout || {}),
    };
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      Plotly.newPlot(el, (chart.spec?.data as any) || [], layout as any, {
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ["select2d", "lasso2d", "toggleSpikelines"],
        toImageButtonOptions: {
          format: "png",
          filename: "act-chart",
          scale: 2,
        },
      });
    } catch {
      el.innerHTML =
        '<div style="display:flex;height:160px;align-items:center;justify-content:center;color:#6b7280;font-size:14px">Couldn’t render this chart.</div>';
    }
    return () => {
      try {
        Plotly.purge(el);
      } catch {
        /* ignore */
      }
    };
  }, [chart]);
  return (
    <div ref={ref} className={className ?? "h-72 w-full overflow-hidden"} />
  );
}
