// A graph answer drawn with Neo4j NVL, as in the reference GraphView.
// trial_graph relationships are {start, end}; NVL wants {from, to}.
import { InteractiveNvlWrapper } from "@neo4j-nvl/react";
import type { Artifact, GraphNode } from "../api/types";

const caption = (n: GraphNode) => {
  const p = n.properties;
  return String(p.acronym ?? p.name ?? p.briefTitle ?? p.facility ?? p.nctId ?? n.labels[0] ?? "").slice(0, 40);
};
const COLOURS: Record<string, string> = { Trial: "#1E3A5F", Sponsor: "#16A34A", Site: "#B45309", Disease: "#8B5CF6" };

export default function GraphView({ artifact }: { artifact: Extract<Artifact, { kind: "graph" }> }) {
  const nodes = artifact.nodes.map((n) => ({ id: n.element_id, caption: caption(n),
    color: COLOURS[n.labels[0]] ?? "#6B7280", size: 24 }));
  const rels = artifact.relationships.map((r) => ({ id: r.element_id, from: r.start, to: r.end, caption: r.type }));
  return (
    <div>
      <div className="h-[380px] rounded-lg border border-line bg-white">
        <InteractiveNvlWrapper nodes={nodes} rels={rels} nvlOptions={{ initialZoom: 1 }} />
      </div>
      {artifact.total_nodes > nodes.length &&
        <p className="mt-1 text-xs text-faint">Showing {nodes.length} of {artifact.total_nodes} nodes.</p>}
    </div>
  );
}
