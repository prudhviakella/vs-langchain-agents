import { useState, type ReactNode } from "react";
import { cx } from "../ui/primitives";

/**
 * A horizontal tab strip for a GROUP of same-kind artifacts (all the charts on
 * one answer, or all the graphs). Replaces stacking every item vertically down
 * the page — on a turn with several charts and several graphs, that made a
 * single conversation message very tall. Only the selected tab's content is
 * mounted; switching tabs unmounts the previous one and mounts the next.
 *
 * That "only one mounted at a time" behavior is also why LazyArtifact's
 * viewport-based lazy-mount isn't used PER TAB here: the tab strip already
 * bounds live chart/graph instances to one, which is the same problem
 * LazyArtifact solves for a long vertical stack, solved a different way. The
 * whole GROUP can still be wrapped in LazyArtifact by the caller so nothing
 * mounts until the group scrolls into view at all.
 *
 * A single item renders with NO tab strip at all — tabs earn their place only
 * when there's something to switch between.
 */
export function ArtifactTabs<T>({
  items,
  titleFor,
  renderItem,
  itemKey,
  label,
}: {
  items: T[];
  titleFor: (item: T, index: number) => string;
  renderItem: (item: T, index: number) => ReactNode;
  itemKey: (item: T, index: number) => string | number;
  /** e.g. "Chart" / "Graph" — used only for the aria-label. */
  label: string;
}) {
  const [active, setActive] = useState(0);
  if (items.length === 0) return null;
  if (items.length === 1) return <>{renderItem(items[0], 0)}</>;

  const safeActive = Math.min(active, items.length - 1);

  return (
    <div>
      <div
        role="tablist"
        aria-label={`${label}s`}
        className="mb-3 flex items-center gap-1 overflow-x-auto border-b border-line"
      >
        {items.map((item, i) => {
          const selected = i === safeActive;
          return (
            <button
              key={itemKey(item, i)}
              role="tab"
              aria-selected={selected}
              onClick={() => setActive(i)}
              className={cx(
                "shrink-0 whitespace-nowrap border-b-2 px-3 py-2 text-sm font-medium transition-colors",
                selected
                  ? "border-brand text-brand"
                  : "border-transparent text-muted hover:text-ink"
              )}
            >
              {titleFor(item, i)}
            </button>
          );
        })}
      </div>
      <div role="tabpanel">{renderItem(items[safeActive], safeActive)}</div>
    </div>
  );
}
