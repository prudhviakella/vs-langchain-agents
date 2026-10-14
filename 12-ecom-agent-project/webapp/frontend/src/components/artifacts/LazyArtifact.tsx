import { useEffect, useRef, useState, type ReactNode } from "react";

/**
 * Mounts a heavy child (a Plotly chart or an NVL graph — each holds a WebGL
 * context plus web workers) ONLY while it is near the viewport, so a long
 * transcript doesn't mount every chart and graph at once (browsers cap WebGL at
 * ~16 contexts, and mounting dozens makes a long conversation stutter or crash).
 *
 * `keepMounted` changes the teardown behavior: once the child has come into view
 * a first time it STAYS mounted even when scrolled away. Use it for NVL graphs —
 * they run a force layout on mount, so re-mounting on every scroll-back makes the
 * graph visibly collapse to a dot and re-expand. Charts leave it off: Plotly
 * re-renders instantly from its spec with no such reflow, so mounting/unmounting
 * them keeps live GPU contexts bounded at no visible cost.
 *
 * A fixed `minHeight` is reserved until first mount, so scroll height — and the
 * reverse-scroll anchoring in ConversationScreen — stays stable. The ~800px
 * `rootMargin` mounts a screen early and (when not keepMounted) lags the unmount
 * so small scroll jitter at the boundary doesn't thrash it. No dependency —
 * IntersectionObserver is native.
 */
export function LazyArtifact({
  minHeight = 320,
  label = "Visualization",
  keepMounted = false,
  children,
}: {
  minHeight?: number;
  label?: string;
  keepMounted?: boolean;
  children: ReactNode;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [inView, setInView] = useState(false);
  const [stuck, setStuck] = useState(false); // keepMounted: has it been shown once?

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    // Safety: if IntersectionObserver isn't available, just render eagerly.
    if (typeof IntersectionObserver === "undefined") {
      setInView(true);
      return;
    }
    const io = new IntersectionObserver(
      (entries) => {
        const visible = entries[0]?.isIntersecting ?? false;
        setInView(visible);
        if (visible && keepMounted) setStuck(true); // mount once, then never tear down
      },
      { rootMargin: "800px 0px", threshold: 0 }
    );
    io.observe(el);
    return () => io.disconnect();
  }, [keepMounted]);

  const show = inView || stuck;
  return (
    <div ref={ref} style={{ minHeight }}>
      {show ? (
        children
      ) : (
        <div
          style={{ minHeight }}
          className="flex items-center justify-center rounded-xl2 border border-dashed border-line bg-canvas/40 text-xs text-faint"
        >
          {label} loads on scroll
        </div>
      )}
    </div>
  );
}
