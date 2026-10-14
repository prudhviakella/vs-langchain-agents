import { Component, type ReactNode } from "react";

// Catches render errors in a subtree (e.g. a malformed artifact from a live
// model answer) and shows a fallback instead of taking down the whole app.
export class ErrorBoundary extends Component<
  { fallback?: ReactNode; children: ReactNode },
  { hasError: boolean }
> {
  state = { hasError: false };
  static getDerivedStateFromError() { return { hasError: true }; }
  componentDidCatch(err: unknown) { if (import.meta.env.DEV) console.warn("Artifact render error:", err); }
  render() { return this.state.hasError ? (this.props.fallback ?? null) : this.props.children; }
}
