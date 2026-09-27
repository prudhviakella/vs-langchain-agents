"""OpenTelemetry helpers for trial_search — named spans, and trace context across agents.

    analyst -> supervisor ──invoke_agent_runtime(traceParent=...)──► specialist
                  │                                                     │
          inject_trace_headers()                     extract_parent_context(headers)
          current span -> W3C traceparent            + attach_context() in execute()
                  │                                                     │
                  └──────────── one trace_id across every agent ────────┘

TRANSPORT IS NOT HERE

Exporting spans is owned by `opentelemetry-instrument` (aws-opentelemetry-distro),
which wraps the process in the Dockerfile CMD. On AgentCore Runtime the OTEL
environment variables are preset by the runtime; spans land in CloudWatch
(aws/spans) and the GenAI Observability dashboard — once CloudWatch Transaction
Search is enabled for the account (infra/observability.py).

WHY PROPAGATION IS EXPLICIT

The supervisor calls a specialist through boto3's invoke_agent_runtime, which
has first-class traceParent / traceState / baggage parameters: nothing fills
them unless we do. On the receiving side, bedrock_agentcore stores the W3C
headers (traceparent, tracestate, baggage pass its forwarding filter;
X-Amzn-Trace-Id does not), but the A2A executor runs where no automatic
context threading reaches — the reference found a sibling agent's spans
missing from the unified trace for exactly this reason, despite an identical
opentelemetry-instrument entrypoint. So the context is extracted and attached
by hand, and detached in a finally.

GRACEFUL, ALWAYS

Every function is a no-op when OpenTelemetry is unavailable or unconfigured,
and none of them raises. Telemetry must never stop an agent answering.
Exceptions inside span() are recorded on the span and then re-raised —
behaviour is identical with and without tracing.

WHAT THIS DOES NOT DO

    It does not configure a tracer provider or an exporter.
"""
from __future__ import annotations

from contextlib import contextmanager

try:
    from opentelemetry import context as _otel_context
    from opentelemetry import propagate as _otel_propagate
    from opentelemetry import trace as _otel_trace
    _tracer = _otel_trace.get_tracer("trial_search.agent")
except Exception:                                   # opentelemetry not installed
    _tracer = _otel_context = _otel_propagate = _otel_trace = None

PREFIX = "trial_search"


def extract_parent_context(headers: dict | None):
    """OTEL context from an inbound traceparent, or None. Header keys are
    normalised: HTTP/2 lowercases them, HTTP/1.1 may not."""
    if _otel_propagate is None or not headers:
        return None
    carrier = {str(k).lower(): v for k, v in headers.items()}
    if "traceparent" not in carrier:
        return None
    try:
        return _otel_propagate.extract(carrier)
    except Exception:
        return None


def attach_context(ctx):
    """Make `ctx` ambient. Returns a token for detach_context, or None."""
    if _otel_context is None or ctx is None:
        return None
    try:
        return _otel_context.attach(ctx)
    except Exception:
        return None


def detach_context(token) -> None:
    """Undo attach_context. Call in a finally: this is a long-lived server, and
    a leaked context would attribute later, unrelated requests to this trace."""
    if _otel_context is None or token is None:
        return
    try:
        _otel_context.detach(token)
    except Exception:
        pass


def inject_trace_headers() -> dict:
    """The current context as W3C headers — traceparent, and tracestate /
    baggage when present. Empty when there is no active trace."""
    if _otel_propagate is None:
        return {}
    carrier: dict = {}
    try:
        _otel_propagate.inject(carrier)
    except Exception:
        return {}
    return {k.lower(): v for k, v in carrier.items()
            if k.lower() in ("traceparent", "tracestate", "baggage")}


@contextmanager
def span(name: str, **attributes):
    """A named span, attributes prefixed "trial_search.". Values must be
    str/bool/int/float; None values are skipped."""
    if _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as sp:
        try:
            set_span_attrs(sp, **attributes)
            yield sp
        except Exception as exc:
            try:
                sp.record_exception(exc)
                from opentelemetry.trace import Status, StatusCode
                sp.set_status(Status(StatusCode.ERROR, str(exc)))
            except Exception:
                pass
            raise                                   # never swallow


def set_span_attrs(sp, **attributes) -> None:
    """Set attributes on an open span — for results known mid-phase."""
    if sp is None:
        return
    for key, value in attributes.items():
        if value is not None:
            try:
                sp.set_attribute(f"{PREFIX}.{key}", value)
            except Exception:
                pass


def shutdown() -> None:
    """Flush spans on exit. The default ProxyTracerProvider has no shutdown()."""
    try:
        provider = _otel_trace.get_tracer_provider() if _otel_trace else None
        if callable(getattr(provider, "shutdown", None)):
            provider.shutdown()
    except Exception:
        pass
