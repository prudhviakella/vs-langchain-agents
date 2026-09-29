"""OpenTelemetry helpers for trial_graph — named spans, and trace context across agents.

    analyst -> supervisor ──invoke_agent_runtime(payload: A2A message)──► specialist
                  │          message.metadata = {traceparent, ...}      │
          inject_trace_headers()                  extract_parent_context(metadata)
          current span -> W3C traceparent         + attach_context() in execute()
                  │                                                     │
                  └──────────── one trace_id across every agent ────────┘

TRANSPORT IS NOT HERE

Exporting spans is owned by `opentelemetry-instrument` (aws-opentelemetry-distro),
which wraps the process in the Dockerfile CMD. On AgentCore Runtime the OTEL
environment variables are preset by the runtime; spans land in CloudWatch
(aws/spans) and the GenAI Observability dashboard — once CloudWatch Transaction
Search is enabled for the account (infra/observability.py).

WHY PROPAGATION IS EXPLICIT, AND WHY IT RIDES IN THE MESSAGE

Nothing carries trace context from the supervisor to a specialist unless we
do. The A2A executor runs where no automatic context threading reaches, so
the context is extracted and attached by hand, and detached in a finally.

It travels in the A2A message metadata, NOT in invoke_agent_runtime's
traceParent / traceState / baggage parameters. Those become SigV4-signed
headers — botocore exempts only x-amzn-trace-id from signing, because
tracing infrastructure rewrites trace headers in transit — and in AWS every
specialist call then failed "The request signature we calculated does not
match the signature you provided". The message body is signed too, but
nothing on the path rewrites it. Executors still accept an inbound
traceparent header first, for callers that send one.

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
    _tracer = _otel_trace.get_tracer("trial_graph.agent")
except Exception:                                   # opentelemetry not installed
    _tracer = _otel_context = _otel_propagate = _otel_trace = None

PREFIX = "trial_graph"


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
    """A named span, attributes prefixed "trial_graph.". Values must be
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


# The attributes Bedrock AgentCore's GenAI Observability classifies spans by.
# Its documentation: a span is an agent span when gen_ai.operation.name is
# invoke_agent (or openinference.span.kind is AGENT / CHAIN), and the agent's
# name is gen_ai.agent.name. The OpenTelemetry GenAI conventions name that
# span "invoke_agent <agent name>". Without these, the agent's root span is
# invisible to the "Agent spans" filter and shows no agent name.
@contextmanager
def agent_span(agent: str, conversation_id: str | None = None, **attributes):
    """The root span of one agent run, named and classified the way AgentCore
    Observability reads it. Extra attributes are prefixed like span()'s."""
    with span(f"invoke_agent {agent}", **attributes) as sp:
        if sp is not None:
            for key, value in {"gen_ai.operation.name": "invoke_agent",
                               "gen_ai.agent.name": agent,
                               "gen_ai.provider.name": "openai",
                               "gen_ai.conversation.id": conversation_id,
                               "openinference.span.kind": "AGENT"}.items():
                if value is not None:
                    try:
                        sp.set_attribute(key, value)
                    except Exception:
                        pass
        yield sp


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
