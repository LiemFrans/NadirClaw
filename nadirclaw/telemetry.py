"""Optional OpenTelemetry integration for NadirClaw.

All exports are no-ops if opentelemetry packages are not installed.
Install with: pip install nadirclaw[telemetry]
"""

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Optional

logger = logging.getLogger("nadirclaw.telemetry")

# Try to import OpenTelemetry — all functionality degrades gracefully
_otel_available = False
_tracer = None

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    _otel_available = True
except ImportError:
    pass


def is_enabled() -> bool:
    """Return True if OpenTelemetry is active and configured."""
    return _tracer is not None


def setup_telemetry(service_name: str = "nadirclaw") -> bool:
    """Initialize OpenTelemetry tracing if packages are installed and endpoint is set.

    Returns True if telemetry was successfully initialized.
    """
    global _tracer

    if not _otel_available:
        logger.debug("OpenTelemetry packages not installed — telemetry disabled")
        return False

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        logger.debug("OTEL_EXPORTER_OTLP_ENDPOINT not set — telemetry disabled")
        return False

    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("nadirclaw")
        logger.info("OpenTelemetry tracing enabled — exporting to %s", endpoint)
        return True
    except Exception as e:
        logger.warning("Failed to initialize OpenTelemetry: %s", e)
        return False


def instrument_fastapi(app: Any) -> bool:
    """Auto-instrument a FastAPI app with OpenTelemetry HTTP spans.

    Returns True if instrumentation was applied.
    """
    if not _otel_available or not is_enabled():
        return False

    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI auto-instrumentation enabled")
        return True
    except ImportError:
        logger.debug("opentelemetry-instrumentation-fastapi not installed")
        return False
    except Exception as e:
        logger.warning("Failed to instrument FastAPI: %s", e)
        return False


@contextmanager
def trace_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """Context manager that creates an OpenTelemetry span.

    Yields the span object, or None if telemetry is not active.
    """
    if not is_enabled() or _tracer is None:
        yield None
        return

    with _tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(key, _safe_attribute(value))
        yield span


def record_llm_call(
    span: Any,
    model: str,
    provider: Optional[str] = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    tier: Optional[str] = None,
    latency_ms: Optional[float] = None,
) -> None:
    """Record GenAI semantic convention attributes on a span.

    Safe to call with span=None (no-op).
    """
    if span is None:
        return

    try:
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.usage.input_tokens", prompt_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", completion_tokens)

        if provider:
            span.set_attribute("gen_ai.system", provider)
        if tier:
            span.set_attribute("nadirclaw.tier", tier)
        if latency_ms is not None:
            span.set_attribute("nadirclaw.latency_ms", latency_ms)
    except Exception:
        pass  # Never crash on telemetry


def _safe_attribute(value: Any) -> Any:
    """Convert a value to an OTel-safe attribute type."""
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
