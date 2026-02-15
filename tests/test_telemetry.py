"""Tests for nadirclaw.telemetry — no-op behavior without OTel packages."""

from nadirclaw.telemetry import is_enabled, record_llm_call, trace_span


class TestTelemetryNoOp:
    def test_is_enabled_false_by_default(self):
        """Without OTel configured, is_enabled() should return False."""
        assert is_enabled() is False

    def test_trace_span_yields_none(self):
        """trace_span should yield None when telemetry is not active."""
        with trace_span("test-span") as span:
            assert span is None

    def test_trace_span_with_attributes(self):
        """trace_span with attributes should not crash."""
        with trace_span("test-span", {"key": "value", "num": 42}) as span:
            assert span is None

    def test_record_llm_call_none_span(self):
        """record_llm_call with None span should not crash."""
        record_llm_call(
            span=None,
            model="gpt-4",
            provider="openai",
            prompt_tokens=100,
            completion_tokens=50,
            tier="complex",
            latency_ms=200.0,
        )

    def test_record_llm_call_minimal(self):
        """record_llm_call with minimal args should not crash."""
        record_llm_call(span=None, model="test-model")
