"""Tests for nadirclaw.report — log parsing and report generation."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from nadirclaw.report import (
    format_report_text,
    generate_report,
    load_log_entries,
    parse_since,
)


# ---------------------------------------------------------------------------
# parse_since
# ---------------------------------------------------------------------------

class TestParseSince:
    def test_hours(self):
        now = datetime.now(timezone.utc)
        result = parse_since("24h")
        assert abs((now - timedelta(hours=24) - result).total_seconds()) < 2

    def test_days(self):
        now = datetime.now(timezone.utc)
        result = parse_since("7d")
        assert abs((now - timedelta(days=7) - result).total_seconds()) < 2

    def test_minutes(self):
        now = datetime.now(timezone.utc)
        result = parse_since("30m")
        assert abs((now - timedelta(minutes=30) - result).total_seconds()) < 2

    def test_iso_date(self):
        result = parse_since("2025-02-01")
        assert result == datetime(2025, 2, 1, tzinfo=timezone.utc)

    def test_iso_datetime(self):
        result = parse_since("2025-02-01T12:00:00")
        assert result == datetime(2025, 2, 1, 12, 0, 0, tzinfo=timezone.utc)

    def test_invalid(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_since("yesterday")

    def test_whitespace(self):
        result = parse_since("  7d  ")
        now = datetime.now(timezone.utc)
        assert abs((now - timedelta(days=7) - result).total_seconds()) < 2


# ---------------------------------------------------------------------------
# load_log_entries
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, entries: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


class TestLoadLogEntries:
    def test_basic_load(self, tmp_path):
        log = tmp_path / "requests.jsonl"
        entries = [
            {"type": "completion", "selected_model": "gpt-4", "timestamp": "2025-06-01T10:00:00+00:00"},
            {"type": "classify", "selected_model": "gemini", "timestamp": "2025-06-01T11:00:00+00:00"},
        ]
        _write_jsonl(log, entries)
        result = load_log_entries(log)
        assert len(result) == 2

    def test_missing_file(self, tmp_path):
        result = load_log_entries(tmp_path / "missing.jsonl")
        assert result == []

    def test_malformed_lines(self, tmp_path):
        log = tmp_path / "requests.jsonl"
        log.write_text('{"type": "ok"}\nthis is not json\n{"type": "also_ok"}\n')
        result = load_log_entries(log)
        assert len(result) == 2

    def test_since_filter(self, tmp_path):
        log = tmp_path / "requests.jsonl"
        entries = [
            {"type": "old", "timestamp": "2025-01-01T00:00:00+00:00"},
            {"type": "new", "timestamp": "2025-06-15T00:00:00+00:00"},
        ]
        _write_jsonl(log, entries)
        since = datetime(2025, 6, 1, tzinfo=timezone.utc)
        result = load_log_entries(log, since=since)
        assert len(result) == 1
        assert result[0]["type"] == "new"

    def test_model_filter(self, tmp_path):
        log = tmp_path / "requests.jsonl"
        entries = [
            {"selected_model": "gemini/gemini-2.5-flash", "timestamp": "2025-06-01T00:00:00+00:00"},
            {"selected_model": "gpt-4o", "timestamp": "2025-06-01T00:00:00+00:00"},
        ]
        _write_jsonl(log, entries)
        result = load_log_entries(log, model_filter="gemini")
        assert len(result) == 1
        assert "gemini" in result[0]["selected_model"]

    def test_model_filter_case_insensitive(self, tmp_path):
        log = tmp_path / "requests.jsonl"
        entries = [{"selected_model": "GPT-4o", "timestamp": "2025-06-01T00:00:00+00:00"}]
        _write_jsonl(log, entries)
        result = load_log_entries(log, model_filter="gpt")
        assert len(result) == 1

    def test_empty_lines_skipped(self, tmp_path):
        log = tmp_path / "requests.jsonl"
        log.write_text('{"type": "a"}\n\n\n{"type": "b"}\n')
        result = load_log_entries(log)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_empty(self):
        report = generate_report([])
        assert report["total_requests"] == 0
        assert report["time_range"] is None

    def test_basic_counts(self):
        entries = [
            {
                "type": "completion",
                "selected_model": "gpt-4",
                "tier": "complex",
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_latency_ms": 200,
                "classifier_latency_ms": 10,
                "timestamp": "2025-06-01T10:00:00+00:00",
            },
            {
                "type": "completion",
                "selected_model": "gemini-flash",
                "tier": "simple",
                "prompt_tokens": 20,
                "completion_tokens": 30,
                "total_latency_ms": 100,
                "classifier_latency_ms": 5,
                "timestamp": "2025-06-01T11:00:00+00:00",
            },
        ]
        report = generate_report(entries)
        assert report["total_requests"] == 2
        assert report["requests_by_type"]["completion"] == 2
        assert report["tokens"]["prompt_tokens"] == 120
        assert report["tokens"]["completion_tokens"] == 80
        assert report["tokens"]["total_tokens"] == 200

    def test_tier_distribution(self):
        entries = [
            {"tier": "simple", "timestamp": "2025-06-01T00:00:00+00:00"},
            {"tier": "simple", "timestamp": "2025-06-01T00:00:00+00:00"},
            {"tier": "complex", "timestamp": "2025-06-01T00:00:00+00:00"},
        ]
        report = generate_report(entries)
        assert report["tier_distribution"]["simple"]["count"] == 2
        assert report["tier_distribution"]["complex"]["count"] == 1
        assert report["tier_distribution"]["simple"]["percentage"] == pytest.approx(66.7, abs=0.1)

    def test_model_usage(self):
        entries = [
            {"selected_model": "gpt-4", "prompt_tokens": 10, "completion_tokens": 5, "timestamp": "2025-06-01T00:00:00+00:00"},
            {"selected_model": "gpt-4", "prompt_tokens": 20, "completion_tokens": 15, "timestamp": "2025-06-01T00:00:00+00:00"},
        ]
        report = generate_report(entries)
        assert report["model_usage"]["gpt-4"]["requests"] == 2
        assert report["model_usage"]["gpt-4"]["total_tokens"] == 50

    def test_latency_stats(self):
        entries = [
            {"total_latency_ms": 100, "timestamp": "2025-06-01T00:00:00+00:00"},
            {"total_latency_ms": 200, "timestamp": "2025-06-01T00:00:00+00:00"},
            {"total_latency_ms": 300, "timestamp": "2025-06-01T00:00:00+00:00"},
        ]
        report = generate_report(entries)
        assert report["latency"]["total"]["avg"] == pytest.approx(200.0)
        assert report["latency"]["total"]["p50"] == 200.0

    def test_fallback_and_errors(self):
        entries = [
            {"fallback_used": "gpt-4", "status": "ok", "timestamp": "2025-06-01T00:00:00+00:00"},
            {"status": "error", "timestamp": "2025-06-01T00:00:00+00:00"},
        ]
        report = generate_report(entries)
        assert report["fallback_count"] == 1
        assert report["error_count"] == 1

    def test_streaming_and_tools(self):
        entries = [
            {"stream": True, "has_tools": True, "tool_count": 3, "timestamp": "2025-06-01T00:00:00+00:00"},
            {"stream": False, "has_tools": False, "timestamp": "2025-06-01T00:00:00+00:00"},
        ]
        report = generate_report(entries)
        assert report["streaming_count"] == 1
        assert report["tool_usage"]["requests_with_tools"] == 1
        assert report["tool_usage"]["total_tool_count"] == 3

    def test_missing_fields(self):
        """Entries with missing fields should not crash."""
        entries = [
            {"timestamp": "2025-06-01T00:00:00+00:00"},
            {"type": "completion", "timestamp": "2025-06-01T00:00:00+00:00"},
        ]
        report = generate_report(entries)
        assert report["total_requests"] == 2
        assert report["tokens"]["total_tokens"] == 0


# ---------------------------------------------------------------------------
# format_report_text
# ---------------------------------------------------------------------------

class TestFormatReportText:
    def test_empty_report(self):
        report = generate_report([])
        text = format_report_text(report)
        assert "Total requests: 0" in text

    def test_includes_sections(self):
        entries = [
            {
                "type": "completion",
                "selected_model": "gpt-4",
                "tier": "complex",
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_latency_ms": 200,
                "timestamp": "2025-06-01T10:00:00+00:00",
            },
        ]
        report = generate_report(entries)
        text = format_report_text(report)
        assert "NadirClaw Report" in text
        assert "Requests by Type" in text
        assert "Tier Distribution" in text
        assert "Model Usage" in text
        assert "Token Usage" in text
        assert "gpt-4" in text
