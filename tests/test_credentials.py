"""Tests for nadirclaw.credentials — save, load, detect provider, refresh."""

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from nadirclaw.credentials import (
    _credentials_path,
    _mask_token,
    _read_credentials,
    _write_credentials,
    detect_provider,
    get_credential,
    get_credential_source,
    list_credentials,
    remove_credential,
    save_credential,
    save_oauth_credential,
)


@pytest.fixture(autouse=True)
def tmp_credentials(tmp_path, monkeypatch):
    """Redirect credentials file to a temp directory for each test."""
    creds_file = tmp_path / "credentials.json"
    monkeypatch.setattr(
        "nadirclaw.credentials._credentials_path", lambda: creds_file
    )
    # Clear env vars that might interfere
    for var in (
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY",
        "GEMINI_API_KEY", "COHERE_API_KEY", "MISTRAL_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    return creds_file


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_and_get(self):
        save_credential("anthropic", "sk-ant-test-123", source="manual")
        assert get_credential("anthropic") == "sk-ant-test-123"

    def test_save_overwrites(self):
        save_credential("openai", "old-key")
        save_credential("openai", "new-key")
        assert get_credential("openai") == "new-key"

    def test_get_missing_returns_none(self):
        assert get_credential("nonexistent") is None

    def test_remove_existing(self):
        save_credential("openai", "key-123")
        assert remove_credential("openai") is True
        assert get_credential("openai") is None

    def test_remove_missing(self):
        assert remove_credential("openai") is False

    def test_credentials_file_permissions(self, tmp_credentials):
        """Credentials file should have 0o600 permissions on Unix."""
        import platform
        if platform.system() == "Windows":
            pytest.skip("Permission check not applicable on Windows")

        save_credential("test", "value")
        mode = tmp_credentials.stat().st_mode & 0o777
        assert mode == 0o600


# ---------------------------------------------------------------------------
# OAuth credentials
# ---------------------------------------------------------------------------

class TestOAuthCredentials:
    def test_save_oauth_credential(self):
        save_oauth_credential("openai-codex", "access-tok", "refresh-tok", 3600)
        assert get_credential("openai-codex") == "access-tok"
        assert get_credential_source("openai-codex") == "oauth"

    def test_oauth_with_metadata(self):
        save_oauth_credential(
            "antigravity", "access", "refresh", 3600,
            metadata={"project_id": "proj-123", "email": "user@test.com"},
        )
        creds = _read_credentials()
        entry = creds["antigravity"]
        assert entry["project_id"] == "proj-123"
        assert entry["email"] == "user@test.com"

    def test_expired_oauth_returns_none_on_refresh_failure(self):
        """Expired token with no refresh function should return None."""
        save_oauth_credential("openai-codex", "expired-tok", "bad-refresh", -100)
        # Token is expired, refresh will fail (mocked import)
        with patch("nadirclaw.credentials._get_refresh_func", return_value=None):
            # No refresh func → returns the stale token (warning only)
            token = get_credential("openai-codex")
            assert token == "expired-tok"


# ---------------------------------------------------------------------------
# Environment variable fallback
# ---------------------------------------------------------------------------

class TestEnvFallback:
    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
        assert get_credential("anthropic") == "sk-from-env"
        assert get_credential_source("anthropic") == "env"

    def test_stored_takes_precedence_over_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
        save_credential("anthropic", "sk-stored", source="manual")
        assert get_credential("anthropic") == "sk-stored"

    def test_gemini_fallback_env(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-gemini")
        assert get_credential("google") == "AIza-gemini"


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

class TestDetectProvider:
    @pytest.mark.parametrize("model,expected", [
        ("claude-sonnet-4-20250514", "anthropic"),
        ("anthropic/claude-3-opus", "anthropic"),
        ("gpt-4o", "openai"),
        ("openai/gpt-4", "openai"),
        ("o3-mini", "openai"),
        ("gemini-2.5-pro", "google"),
        ("gemini/gemini-3-flash", "google"),
        ("ollama/llama3", "ollama"),
        ("openai-codex/gpt-5.3-codex", "openai-codex"),
        ("unknown-model", None),
    ])
    def test_detect_provider(self, model, expected):
        assert detect_provider(model) == expected


# ---------------------------------------------------------------------------
# Token masking
# ---------------------------------------------------------------------------

class TestMaskToken:
    def test_short_token(self):
        assert _mask_token("abc") == "abc***"

    def test_long_token(self):
        masked = _mask_token("sk-ant-1234567890abcdef")
        assert masked.startswith("sk-ant-1")
        assert masked.endswith("cdef")
        assert "..." in masked


# ---------------------------------------------------------------------------
# List credentials
# ---------------------------------------------------------------------------

class TestListCredentials:
    def test_list_empty(self):
        assert list_credentials() == []

    def test_list_with_stored(self):
        save_credential("anthropic", "sk-ant-test-key", source="manual")
        result = list_credentials()
        assert len(result) >= 1
        anthropic = next(c for c in result if c["provider"] == "anthropic")
        assert anthropic["source"] == "manual"
        assert "***" in anthropic["masked_token"] or "..." in anthropic["masked_token"]
