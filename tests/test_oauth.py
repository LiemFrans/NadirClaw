"""Tests for nadirclaw.oauth — PKCE helpers, token validation, config resolution."""

import base64
import hashlib

import pytest

from nadirclaw.oauth import (
    _generate_code_challenge,
    _generate_code_verifier,
    validate_anthropic_setup_token,
)


class TestPKCE:
    def test_verifier_length(self):
        verifier = _generate_code_verifier()
        assert 43 <= len(verifier) <= 128

    def test_verifier_is_url_safe(self):
        verifier = _generate_code_verifier()
        # Should only contain URL-safe base64 characters (no padding)
        assert "=" not in verifier
        assert "+" not in verifier
        assert "/" not in verifier

    def test_challenge_matches_verifier(self):
        verifier = _generate_code_verifier()
        challenge = _generate_code_challenge(verifier)

        # Manually compute expected challenge
        digest = hashlib.sha256(verifier.encode("utf-8")).digest()
        expected = base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")
        assert challenge == expected

    def test_different_verifiers_produce_different_challenges(self):
        v1 = _generate_code_verifier()
        v2 = _generate_code_verifier()
        assert v1 != v2
        assert _generate_code_challenge(v1) != _generate_code_challenge(v2)


class TestAnthropicSetupToken:
    def test_valid_token(self):
        token = "sk-ant-oat01-" + "x" * 80
        assert validate_anthropic_setup_token(token) is None

    def test_empty_token(self):
        error = validate_anthropic_setup_token("")
        assert error is not None
        assert "empty" in error.lower()

    def test_wrong_prefix(self):
        error = validate_anthropic_setup_token("sk-ant-wrong-" + "x" * 80)
        assert error is not None
        assert "sk-ant-oat01-" in error

    def test_too_short(self):
        error = validate_anthropic_setup_token("sk-ant-oat01-short")
        assert error is not None
        assert "short" in error.lower()

    def test_whitespace_trimmed(self):
        token = "  sk-ant-oat01-" + "x" * 80 + "  "
        assert validate_anthropic_setup_token(token) is None


class TestGeminiClientConfig:
    def test_env_var_override(self, monkeypatch):
        from nadirclaw.oauth import _resolve_gemini_client_config

        monkeypatch.setenv("NADIRCLAW_GEMINI_OAUTH_CLIENT_ID", "test-client-id")
        monkeypatch.setenv("NADIRCLAW_GEMINI_OAUTH_CLIENT_SECRET", "test-secret")

        config = _resolve_gemini_client_config()
        assert config["client_id"] == "test-client-id"
        assert config["client_secret"] == "test-secret"

    def test_no_gemini_cli_returns_empty(self, monkeypatch):
        from nadirclaw.oauth import _resolve_gemini_client_config

        # Clear all env vars
        for key in (
            "NADIRCLAW_GEMINI_OAUTH_CLIENT_ID",
            "OPENCLAW_GEMINI_OAUTH_CLIENT_ID",
            "GEMINI_CLI_OAUTH_CLIENT_ID",
        ):
            monkeypatch.delenv(key, raising=False)
        # Mock shutil.which to return None (no gemini CLI)
        monkeypatch.setattr("nadirclaw.oauth.shutil.which", lambda _: None)

        config = _resolve_gemini_client_config()
        assert config == {}
