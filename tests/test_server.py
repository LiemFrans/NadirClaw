"""Tests for nadirclaw.server — health endpoint and basic API contract."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client for the NadirClaw FastAPI app."""
    from nadirclaw.server import app
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "simple_model" in data
        assert "complex_model" in data

    def test_root_returns_info(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "NadirClaw"
        assert data["status"] == "ok"
        assert "version" in data


class TestModelsEndpoint:
    def test_list_models(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        assert len(data["data"]) >= 1
        # Each model should have an id
        for model in data["data"]:
            assert "id" in model
            assert model["object"] == "model"


class TestClassifyEndpoint:
    def test_classify_returns_classification(self, client):
        resp = client.post("/v1/classify", json={"prompt": "What is 2+2?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "classification" in data
        assert data["classification"]["tier"] in ("simple", "complex")
        assert "confidence" in data["classification"]
        assert "selected_model" in data["classification"]

    def test_classify_batch(self, client):
        resp = client.post(
            "/v1/classify/batch",
            json={"prompts": ["Hello", "Design a distributed system"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["results"]) == 2


class TestSetupWebhookEndpoint:
    def test_setup_webhook_updates_ollama_base_and_fetches_models(self, client, monkeypatch):
        upsert = MagicMock()
        monkeypatch.setattr("nadirclaw.server._upsert_nadirclaw_env_var", upsert)

        monkeypatch.setattr(
            "nadirclaw.setup.fetch_provider_models",
            lambda provider, credential, ollama_api_base="": ["ollama/llama3.2:3b"]
            if provider == "ollama"
            else [],
        )

        resp = client.post(
            "/v1/setup/webhook",
            json={"ollama_api_base": "10.4.136.145:11434", "fetch_models": True},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["ollama_api_base"] == "http://10.4.136.145:11434"
        assert data["models"] == ["ollama/llama3.2:3b"]
        assert data["model_count"] == 1
        upsert.assert_called_once_with("OLLAMA_API_BASE", "http://10.4.136.145:11434")

    def test_setup_webhook_can_skip_model_fetch(self, client, monkeypatch):
        upsert = MagicMock()
        monkeypatch.setattr("nadirclaw.server._upsert_nadirclaw_env_var", upsert)

        called = {"count": 0}

        def _fake_fetch(provider, credential, ollama_api_base=""):
            called["count"] += 1
            return ["ollama/llama3.2:3b"]

        monkeypatch.setattr("nadirclaw.setup.fetch_provider_models", _fake_fetch)

        resp = client.post(
            "/v1/setup/webhook",
            json={"ollama_api_base": "http://localhost:11434", "fetch_models": False},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["models"] == []
        assert data["model_count"] == 0
        assert called["count"] == 0
        upsert.assert_called_once_with("OLLAMA_API_BASE", "http://localhost:11434")

    def test_setup_webhook_updates_env_map_and_ignores_unknown_keys(self, client, monkeypatch):
        upsert = MagicMock()
        monkeypatch.setattr("nadirclaw.server._upsert_nadirclaw_env_var", upsert)

        resp = client.post(
            "/v1/setup/webhook",
            json={
                "env": {
                    "nadirclaw_simple_model": "gemini-2.5-flash",
                    "nadirclaw_complex_model": "gpt-4.1",
                    "nadirclaw_log_raw": True,
                    "UNKNOWN_KEY": "ignored",
                },
                "fetch_models": False,
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["updated"]["NADIRCLAW_SIMPLE_MODEL"] == "gemini-2.5-flash"
        assert data["updated"]["NADIRCLAW_COMPLEX_MODEL"] == "gpt-4.1"
        assert data["updated"]["NADIRCLAW_LOG_RAW"] == "true"
        assert "UNKNOWN_KEY" in data["ignored"]

        expected_calls = {
            ("NADIRCLAW_SIMPLE_MODEL", "gemini-2.5-flash"),
            ("NADIRCLAW_COMPLEX_MODEL", "gpt-4.1"),
            ("NADIRCLAW_LOG_RAW", "true"),
            ("OLLAMA_API_BASE", "http://localhost:11434"),
        }
        actual_calls = {(c.args[0], c.args[1]) for c in upsert.call_args_list}
        assert expected_calls.issubset(actual_calls)


class TestAdminWebUI:
    def test_admin_page_shows_login_when_not_authenticated(self, client):
        resp = client.get("/admin")
        assert resp.status_code == 200
        assert "NadirClaw Admin" in resp.text
        assert "Login" in resp.text

    def test_admin_login_success_sets_cookie(self, client, monkeypatch):
        monkeypatch.setenv("NADIRCLAW_ADMIN_PASSWORD", "secret")
        resp = client.post(
            "/admin/login",
            data={"password": "secret"},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert resp.headers.get("location") == "/admin"
        assert "nadirclaw_admin_session=" in resp.headers.get("set-cookie", "")

    def test_admin_settings_requires_auth(self, client):
        resp = client.post("/admin/settings", data={}, follow_redirects=False)
        assert resp.status_code == 200
        assert "Session expired" in resp.text or "NadirClaw Admin" in resp.text

    def test_admin_settings_update_calls_setup_logic(self, client, monkeypatch):
        monkeypatch.setenv("NADIRCLAW_ADMIN_PASSWORD", "secret")
        monkeypatch.setattr(
            "nadirclaw.server._apply_setup_updates",
            lambda payload: {
                "status": "ok",
                "updated": payload.env or {},
                "models": [],
                "model_count": 0,
                "ollama_api_base": payload.ollama_api_base or "http://localhost:11434",
                "ignored": [],
                "env_file": "~/.nadirclaw/.env",
            },
        )

        login = client.post("/admin/login", data={"password": "secret"}, follow_redirects=False)
        assert login.status_code == 303

        resp = client.post(
            "/admin/settings",
            data={
                "nadirclaw_simple_model": "gemini-2.5-flash",
                "nadirclaw_complex_model": "gpt-4.1",
                "fetch_models": "on",
            },
        )
        assert resp.status_code == 200
        assert "Last update result" in resp.text
        assert "gemini-2.5-flash" in resp.text


class TestTierFailoverPools:
    def test_build_failover_candidates_prefers_same_tier_then_cross_tier(self, monkeypatch):
        monkeypatch.setenv("NADIRCLAW_SIMPLE_MODEL", "simple-a,simple-b")
        monkeypatch.setenv("NADIRCLAW_COMPLEX_MODEL", "complex-a,complex-b")

        from nadirclaw.server import _build_failover_candidates

        candidates = _build_failover_candidates("simple-a", "simple")
        assert candidates[0] == "simple-a"
        assert candidates[1] == "simple-b"
        assert "complex-a" in candidates
        assert "complex-b" in candidates

    def test_build_failover_candidates_reasoning_includes_reasoning_then_complex(self, monkeypatch):
        monkeypatch.setenv("NADIRCLAW_REASONING_MODEL", "o3,o4-mini")
        monkeypatch.setenv("NADIRCLAW_COMPLEX_MODEL", "gpt-4.1")

        from nadirclaw.server import _build_failover_candidates

        candidates = _build_failover_candidates("o3", "reasoning")
        assert candidates[0] == "o3"
        assert candidates[1] == "o4-mini"
        assert "gpt-4.1" in candidates
