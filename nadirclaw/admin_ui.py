"""Admin UI helpers for NadirClaw.

Contains session handling and HTML rendering for:
- /admin login page
- /admin settings dashboard
"""

import json
import os
import time
import uuid
from html import escape
from threading import Lock
from typing import Any, Dict, Optional
from urllib.parse import parse_qs

from fastapi import Request
from fastapi.responses import HTMLResponse

_ADMIN_SESSION_COOKIE = "nadirclaw_admin_session"
_ADMIN_SESSION_TTL_SECONDS = 8 * 60 * 60
_admin_sessions: Dict[str, float] = {}
_admin_sessions_lock = Lock()


def admin_session_cookie_name() -> str:
    """Return the cookie name used for admin sessions."""
    return _ADMIN_SESSION_COOKIE


def admin_session_ttl_seconds() -> int:
    """Return admin session TTL in seconds."""
    return _ADMIN_SESSION_TTL_SECONDS


def get_admin_password(auth_token: str) -> str:
    """Get admin password from env, with auth token fallback."""
    return os.getenv("NADIRCLAW_ADMIN_PASSWORD", "").strip() or auth_token


def create_admin_session() -> str:
    """Create and store a new admin session token."""
    token = uuid.uuid4().hex
    now = time.time()
    expires_at = now + _ADMIN_SESSION_TTL_SECONDS

    with _admin_sessions_lock:
        expired = [k for k, v in _admin_sessions.items() if v <= now]
        for k in expired:
            _admin_sessions.pop(k, None)
        _admin_sessions[token] = expires_at

    return token


def is_admin_authenticated(request: Request) -> bool:
    """Check whether request has a valid admin session cookie."""
    token = request.cookies.get(_ADMIN_SESSION_COOKIE, "")
    if not token:
        return False

    now = time.time()
    with _admin_sessions_lock:
        expires_at = _admin_sessions.get(token)
        if not expires_at:
            return False
        if expires_at <= now:
            _admin_sessions.pop(token, None)
            return False
        return True


def pop_admin_session(token: str) -> None:
    """Remove a session token if present."""
    if not token:
        return
    with _admin_sessions_lock:
        _admin_sessions.pop(token, None)


def parse_form_body(body: bytes) -> Dict[str, str]:
    """Parse x-www-form-urlencoded body into single-value dict."""
    decoded = body.decode("utf-8", errors="ignore")
    return {k: v[0] for k, v in parse_qs(decoded, keep_blank_values=True).items()}


def render_admin_login(error: str = "") -> HTMLResponse:
    """Render admin login page."""
    message_html = (
        f'<div class="alert alert-error">{escape(error)}</div>' if error else ""
    )
    html = f"""
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
        <title>NadirClaw Admin Login</title>
        <style>
            :root {{
                --bg: #0b1020;
                --panel: #121a33;
                --panel-border: #24325c;
                --text: #eef2ff;
                --muted: #9dadcf;
                --primary: #4f8cff;
                --primary-hover: #3d77df;
                --danger-bg: #3d1b22;
                --danger-border: #8a3041;
                --danger-text: #ffd5dc;
            }}
            * {{ box-sizing: border-box; }}
            body {{
                margin: 0;
                font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background: radial-gradient(1200px 600px at 20% -10%, #1a2853 0%, var(--bg) 55%);
                color: var(--text);
            }}
            .container {{
                max-width: 560px;
                margin: 48px auto;
                padding: 0 16px;
            }}
            .card {{
                background: var(--panel);
                border: 1px solid var(--panel-border);
                border-radius: 14px;
                padding: 24px;
                box-shadow: 0 12px 28px rgba(0, 0, 0, 0.28);
            }}
            h1 {{ margin: 0 0 8px 0; }}
            .subtitle {{ color: var(--muted); margin: 0 0 18px 0; }}
            label {{ display: block; margin-bottom: 8px; font-weight: 600; }}
            input {{
                width: 100%;
                padding: 11px 12px;
                border-radius: 10px;
                border: 1px solid var(--panel-border);
                background: #0f1730;
                color: var(--text);
            }}
            input:focus {{ outline: 2px solid rgba(79, 140, 255, 0.35); border-color: var(--primary); }}
            .hint {{ margin-top: 8px; font-size: 12px; color: var(--muted); }}
            button {{
                margin-top: 16px;
                width: 100%;
                border: 0;
                border-radius: 10px;
                background: var(--primary);
                color: #fff;
                padding: 11px 14px;
                font-weight: 700;
                cursor: pointer;
            }}
            button:hover {{ background: var(--primary-hover); }}
            .alert {{
                margin: 0 0 14px 0;
                border-radius: 10px;
                padding: 10px 12px;
                font-size: 14px;
            }}
            .alert-error {{
                background: var(--danger-bg);
                border: 1px solid var(--danger-border);
                color: var(--danger-text);
            }}
        </style>
    </head>
    <body>
        <main class="container">
            <section class="card">
                <h1>NadirClaw Admin</h1>
                <p class="subtitle">Sign in to manage runtime settings.</p>
                {message_html}
                <form method=\"post\" action=\"/admin/login\">
                    <label for=\"password\">Password</label>
                    <input id=\"password\" type=\"password\" name=\"password\" required autocomplete=\"current-password\" />
                    <p class="hint">Use <code>NADIRCLAW_ADMIN_PASSWORD</code> (or <code>NADIRCLAW_AUTH_TOKEN</code> as fallback).</p>
                    <button type=\"submit\">Login</button>
                </form>
            </section>
        </main>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


def render_admin_settings(
    result: Optional[Dict[str, Any]],
    settings_obj: Any,
) -> HTMLResponse:
    """Render admin settings dashboard page."""

    def current(env_var: str) -> str:
        return escape(os.getenv(env_var, ""), quote=True)

    result_html = ""
    if result is not None:
        result_pretty = escape(json.dumps(result, indent=2), quote=False)
        result_html = (
            "<h3>Last update result</h3>"
            f"<pre class='result-box'>{result_pretty}</pre>"
        )

    default_ollama = escape(settings_obj.OLLAMA_API_BASE, quote=True)

    html = f"""
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
        <title>NadirClaw Admin Settings</title>
        <style>
            :root {{
                --bg: #0b1020;
                --panel: #121a33;
                --panel-border: #24325c;
                --text: #eef2ff;
                --muted: #9dadcf;
                --primary: #4f8cff;
                --primary-hover: #3d77df;
                --chip: #1a2547;
                --chip-border: #2d3d70;
            }}
            * {{ box-sizing: border-box; }}
            body {{
                margin: 0;
                font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background: radial-gradient(1200px 600px at 20% -10%, #1a2853 0%, var(--bg) 55%);
                color: var(--text);
            }}
            .container {{ max-width: 1040px; margin: 24px auto; padding: 0 16px 24px; }}
            .card {{
                background: var(--panel);
                border: 1px solid var(--panel-border);
                border-radius: 14px;
                padding: 22px;
                box-shadow: 0 12px 28px rgba(0, 0, 0, 0.28);
            }}
            .header {{ display:flex; justify-content:space-between; align-items:flex-start; gap: 12px; }}
            h1 {{ margin: 0 0 8px 0; }}
            h2 {{ margin: 16px 0 8px 0; font-size: 16px; }}
            .subtitle {{ color: var(--muted); margin: 0 0 16px 0; }}
            .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
            .field {{ display: flex; flex-direction: column; gap: 6px; }}
            label {{ font-weight: 600; }}
            .hint {{ font-size: 12px; color: var(--muted); }}
            input {{
                padding: 10px 11px;
                border-radius: 10px;
                border: 1px solid var(--panel-border);
                background: #0f1730;
                color: var(--text);
            }}
            input:focus {{ outline: 2px solid rgba(79, 140, 255, 0.35); border-color: var(--primary); }}
            .actions {{ margin-top: 16px; display:flex; gap:10px; flex-wrap:wrap; align-items:center; }}
            button {{
                border: 0;
                border-radius: 10px;
                background: var(--primary);
                color: #fff;
                padding: 10px 14px;
                font-weight: 700;
                cursor: pointer;
            }}
            button:hover {{ background: var(--primary-hover); }}
            .button-secondary {{ background: transparent; border: 1px solid var(--panel-border); color: var(--text); }}
            .section {{ margin-top: 14px; padding-top: 10px; border-top: 1px dashed var(--panel-border); }}
            .chip {{
                display:inline-block;
                padding: 5px 8px;
                border-radius: 999px;
                border: 1px solid var(--chip-border);
                background: var(--chip);
                color: var(--muted);
                font-size: 12px;
            }}
            .result-box {{
                white-space: pre-wrap;
                background: #0f1730;
                border: 1px solid var(--panel-border);
                border-radius: 10px;
                padding: 12px;
                overflow: auto;
                color: #d8e1ff;
            }}
            @media (max-width: 880px) {{ .grid {{ grid-template-columns: 1fr; }} }}
        </style>
    </head>
    <body>
        <main class="container">
            <section class="card">
                <div class="header">
                    <div>
                        <h1>NadirClaw Admin Settings</h1>
                        <p class="subtitle">Configure routing, auth, providers, logging, and telemetry.</p>
                        <span class="chip">Leave key/token fields empty to keep current secrets</span>
                    </div>
                    <form method=\"post\" action=\"/admin/logout\">
                        <button class="button-secondary" type=\"submit\">Logout</button>
                    </form>
                </div>

                <form method=\"post\" action=\"/admin/settings\">
                    <div class="section">
                        <h2>Routing Models</h2>
                        <div class=\"grid\">
                            <div class=\"field\"><label>nadirclaw_simple_model</label><input name=\"nadirclaw_simple_model\" value=\"{current('NADIRCLAW_SIMPLE_MODEL')}\" placeholder=\"gemini-2.5-flash\" /></div>
                            <div class=\"field\"><label>nadirclaw_complex_model</label><input name=\"nadirclaw_complex_model\" value=\"{current('NADIRCLAW_COMPLEX_MODEL')}\" placeholder=\"gpt-4.1\" /></div>
                            <div class=\"field\"><label>nadirclaw_reasoning_model</label><input name=\"nadirclaw_reasoning_model\" value=\"{current('NADIRCLAW_REASONING_MODEL')}\" placeholder=\"o3\" /></div>
                            <div class=\"field\"><label>nadirclaw_free_model</label><input name=\"nadirclaw_free_model\" value=\"{current('NADIRCLAW_FREE_MODEL')}\" placeholder=\"ollama/llama3.1:8b\" /></div>
                            <div class=\"field\"><label>nadirclaw_models</label><input name=\"nadirclaw_models\" value=\"{current('NADIRCLAW_MODELS')}\" placeholder=\"comma-separated fallback list\" /></div>
                            <div class=\"field\"><label>nadirclaw_confidence_threshold</label><input name=\"nadirclaw_confidence_threshold\" value=\"{current('NADIRCLAW_CONFIDENCE_THRESHOLD')}\" placeholder=\"0.06\" /></div>
                        </div>
                    </div>

                    <div class="section">
                        <h2>Providers & Authentication</h2>
                        <div class=\"grid\">
                            <div class=\"field\"><label>ollama_api_base</label><input name=\"ollama_api_base\" value=\"{current('OLLAMA_API_BASE') or default_ollama}\" placeholder=\"http://localhost:11434\" /></div>
                            <div class=\"field\"><label>nadirclaw_auth_token</label><input type=\"password\" name=\"nadirclaw_auth_token\" placeholder=\"unchanged unless provided\" /></div>
                            <div class=\"field\"><label>gemini_api_key</label><input type=\"password\" name=\"gemini_api_key\" placeholder=\"unchanged unless provided\" /></div>
                            <div class=\"field\"><label>anthropic_api_key</label><input type=\"password\" name=\"anthropic_api_key\" placeholder=\"unchanged unless provided\" /></div>
                            <div class=\"field\"><label>openai_api_key</label><input type=\"password\" name=\"openai_api_key\" placeholder=\"unchanged unless provided\" /></div>
                        </div>
                    </div>

                    <div class="section">
                        <h2>Server & Observability</h2>
                        <div class=\"grid\">
                            <div class=\"field\"><label>nadirclaw_port</label><input name=\"nadirclaw_port\" value=\"{current('NADIRCLAW_PORT')}\" placeholder=\"8856\" /></div>
                            <div class=\"field\"><label>nadirclaw_log_dir</label><input name=\"nadirclaw_log_dir\" value=\"{current('NADIRCLAW_LOG_DIR')}\" placeholder=\"~/.nadirclaw/logs\" /></div>
                            <div class=\"field\"><label>nadirclaw_log_raw</label><input name=\"nadirclaw_log_raw\" value=\"{current('NADIRCLAW_LOG_RAW')}\" placeholder=\"true or false\" /></div>
                            <div class=\"field\"><label>otel_exporter_otlp_endpoint</label><input name=\"otel_exporter_otlp_endpoint\" value=\"{current('OTEL_EXPORTER_OTLP_ENDPOINT')}\" placeholder=\"http://localhost:4317\" /></div>
                        </div>
                    </div>

                    <div class=\"actions\">
                        <label><input type=\"checkbox\" name=\"fetch_models\" checked /> fetch_models</label>
                        <button type=\"submit\">Save settings</button>
                        <span class="hint">Saves to <code>~/.nadirclaw/.env</code> and applies immediately.</span>
                    </div>
                </form>

                {result_html}
            </section>
        </main>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
