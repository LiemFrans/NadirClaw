"""Admin UI helpers for NadirClaw.

Contains session handling and HTML rendering for:
- /admin login page
- /admin settings dashboard
"""

import json
import os
import re
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
    def provider_for_model(model_name: str) -> str:
        m = (model_name or "").strip().lower()
        if not m:
            return "other"
        if "/" in m:
            return m.split("/", 1)[0]
        if m.startswith("gpt") or re.match(r"^o[1-9](?:-|$)", m):
            return "openai"
        if m.startswith("claude"):
            return "anthropic"
        if m.startswith("gemini"):
            return "google"
        if m.startswith("deepseek"):
            return "deepseek"
        return "other"

    def current(env_var: str) -> str:
        return escape(os.getenv(env_var, ""), quote=True)

    def split_model_for_admin(model_name: str) -> tuple[str, str]:
        """Split full model into (provider, model_without_prefix) for form UI."""
        m = (model_name or "").strip()
        if not m:
            return "openai", ""
        if "/" in m:
            provider, rest = m.split("/", 1)
            return provider.lower(), rest
        return provider_for_model(m), m

    def model_name_for_provider(provider: str, model_name: str) -> str:
        """Return model name without provider prefix for provider-specific UI."""
        p = (provider or "").lower().strip()
        m = (model_name or "").strip()
        if not m:
            return ""
        prefix = f"{p}/"
        if p and m.lower().startswith(prefix):
            return m[len(prefix):]
        return m

    def model_list_for_provider(provider: str, raw_list: str) -> str:
        """Convert full model list into provider-local comma-separated names."""
        items = [m.strip() for m in (raw_list or "").split(",") if m.strip()]
        normalized = [model_name_for_provider(provider, m) for m in items]
        return ",".join([m for m in normalized if m])

    def provider_options_html(selected: str) -> str:
        providers = [
            "ollama",
            "google",
            "openai",
            "anthropic",
            "deepseek",
            "openai-codex",
            "custom",
        ]
        selected_norm = (selected or "custom").lower()
        return "".join(
            f'<option value="{p}"{" selected" if p == selected_norm else ""}>{p}</option>'
            for p in providers
        )

    def model_options_html_for_provider(selected_provider: str, selected_name: str = "") -> str:
        provider = (selected_provider or "").lower().strip()
        selected_clean = (selected_name or "").strip()
        if not provider or provider == "custom":
            return (
                f'<option value="{escape(selected_clean, quote=True)}" selected>{escape(selected_clean)}</option>'
                if selected_clean
                else ""
            )

        names = set()
        for full_model in sorted_models:
            if provider_for_model(full_model) != provider:
                continue
            names.add(model_name_for_provider(provider, full_model))

        if selected_clean:
            names.add(selected_clean)

        return "".join(
            f'<option value="{escape(name, quote=True)}"{" selected" if name == selected_clean else ""}>{escape(name)}</option>'
            for name in sorted(names)
            if name
        )

    result_html = ""
    if result is not None:
        result_pretty = escape(json.dumps(result, indent=2), quote=False)
        result_html = (
            "<h3>Last update result</h3>"
            f"<pre class='result-box'>{result_pretty}</pre>"
        )

    default_ollama = escape(settings_obj.OLLAMA_API_BASE, quote=True)

    model_candidates = {
        settings_obj.SIMPLE_MODEL,
        settings_obj.COMPLEX_MODEL,
        settings_obj.REASONING_MODEL,
        settings_obj.FREE_MODEL,
    }

    raw_models = os.getenv("NADIRCLAW_MODELS", "")
    if raw_models:
        model_candidates.update(m.strip() for m in raw_models.split(",") if m.strip())

    try:
        from nadirclaw.routing import MODEL_REGISTRY

        model_candidates.update(MODEL_REGISTRY.keys())
    except Exception:
        pass

    sorted_models = sorted(
        [m for m in model_candidates if m],
        key=lambda m: (provider_for_model(m), m.lower()),
    )

    simple_full = os.getenv("NADIRCLAW_SIMPLE_MODEL", "") or settings_obj.SIMPLE_MODEL
    complex_full = os.getenv("NADIRCLAW_COMPLEX_MODEL", "") or settings_obj.COMPLEX_MODEL
    reasoning_full = os.getenv("NADIRCLAW_REASONING_MODEL", "") or settings_obj.REASONING_MODEL
    free_full = os.getenv("NADIRCLAW_FREE_MODEL", "") or settings_obj.FREE_MODEL

    simple_provider, simple_name = split_model_for_admin(simple_full)
    complex_provider, complex_name = split_model_for_admin(complex_full)
    reasoning_provider, reasoning_name = split_model_for_admin(reasoning_full)
    free_provider, free_name = split_model_for_admin(free_full)

    simple_list_value = model_list_for_provider(
        simple_provider,
        os.getenv("NADIRCLAW_SIMPLE_MODELS", simple_full),
    )
    complex_list_value = model_list_for_provider(
        complex_provider,
        os.getenv("NADIRCLAW_COMPLEX_MODELS", complex_full),
    )
    reasoning_list_value = model_list_for_provider(
        reasoning_provider,
        os.getenv("NADIRCLAW_REASONING_MODELS", reasoning_full),
    )
    free_list_value = model_list_for_provider(
        free_provider,
        os.getenv("NADIRCLAW_FREE_MODELS", free_full),
    )

    html = f"""
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
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
            .field-row {{ display: flex; gap: 8px; align-items: center; }}
            .field-row input {{ flex: 1; }}
            .failover-list {{ display: flex; flex-direction: column; gap: 8px; }}
            .failover-row {{ display: grid; grid-template-columns: minmax(130px, 180px) 1fr auto; gap: 8px; }}
            .icon-btn {{
                border: 1px solid var(--panel-border);
                border-radius: 10px;
                background: transparent;
                color: var(--text);
                padding: 8px 10px;
                cursor: pointer;
                font-weight: 700;
            }}
            .icon-btn:hover {{ border-color: var(--primary); color: var(--primary); }}
            label {{ font-weight: 600; }}
            .hint {{ font-size: 12px; color: var(--muted); }}
            input, select {{
                padding: 10px 11px;
                border-radius: 10px;
                border: 1px solid var(--panel-border);
                background: #0f1730;
                color: var(--text);
            }}
            input:focus, select:focus {{ outline: 2px solid rgba(79, 140, 255, 0.35); border-color: var(--primary); }}
            .model-pair {{ display: grid; grid-template-columns: 180px 1fr; gap: 8px; }}
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
                    <form method="post" action="/admin/logout">
                        <button class="button-secondary" type="submit">Logout</button>
                    </form>
                </div>

                <form method="post" action="/admin/settings">
                    <div class="section">
                        <h2>Routing Models</h2>
                        <p class="hint">Choose provider, pick primary model, then add failover models using the + button below each tier.</p>
                        <div class="grid">
                            <div class="field"><label>nadirclaw_simple_model</label><div class="model-pair"><select name="nadirclaw_simple_model_provider">{provider_options_html(simple_provider)}</select><select name="nadirclaw_simple_model">{model_options_html_for_provider(simple_provider, simple_name)}</select></div></div>
                            <div class="field"><label>nadirclaw_complex_model</label><div class="model-pair"><select name="nadirclaw_complex_model_provider">{provider_options_html(complex_provider)}</select><select name="nadirclaw_complex_model">{model_options_html_for_provider(complex_provider, complex_name)}</select></div></div>
                            <div class="field"><label>nadirclaw_simple_models (failover)</label><div id="simple-failover-list" class="failover-list" data-initial="{escape(simple_list_value, quote=True)}"></div><input type="hidden" name="nadirclaw_simple_models" value="" /><button type="button" class="icon-btn add-failover-btn" data-tier="simple">+</button></div>
                            <div class="field"><label>nadirclaw_complex_models (failover)</label><div id="complex-failover-list" class="failover-list" data-initial="{escape(complex_list_value, quote=True)}"></div><input type="hidden" name="nadirclaw_complex_models" value="" /><button type="button" class="icon-btn add-failover-btn" data-tier="complex">+</button></div>
                            <div class="field"><label>nadirclaw_reasoning_model</label><div class="model-pair"><select name="nadirclaw_reasoning_model_provider">{provider_options_html(reasoning_provider)}</select><select name="nadirclaw_reasoning_model">{model_options_html_for_provider(reasoning_provider, reasoning_name)}</select></div></div>
                            <div class="field"><label>nadirclaw_free_model</label><div class="model-pair"><select name="nadirclaw_free_model_provider">{provider_options_html(free_provider)}</select><select name="nadirclaw_free_model">{model_options_html_for_provider(free_provider, free_name)}</select></div></div>
                            <div class="field"><label>nadirclaw_reasoning_models (failover)</label><div id="reasoning-failover-list" class="failover-list" data-initial="{escape(reasoning_list_value, quote=True)}"></div><input type="hidden" name="nadirclaw_reasoning_models" value="" /><button type="button" class="icon-btn add-failover-btn" data-tier="reasoning">+</button></div>
                            <div class="field"><label>nadirclaw_free_models (failover)</label><div id="free-failover-list" class="failover-list" data-initial="{escape(free_list_value, quote=True)}"></div><input type="hidden" name="nadirclaw_free_models" value="" /><button type="button" class="icon-btn add-failover-btn" data-tier="free">+</button></div>
                            <div class="field"><label>nadirclaw_models</label><input name="nadirclaw_models" value="{current('NADIRCLAW_MODELS')}" placeholder="comma-separated fallback list" /></div>
                            <div class="field"><label>nadirclaw_confidence_threshold</label><input name="nadirclaw_confidence_threshold" value="{current('NADIRCLAW_CONFIDENCE_THRESHOLD')}" placeholder="0.06" /></div>
                        </div>
                    </div>

                    <div class="section">
                        <h2>Providers & Authentication</h2>
                        <div class="grid">
                            <div class="field"><label>ollama_api_base</label><input name="ollama_api_base" value="{current('OLLAMA_API_BASE') or default_ollama}" placeholder="http://localhost:11434" /></div>
                            <div class="field"><label>nadirclaw_auth_token</label><div class="field-row"><input type="password" name="nadirclaw_auth_token" placeholder="unchanged unless provided" /><button type="button" class="button-secondary clear-secret-btn" data-clear-target="clear_nadirclaw_auth_token">Clear</button></div><input type="hidden" name="clear_nadirclaw_auth_token" value="0" /></div>
                            <div class="field"><label>gemini_api_key</label><div class="field-row"><input type="password" name="gemini_api_key" placeholder="unchanged unless provided" /><button type="button" class="button-secondary clear-secret-btn" data-clear-target="clear_gemini_api_key">Clear</button></div><input type="hidden" name="clear_gemini_api_key" value="0" /></div>
                            <div class="field"><label>anthropic_api_key</label><div class="field-row"><input type="password" name="anthropic_api_key" placeholder="unchanged unless provided" /><button type="button" class="button-secondary clear-secret-btn" data-clear-target="clear_anthropic_api_key">Clear</button></div><input type="hidden" name="clear_anthropic_api_key" value="0" /></div>
                            <div class="field"><label>openai_api_key</label><div class="field-row"><input type="password" name="openai_api_key" placeholder="unchanged unless provided" /><button type="button" class="button-secondary clear-secret-btn" data-clear-target="clear_openai_api_key">Clear</button></div><input type="hidden" name="clear_openai_api_key" value="0" /></div>
                            <div class="field"><label>deepseek_api_key</label><div class="field-row"><input type="password" name="deepseek_api_key" placeholder="unchanged unless provided" /><button type="button" class="button-secondary clear-secret-btn" data-clear-target="clear_deepseek_api_key">Clear</button></div><input type="hidden" name="clear_deepseek_api_key" value="0" /></div>
                        </div>
                    </div>

                    <div class="section">
                        <h2>Server & Observability</h2>
                        <div class="grid">
                            <div class="field"><label>nadirclaw_port</label><input name="nadirclaw_port" value="{current('NADIRCLAW_PORT')}" placeholder="8856" /></div>
                            <div class="field"><label>nadirclaw_log_dir</label><input name="nadirclaw_log_dir" value="{current('NADIRCLAW_LOG_DIR')}" placeholder="~/.nadirclaw/logs" /></div>
                            <div class="field"><label>nadirclaw_log_raw</label><input name="nadirclaw_log_raw" value="{current('NADIRCLAW_LOG_RAW')}" placeholder="true or false" /></div>
                            <div class="field"><label>otel_exporter_otlp_endpoint</label><input name="otel_exporter_otlp_endpoint" value="{current('OTEL_EXPORTER_OTLP_ENDPOINT')}" placeholder="http://localhost:4317" /></div>
                        </div>
                    </div>

                    <div class="actions">
                        <label><input type="checkbox" name="fetch_models" checked /> fetch_models</label>
                        <button type="submit">Save settings</button>
                        <span class="hint">Saves to <code>~/.nadirclaw/.env</code> and applies immediately.</span>
                    </div>
                </form>

                {result_html}
            </section>
        </main>
        <script>
            (function () {{
                const modelOptionsCache = new Map();
                const failoverRowCounters = {{ simple: 0, complex: 0, reasoning: 0, free: 0 }};

                function tierMeta(tier) {{
                    const map = {{
                        simple: {{ providerSelect: 'nadirclaw_simple_model_provider', primaryModel: 'nadirclaw_simple_model', listId: 'simple-failover-list', fieldPrefix: 'nadirclaw_simple_models' }},
                        complex: {{ providerSelect: 'nadirclaw_complex_model_provider', primaryModel: 'nadirclaw_complex_model', listId: 'complex-failover-list', fieldPrefix: 'nadirclaw_complex_models' }},
                        reasoning: {{ providerSelect: 'nadirclaw_reasoning_model_provider', primaryModel: 'nadirclaw_reasoning_model', listId: 'reasoning-failover-list', fieldPrefix: 'nadirclaw_reasoning_models' }},
                        free: {{ providerSelect: 'nadirclaw_free_model_provider', primaryModel: 'nadirclaw_free_model', listId: 'free-failover-list', fieldPrefix: 'nadirclaw_free_models' }},
                    }};
                    return map[tier];
                }}

                function composeModelId(provider, modelName) {{
                    const m = (modelName || '').trim();
                    if (!m) return '';
                    if (m.includes('/')) return m;

                    const p = (provider || '').trim().toLowerCase();
                    if (!p || p === 'custom' || p === 'other' || p === 'auto') return m;
                    if (p === 'openai' || p === 'google' || p === 'anthropic') return m;
                    return `${{p}}/${{m}}`;
                }}

                function splitProviderModel(rawValue, fallbackProvider) {{
                    const raw = (rawValue || '').trim();
                    const fallback = (fallbackProvider || '').trim();
                    if (!raw) return {{ provider: fallback, model: '' }};
                    const slash = raw.indexOf('/');
                    if (slash > 0) {{
                        return {{
                            provider: raw.slice(0, slash).trim() || fallback,
                            model: raw.slice(slash + 1).trim(),
                        }};
                    }}

                    // Infer provider from common unprefixed model naming schemes.
                    // This avoids mislabeling failover rows as the primary tier provider.
                    const lower = raw.toLowerCase();
                    if (lower.startsWith('gpt') || /^o[1-9](?:-|$)/.test(lower)) {{
                        return {{ provider: 'openai', model: raw }};
                    }}
                    if (lower.startsWith('claude')) {{
                        return {{ provider: 'anthropic', model: raw }};
                    }}
                    if (lower.startsWith('gemini')) {{
                        return {{ provider: 'google', model: raw }};
                    }}
                    if (lower.startsWith('deepseek')) {{
                        return {{ provider: 'deepseek', model: raw }};
                    }}

                    return {{ provider: fallback, model: raw }};
                }}

                function getPrimaryModelId(tier) {{
                    const meta = tierMeta(tier);
                    if (!meta) return '';
                    const providerEl = document.querySelector(`select[name="${{meta.providerSelect}}"]`);
                    const modelEl = document.querySelector(`select[name="${{meta.primaryModel}}"]`);
                    return composeModelId(providerEl?.value || '', modelEl?.value || '');
                }}

                function syncFailoverHidden(tier) {{
                    const meta = tierMeta(tier);
                    if (!meta) return;
                    const hidden = document.querySelector(`input[name="${{meta.fieldPrefix}}"]`);
                    const listEl = document.getElementById(meta.listId);
                    if (!hidden || !listEl) return;

                    const rows = listEl.querySelectorAll('.failover-row');
                    const values = [];
                    rows.forEach((row) => {{
                        const providerName = row.dataset.providerName || '';
                        const modelName = row.dataset.modelName || '';
                        if (!providerName || !modelName) return;
                        const providerEl = document.querySelector(`select[name="${{providerName}}"]`);
                        const modelEl = document.querySelector(`select[name="${{modelName}}"]`);
                        const full = composeModelId(providerEl?.value || '', modelEl?.value || '');
                        if (full) values.push(full);
                    }});
                    hidden.value = values.join(',');
                }}

                async function fetchProviderModels(provider) {{
                    const p = (provider || '').trim();
                    if (!p || p === 'custom') return [];
                    if (modelOptionsCache.has(p)) return modelOptionsCache.get(p);

                    const ollamaBaseEl = document.querySelector('input[name="ollama_api_base"]');
                    const ollamaBase = (ollamaBaseEl?.value || '').trim();

                    function credentialOverrideForProvider(key) {{
                        const keyByProvider = {{
                            openai: 'openai_api_key',
                            'openai-codex': 'openai_api_key',
                            anthropic: 'anthropic_api_key',
                            google: 'gemini_api_key',
                            deepseek: 'deepseek_api_key',
                        }};
                        const fieldName = keyByProvider[key] || '';
                        if (!fieldName) return '';
                        const fieldEl = document.querySelector(`input[name="${{fieldName}}"]`);
                        return (fieldEl?.value || '').trim();
                    }}

                    try {{
                        const res = await fetch('/admin/provider-models', {{
                            method: 'POST',
                            credentials: 'same-origin',
                            headers: {{ 'Content-Type': 'application/json' }},
                            body: JSON.stringify({{
                                provider: p,
                                ollama_api_base: ollamaBase || null,
                                credential_override: credentialOverrideForProvider(p) || null,
                            }}),
                        }});
                        if (!res.ok) return [];
                        const data = await res.json();
                        const models = Array.isArray(data.models) ? data.models : [];
                        modelOptionsCache.set(p, models);
                        return models;
                    }} catch (_e) {{
                        return [];
                    }}
                }}

                async function loadProviderModels(providerSelectName, modelSelectName, preserveCurrent = true, excludedModelId = '') {{
                    const providerEl = document.querySelector(`select[name="${{providerSelectName}}"]`);
                    const modelEl = document.querySelector(`select[name="${{modelSelectName}}"]`);
                    if (!providerEl || !modelEl) return;

                    const provider = (providerEl.value || '').trim();
                    if (!provider || provider === 'custom') return;

                    try {{
                        const models = await fetchProviderModels(provider);
                        const current = preserveCurrent ? (modelEl.value || '').trim() : '';
                        const options = [...new Set([...models, ...(current ? [current] : [])])]
                            .filter(Boolean)
                            .filter((m) => composeModelId(provider, m) !== excludedModelId)
                            .sort();
                        modelEl.innerHTML = options
                            .map((m) => `<option value="${{String(m).replace(/"/g, '&quot;')}}"${{m === current ? ' selected' : ''}}>${{String(m)}}</option>`)
                            .join('');
                        if (!options.includes(current) && options.length > 0) {{
                            modelEl.value = options[0];
                        }}
                    }} catch (_e) {{
                        // Keep existing options if request fails.
                    }}
                }}

                function addFailoverRow(tier, providerName = '', modelName = '') {{
                    const meta = tierMeta(tier);
                    if (!meta) return;
                    const listEl = document.getElementById(meta.listId);
                    const primaryProviderEl = document.querySelector(`select[name="${{meta.providerSelect}}"]`);
                    if (!listEl || !primaryProviderEl) return;

                    const idx = failoverRowCounters[tier] || 0;
                    failoverRowCounters[tier] = idx + 1;

                    const row = document.createElement('div');
                    row.className = 'failover-row';

                    const rowProviderName = `${{meta.fieldPrefix}}__provider__${{idx}}`;
                    const rowModelName = `${{meta.fieldPrefix}}__item__${{idx}}`;
                    row.dataset.providerName = rowProviderName;
                    row.dataset.modelName = rowModelName;

                    const providerSelect = document.createElement('select');
                    providerSelect.name = rowProviderName;
                    providerSelect.innerHTML = primaryProviderEl ? primaryProviderEl.innerHTML : '';
                    providerSelect.value = (providerName || primaryProviderEl?.value || '').trim();

                    const select = document.createElement('select');
                    select.name = rowModelName;

                    const removeBtn = document.createElement('button');
                    removeBtn.type = 'button';
                    removeBtn.className = 'icon-btn';
                    removeBtn.textContent = '−';
                    removeBtn.addEventListener('click', () => {{
                        row.remove();
                        syncFailoverHidden(tier);
                    }});

                    providerSelect.addEventListener('change', () => {{
                        loadProviderModels(providerSelect.name, select.name, false, getPrimaryModelId(tier)).then(() => syncFailoverHidden(tier));
                    }});
                    select.addEventListener('change', () => syncFailoverHidden(tier));

                    row.appendChild(providerSelect);
                    row.appendChild(select);
                    row.appendChild(removeBtn);
                    listEl.appendChild(row);

                    loadProviderModels(providerSelect.name, select.name, false, getPrimaryModelId(tier)).then(() => {{
                        if (modelName) select.value = modelName;
                        syncFailoverHidden(tier);
                    }});
                }}

                function reloadFailoverRowsForTier(tier) {{
                    const meta = tierMeta(tier);
                    if (!meta) return;
                    const listEl = document.getElementById(meta.listId);
                    if (!listEl) return;
                    const excluded = getPrimaryModelId(tier);
                    const rows = listEl.querySelectorAll('.failover-row');
                    rows.forEach((row) => {{
                        const providerName = row.dataset.providerName || '';
                        const modelName = row.dataset.modelName || '';
                        if (!providerName || !modelName) return;
                        loadProviderModels(providerName, modelName, true, excluded).then(() => syncFailoverHidden(tier));
                    }});
                }}

                const pairs = [
                    ['nadirclaw_simple_model_provider', 'nadirclaw_simple_model'],
                    ['nadirclaw_complex_model_provider', 'nadirclaw_complex_model'],
                    ['nadirclaw_reasoning_model_provider', 'nadirclaw_reasoning_model'],
                    ['nadirclaw_free_model_provider', 'nadirclaw_free_model'],
                ];

                for (const [providerName, modelName] of pairs) {{
                    const providerEl = document.querySelector(`select[name="${{providerName}}"]`);
                    const modelEl = document.querySelector(`select[name="${{modelName}}"]`);
                    if (providerEl) {{
                        providerEl.addEventListener('change', () => {{
                            if (modelEl) {{
                                modelEl.innerHTML = '';
                                modelEl.value = '';
                            }}
                            const tier = providerName.replace('_model_provider', '').replace('nadirclaw_', '');
                            loadProviderModels(providerName, modelName, false);
                            reloadFailoverRowsForTier(tier);
                        }});
                        loadProviderModels(providerName, modelName);
                    }}
                    if (modelEl) {{
                        modelEl.addEventListener('change', () => {{
                            const tier = modelName.replace('_model', '').replace('nadirclaw_', '');
                            reloadFailoverRowsForTier(tier);
                        }});
                    }}
                }}

                const addButtons = document.querySelectorAll('.add-failover-btn');
                addButtons.forEach((btn) => {{
                    btn.addEventListener('click', () => {{
                        const tier = btn.getAttribute('data-tier');
                        addFailoverRow(tier || '');
                    }});
                }});

                for (const tier of ['simple', 'complex', 'reasoning', 'free']) {{
                    const meta = tierMeta(tier);
                    const listEl = meta ? document.getElementById(meta.listId) : null;
                    if (!meta || !listEl) continue;
                    const primaryProviderEl = document.querySelector(`select[name="${{meta.providerSelect}}"]`);
                    const primaryProvider = (primaryProviderEl?.value || '').trim();
                    const primaryModelId = getPrimaryModelId(tier);
                    const initial = (listEl.getAttribute('data-initial') || '').split(',').map((v) => v.trim()).filter(Boolean);
                    if (initial.length === 0) continue;
                    initial.forEach((raw) => {{
                        const parsed = splitProviderModel(raw, primaryProvider);
                        const composed = composeModelId(parsed.provider, parsed.model);
                        if (!parsed.model || composed === primaryModelId) return;
                        addFailoverRow(tier, parsed.provider, parsed.model);
                    }});
                    syncFailoverHidden(tier);
                }}

                const settingsForm = document.querySelector('form[action="/admin/settings"]');
                if (settingsForm) {{
                    settingsForm.addEventListener('submit', () => {{
                        for (const tier of ['simple', 'complex', 'reasoning', 'free']) {{
                            syncFailoverHidden(tier);
                        }}
                    }});
                }}

                const ollamaBaseEl = document.querySelector('input[name="ollama_api_base"]');
                if (ollamaBaseEl) {{
                    ollamaBaseEl.addEventListener('change', () => {{
                        modelOptionsCache.clear();
                        for (const [providerName, modelName] of pairs) {{
                            loadProviderModels(providerName, modelName);
                        }}
                        for (const tier of ['simple', 'complex', 'reasoning', 'free']) {{
                            reloadFailoverRowsForTier(tier);
                        }}
                    }});
                }}

                const clearButtons = document.querySelectorAll('.clear-secret-btn');
                clearButtons.forEach((btn) => {{
                    btn.addEventListener('click', () => {{
                        const target = btn.getAttribute('data-clear-target');
                        if (!target) return;
                        const hidden = document.querySelector(`input[name="${{target}}"]`);
                        const row = btn.closest('.field-row');
                        const pwdInput = row ? row.querySelector('input[type="password"]') : null;
                        if (hidden) hidden.value = '1';
                        if (pwdInput) pwdInput.value = '';
                        btn.textContent = 'Will clear';
                    }});

                    const row = btn.closest('.field-row');
                    const pwdInput = row ? row.querySelector('input[type="password"]') : null;
                    if (pwdInput) {{
                        pwdInput.addEventListener('input', () => {{
                            const target = btn.getAttribute('data-clear-target');
                            const hidden = target ? document.querySelector(`input[name="${{target}}"]`) : null;
                            if (hidden && pwdInput.value.trim() !== '') {{
                                hidden.value = '0';
                                btn.textContent = 'Clear';
                            }}
                        }});
                    }}
                }});
            }})();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
