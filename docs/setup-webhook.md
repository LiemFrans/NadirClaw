# Setup Webhook API

The setup webhook lets you remotely update supported NadirClaw environment variables, persist them to `~/.nadirclaw/.env`, and optionally fetch Ollama models.

- **Endpoint**: `POST /v1/setup/webhook`
- **Auth**: same as other `/v1/*` endpoints (`NADIRCLAW_AUTH_TOKEN` if enabled)

## Admin Web UI

You can manage the same settings through a web interface:

- **URL**: `GET /admin`
- **Login**: `POST /admin/login`
- **Save form**: `POST /admin/settings`
- **Logout**: `POST /admin/logout`

Admin password source:

1. `NADIRCLAW_ADMIN_PASSWORD` (preferred)
2. `NADIRCLAW_AUTH_TOKEN` (fallback)

If neither is set, admin login is disabled.

## Request Body

```json
{
  "ollama_api_base": "10.4.136.145:11434",
  "env": {
    "nadirclaw_simple_model": "gemini-2.5-flash",
    "nadirclaw_complex_model": "gpt-4.1",
    "nadirclaw_reasoning_model": "o3",
    "nadirclaw_free_model": "ollama/llama3.1:8b",
    "nadirclaw_auth_token": "local",
    "gemini_api_key": "AIza...",
    "anthropic_api_key": "sk-ant-...",
    "openai_api_key": "sk-...",
    "ollama_api_base": "http://localhost:11434",
    "nadirclaw_confidence_threshold": "0.06",
    "nadirclaw_port": "8856",
    "nadirclaw_log_dir": "~/.nadirclaw/logs",
    "nadirclaw_log_raw": "false",
    "nadirclaw_models": "openai-codex/gpt-5.3-codex,gemini-3-flash-preview",
    "otel_exporter_otlp_endpoint": "http://localhost:4317"
  },
  "fetch_models": true
}
```

### Fields

- `ollama_api_base` *(optional, string)*
  - Ollama base URL or host:port.
  - Normalization rules:
    - Empty value falls back to current configured base (or default `http://localhost:11434`).
    - If scheme is missing, `http://` is added.
    - Trailing `/` is removed.
- `env` *(optional, object)*
  - Key-value updates for supported environment variables.
  - Only the variables listed below are applied; unknown keys are ignored.
- `fetch_models` *(optional, boolean, default: `true`)*
  - If `true`, NadirClaw calls `${OLLAMA_API_BASE}/api/tags` and returns discovered models.

## Supported `env` Variables

- `nadirclaw_simple_model` → `NADIRCLAW_SIMPLE_MODEL`
- `nadirclaw_complex_model` → `NADIRCLAW_COMPLEX_MODEL`
- `nadirclaw_reasoning_model` → `NADIRCLAW_REASONING_MODEL`
- `nadirclaw_free_model` → `NADIRCLAW_FREE_MODEL`
- `nadirclaw_auth_token` → `NADIRCLAW_AUTH_TOKEN`
- `gemini_api_key` → `GEMINI_API_KEY`
- `anthropic_api_key` → `ANTHROPIC_API_KEY`
- `openai_api_key` → `OPENAI_API_KEY`
- `ollama_api_base` → `OLLAMA_API_BASE`
- `nadirclaw_confidence_threshold` → `NADIRCLAW_CONFIDENCE_THRESHOLD`
- `nadirclaw_port` → `NADIRCLAW_PORT`
- `nadirclaw_log_dir` → `NADIRCLAW_LOG_DIR`
- `nadirclaw_log_raw` → `NADIRCLAW_LOG_RAW`
- `nadirclaw_models` → `NADIRCLAW_MODELS`
- `otel_exporter_otlp_endpoint` → `OTEL_EXPORTER_OTLP_ENDPOINT`

## Response

```json
{
  "status": "ok",
  "ollama_api_base": "http://10.4.136.145:11434",
  "env_file": "/home/user/.nadirclaw/.env",
  "updated": {
    "NADIRCLAW_SIMPLE_MODEL": "gemini-2.5-flash",
    "OLLAMA_API_BASE": "http://10.4.136.145:11434"
  },
  "ignored": [],
  "models": ["ollama/llama3.2:3b", "ollama/qwen3:32b"],
  "model_count": 2
}
```

### Response Fields

- `status`: `"ok"` on success.
- `ollama_api_base`: normalized base URL applied by the server.
- `env_file`: path to the written `.env` file.
- `updated`: applied environment variables (normalized where relevant).
- `ignored`: unsupported keys from `env` that were ignored.
- `models`: discovered Ollama models (empty if fetch disabled or none found).
- `model_count`: `models` length.

## Behavior

On success, the endpoint:

1. Applies supported `env` updates to process environment.
2. Normalizes `OLLAMA_API_BASE` (from `ollama_api_base` and/or `env`).
3. Upserts updated keys into `~/.nadirclaw/.env`.
4. Optionally fetches model tags from `${OLLAMA_API_BASE}/api/tags`.

## curl Examples

### Update base + fetch models

```bash
curl -X POST http://localhost:8856/v1/setup/webhook \
  -H "Content-Type: application/json" \
  -d '{"ollama_api_base":"10.4.136.145:11434","fetch_models":true}'
```

### Update base only (skip fetch)

```bash
curl -X POST http://localhost:8856/v1/setup/webhook \
  -H "Content-Type: application/json" \
  -d '{"ollama_api_base":"http://localhost:11434","fetch_models":false}'
```

### Update multiple setup variables

```bash
curl -X POST http://localhost:8856/v1/setup/webhook \
  -H "Content-Type: application/json" \
  -d '{
    "env": {
      "nadirclaw_simple_model": "gemini-2.5-flash",
      "nadirclaw_complex_model": "gpt-4.1",
      "nadirclaw_reasoning_model": "o3",
      "nadirclaw_free_model": "ollama/llama3.1:8b",
      "nadirclaw_auth_token": "local",
      "nadirclaw_confidence_threshold": "0.06",
      "nadirclaw_port": "8856",
      "nadirclaw_log_dir": "~/.nadirclaw/logs",
      "nadirclaw_log_raw": "true",
      "nadirclaw_models": "openai-codex/gpt-5.3-codex,gemini-3-flash-preview",
      "otel_exporter_otlp_endpoint": "http://localhost:4317"
    },
    "fetch_models": false
  }'
```

### With auth token enabled

```bash
curl -X POST http://localhost:8856/v1/setup/webhook \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"ollama_api_base":"10.4.136.145:11434"}'
```
