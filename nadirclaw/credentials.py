"""Credential storage and resolution for NadirClaw.

Stores provider API keys/tokens in ~/.nadirclaw/credentials.json.
Resolution chain: OpenClaw stored token → NadirClaw stored token → env var.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("nadirclaw")

# Provider name → env var mapping
_ENV_VAR_MAP = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "cohere": "COHERE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}

# Model prefix/pattern → provider mapping
_MODEL_PROVIDER_PATTERNS = {
    "anthropic/": "anthropic",
    "claude-": "anthropic",
    "openai/": "openai",
    "gpt-": "openai",
    "o1-": "openai",
    "o3-": "openai",
    "o4-": "openai",
    "gemini/": "google",
    "gemini-": "google",
    "ollama/": "ollama",
    "cohere/": "cohere",
    "mistral/": "mistral",
    "together_ai/": "together_ai",
    "replicate/": "replicate",
}


def _credentials_path() -> Path:
    return Path.home() / ".nadirclaw" / "credentials.json"


def _read_credentials() -> dict:
    path = _credentials_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not read credentials file: %s", e)
        return {}


def _write_credentials(data: dict) -> None:
    path = _credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    # Restrict permissions to owner only
    path.chmod(0o600)


def save_credential(provider: str, token: str, source: str = "manual") -> None:
    """Save a credential for a provider.

    Args:
        provider: Provider name (e.g. "anthropic", "openai").
        token: The API key or token.
        source: How it was added ("setup-token", "manual", etc.).
    """
    creds = _read_credentials()
    creds[provider] = {"token": token, "source": source}
    _write_credentials(creds)


def remove_credential(provider: str) -> bool:
    """Remove a stored credential. Returns True if it existed."""
    creds = _read_credentials()
    if provider in creds:
        del creds[provider]
        _write_credentials(creds)
        return True
    return False


def _check_openclaw(provider: str) -> Optional[str]:
    """Check OpenClaw config for a stored token."""
    openclaw_path = Path.home() / ".openclaw" / "openclaw.json"
    if not openclaw_path.exists():
        return None
    try:
        config = json.loads(openclaw_path.read_text())
        # Check auth profiles for provider tokens
        auth = config.get("auth", {})
        profiles = auth.get("profiles", {})
        for profile in profiles.values():
            if profile.get("provider") == provider and profile.get("token"):
                return profile["token"]
        # Check provider-specific keys
        keys = auth.get("keys", {})
        env_name = _ENV_VAR_MAP.get(provider, "")
        if env_name and keys.get(env_name):
            return keys[env_name]
    except (json.JSONDecodeError, OSError, KeyError):
        pass
    return None


def get_credential(provider: str) -> Optional[str]:
    """Resolve a credential for a provider.

    Resolution order:
      1. OpenClaw stored token (~/.openclaw/openclaw.json)
      2. NadirClaw stored token (~/.nadirclaw/credentials.json)
      3. Environment variable
      4. None

    Args:
        provider: Provider name (e.g. "anthropic", "openai").

    Returns:
        The token string, or None if no credential found.
    """
    # 1. OpenClaw
    token = _check_openclaw(provider)
    if token:
        return token

    # 2. NadirClaw stored credentials
    creds = _read_credentials()
    entry = creds.get(provider)
    if entry and entry.get("token"):
        return entry["token"]

    # 3. Environment variable
    env_var = _ENV_VAR_MAP.get(provider)
    if env_var:
        val = os.getenv(env_var, "")
        if val:
            return val

    return None


def get_credential_source(provider: str) -> Optional[str]:
    """Return the source label for how a credential was resolved.

    Returns one of: "openclaw", "setup-token", "manual", "env", or None.
    """
    # 1. OpenClaw
    if _check_openclaw(provider):
        return "openclaw"

    # 2. NadirClaw stored
    creds = _read_credentials()
    entry = creds.get(provider)
    if entry and entry.get("token"):
        return entry.get("source", "stored")

    # 3. Env var
    env_var = _ENV_VAR_MAP.get(provider)
    if env_var and os.getenv(env_var, ""):
        return "env"

    return None


def detect_provider(model: str) -> Optional[str]:
    """Detect provider from a model name.

    Args:
        model: Model name like "claude-sonnet-4-20250514" or "openai/gpt-4o".

    Returns:
        Provider name (e.g. "anthropic") or None if unknown.
    """
    for pattern, provider in _MODEL_PROVIDER_PATTERNS.items():
        if model.startswith(pattern):
            return provider
    return None


def list_credentials() -> list[dict]:
    """List all configured providers with masked tokens and sources.

    Checks all resolution sources for known providers.

    Returns:
        List of dicts with provider, source, and masked_token keys.
    """
    results = []
    # Check all known providers
    providers = set(_ENV_VAR_MAP.keys())
    # Also include any providers in the credentials file
    creds = _read_credentials()
    providers.update(creds.keys())

    for provider in sorted(providers):
        source = get_credential_source(provider)
        if source:
            token = get_credential(provider)
            masked = _mask_token(token) if token else "???"
            results.append({
                "provider": provider,
                "source": source,
                "masked_token": masked,
            })

    return results


def _mask_token(token: str) -> str:
    """Mask a token for display, showing first 8 and last 4 chars."""
    if len(token) <= 12:
        return token[:4] + "***"
    return token[:8] + "..." + token[-4:]
