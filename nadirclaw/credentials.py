"""Credential storage and resolution for NadirClaw.

Stores provider API keys/tokens in ~/.nadirclaw/credentials.json.
Resolution chain: OpenClaw stored token → NadirClaw stored token → env var.
Supports OAuth tokens with automatic refresh for openai-codex.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("nadirclaw")

# Provider name → env var mapping
_ENV_VAR_MAP = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openai-codex": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "cohere": "COHERE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}

# Alternative env vars checked as fallback (order matters)
_ENV_VAR_FALLBACKS = {
    "google": ["GEMINI_API_KEY"],
}

# Model prefix/pattern → provider mapping
# NOTE: order matters — more specific prefixes must come before shorter ones
_MODEL_PROVIDER_PATTERNS = {
    "anthropic/": "anthropic",
    "claude-": "anthropic",
    "openai-codex/": "openai-codex",
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


def save_oauth_credential(
    provider: str,
    access_token: str,
    refresh_token: str,
    expires_in: int,
) -> None:
    """Save an OAuth credential with refresh token and expiry.

    Args:
        provider: Provider name (e.g. "openai-codex").
        access_token: The OAuth access token.
        refresh_token: The OAuth refresh token for renewal.
        expires_in: Seconds until the access token expires.
    """
    creds = _read_credentials()
    creds[provider] = {
        "token": access_token,
        "refresh_token": refresh_token,
        "expires_at": int(time.time()) + expires_in,
        "source": "oauth",
    }
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
    """Check OpenClaw config for a stored token.

    Checks two locations:
      1. ~/.openclaw/agents/main/agent/auth-profiles.json  (OAuth tokens — access, refresh, expires)
      2. ~/.openclaw/openclaw.json  (legacy key storage)
    """
    # --- 1. auth-profiles.json (where OpenClaw actually stores OAuth tokens) ---
    auth_profiles_path = (
        Path.home() / ".openclaw" / "agents" / "main" / "agent" / "auth-profiles.json"
    )
    if auth_profiles_path.exists():
        try:
            data = json.loads(auth_profiles_path.read_text())
            profiles = data.get("profiles", {})
            for profile in profiles.values():
                if profile.get("provider") == provider:
                    # OAuth profiles use "access" key
                    token = profile.get("access") or profile.get("token") or profile.get("key")
                    if token:
                        return token
        except (json.JSONDecodeError, OSError, KeyError):
            pass

    # --- 2. openclaw.json (legacy / API key style) ---
    openclaw_path = Path.home() / ".openclaw" / "openclaw.json"
    if not openclaw_path.exists():
        return None
    try:
        config = json.loads(openclaw_path.read_text())
        # Check auth profiles in the main config
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


def _maybe_refresh_oauth(provider: str, entry: dict) -> Optional[str]:
    """If the stored credential is an OAuth token that's expired, refresh it.

    Returns the (possibly refreshed) access token, or None on failure.
    """
    if entry.get("source") != "oauth":
        return entry.get("token")

    expires_at = entry.get("expires_at", 0)
    refresh_token = entry.get("refresh_token")

    # Refresh if within 60 seconds of expiry
    if time.time() < (expires_at - 60):
        return entry.get("token")

    if not refresh_token:
        logger.warning("OAuth token expired for %s but no refresh token available", provider)
        return entry.get("token")  # return stale token; the API will reject it

    logger.info("Refreshing expired OAuth token for %s...", provider)
    try:
        from nadirclaw.oauth import refresh_access_token

        token_data = refresh_access_token(refresh_token)
        new_access = token_data["access_token"]
        new_refresh = token_data.get("refresh_token", refresh_token)
        new_expires = token_data.get("expires_in", 3600)

        save_oauth_credential(provider, new_access, new_refresh, new_expires)
        logger.info("OAuth token refreshed for %s (expires in %ds)", provider, new_expires)
        return new_access
    except Exception as e:
        logger.error("Failed to refresh OAuth token for %s: %s", provider, e)
        return entry.get("token")  # return stale token as last resort


def _check_openclaw_with_refresh(provider: str) -> Optional[str]:
    """Check OpenClaw auth-profiles for a token, refreshing if expired.

    OpenClaw stores OAuth tokens with 'access', 'refresh', 'expires' (ms) fields.
    This function reads them and auto-refreshes expired tokens.
    """
    auth_profiles_path = (
        Path.home() / ".openclaw" / "agents" / "main" / "agent" / "auth-profiles.json"
    )
    if not auth_profiles_path.exists():
        return None

    try:
        data = json.loads(auth_profiles_path.read_text())
        profiles = data.get("profiles", {})
        for profile in profiles.values():
            if profile.get("provider") != provider:
                continue
            if profile.get("type") != "oauth":
                # API key profile — return the key directly
                return profile.get("key") or profile.get("access")

            access_token = profile.get("access")
            refresh_tok = profile.get("refresh")
            # OpenClaw stores expires in milliseconds
            expires_ms = profile.get("expires", 0)
            expires_at = expires_ms / 1000  # convert to seconds

            if not access_token:
                continue

            # Check if token is still valid (with 60s buffer)
            if time.time() < (expires_at - 60):
                return access_token

            # Token expired — try to refresh
            if not refresh_tok:
                logger.warning("OpenClaw token expired for %s, no refresh token", provider)
                return access_token

            logger.info("Refreshing expired OpenClaw token for %s...", provider)
            try:
                from nadirclaw.oauth import refresh_access_token
                token_data = refresh_access_token(refresh_tok)
                new_access = token_data["access_token"]
                # Also save to NadirClaw's own credential store
                new_refresh = token_data.get("refresh_token", refresh_tok)
                new_expires_in = token_data.get("expires_in", 3600)
                save_oauth_credential(provider, new_access, new_refresh, new_expires_in)
                logger.info("Token refreshed for %s (expires in %ds)", provider, new_expires_in)
                return new_access
            except Exception as e:
                logger.error("Failed to refresh OpenClaw token for %s: %s", provider, e)
                return access_token  # return stale token as last resort

    except (json.JSONDecodeError, OSError, KeyError) as e:
        logger.debug("Could not read OpenClaw auth-profiles: %s", e)

    return None


def get_credential(provider: str) -> Optional[str]:
    """Resolve a credential for a provider.

    Resolution order:
      1. OpenClaw stored token (~/.openclaw/openclaw.json)
      2. NadirClaw stored token (~/.nadirclaw/credentials.json)
         — with automatic OAuth refresh if expired
      3. Environment variable
      4. None

    Args:
        provider: Provider name (e.g. "anthropic", "openai").

    Returns:
        The token string, or None if no credential found.
    """
    # 1. OpenClaw auth-profiles (with auto-refresh for OAuth tokens)
    token = _check_openclaw_with_refresh(provider)
    if token:
        return token

    # 1b. OpenClaw legacy (openclaw.json)
    token = _check_openclaw(provider)
    if token:
        return token

    # 2. NadirClaw stored credentials (with OAuth auto-refresh)
    creds = _read_credentials()
    entry = creds.get(provider)
    if entry and entry.get("token"):
        return _maybe_refresh_oauth(provider, entry)

    # 3. Environment variable (primary)
    env_var = _ENV_VAR_MAP.get(provider)
    if env_var:
        val = os.getenv(env_var, "")
        if val:
            return val

    # 4. Fallback env vars (e.g. GEMINI_API_KEY for google)
    for fallback_var in _ENV_VAR_FALLBACKS.get(provider, []):
        val = os.getenv(fallback_var, "")
        if val:
            return val

    return None


def get_credential_source(provider: str) -> Optional[str]:
    """Return the source label for how a credential was resolved.

    Returns one of: "openclaw", "oauth", "setup-token", "manual", "env", or None.
    """
    # 1. OpenClaw (auth-profiles with OAuth + legacy)
    if _check_openclaw_with_refresh(provider) or _check_openclaw(provider):
        return "openclaw"

    # 2. NadirClaw stored
    creds = _read_credentials()
    entry = creds.get(provider)
    if entry and entry.get("token"):
        return entry.get("source", "stored")

    # 3. Env var (primary)
    env_var = _ENV_VAR_MAP.get(provider)
    if env_var and os.getenv(env_var, ""):
        return "env"

    # 4. Fallback env vars
    for fallback_var in _ENV_VAR_FALLBACKS.get(provider, []):
        if os.getenv(fallback_var, ""):
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
