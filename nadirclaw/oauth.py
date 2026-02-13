"""OpenAI Codex OAuth helpers for NadirClaw.

Supports two login strategies:
  1. Delegate to the official Codex CLI (`codex login`) — preferred.
  2. Read existing credentials from OpenClaw's auth-profiles.json.

Token refresh uses the OpenAI public OAuth client_id (PKCE).
"""

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional
import urllib.parse
import urllib.request

logger = logging.getLogger("nadirclaw")

# ---------------------------------------------------------------------------
# OpenAI OAuth configuration (matches official Codex CLI / OpenClaw)
# ---------------------------------------------------------------------------

# Public OAuth client_id used by the Codex CLI (from JWT `client_id` claim).
# This is a public PKCE client — no client_secret.
_OPENAI_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_OPENAI_TOKEN_URL = "https://auth.openai.com/oauth/token"


# ---------------------------------------------------------------------------
# Read credentials stored by other tools
# ---------------------------------------------------------------------------

_OPENCLAW_AUTH_PROFILES = (
    Path.home() / ".openclaw" / "agents" / "main" / "agent" / "auth-profiles.json"
)


def read_openclaw_codex_credentials() -> Optional[dict]:
    """Read openai-codex OAuth credentials from OpenClaw's auth-profiles.json.

    Returns dict with keys: access, refresh, expires, accountId — or None.
    """
    if not _OPENCLAW_AUTH_PROFILES.exists():
        return None
    try:
        data = json.loads(_OPENCLAW_AUTH_PROFILES.read_text())
        profiles = data.get("profiles", {})
        # Look for any openai-codex profile
        for key, profile in profiles.items():
            if profile.get("provider") == "openai-codex" and profile.get("access"):
                return profile
        return None
    except (json.JSONDecodeError, OSError, KeyError) as e:
        logger.debug("Could not read OpenClaw auth-profiles: %s", e)
        return None


def read_codex_cli_credentials() -> Optional[dict]:
    """Read credentials stored by the official Codex CLI (~/.codex/).

    Returns dict with keys: access_token, refresh_token, expires_at — or None.
    """
    # Codex CLI stores auth in ~/.codex/ after login
    codex_dir = Path.home() / ".codex"
    # Try known credential file locations
    for name in ("auth.json", "credentials.json", ".auth"):
        cred_file = codex_dir / name
        if cred_file.exists():
            try:
                data = json.loads(cred_file.read_text())
                if data.get("access_token"):
                    return data
            except (json.JSONDecodeError, OSError):
                continue
    return None


# ---------------------------------------------------------------------------
# Token refresh (works with the public PKCE client_id)
# ---------------------------------------------------------------------------

def refresh_access_token(refresh_token: str) -> dict:
    """Use a refresh token to obtain a new access token from OpenAI.

    Uses the same public client_id as the Codex CLI.

    Returns dict with: access_token, refresh_token, expires_in, token_type.
    Raises RuntimeError on failure.
    """
    data = urllib.parse.urlencode({
        "grant_type": "refresh_token",
        "client_id": _OPENAI_CLIENT_ID,
        "refresh_token": refresh_token,
    }).encode("utf-8")

    req = urllib.request.Request(
        _OPENAI_TOKEN_URL,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Token refresh failed ({e.code}): {body}") from e


# ---------------------------------------------------------------------------
# Login via Codex CLI delegation
# ---------------------------------------------------------------------------

def _find_codex_cli() -> Optional[str]:
    """Find the `codex` CLI binary."""
    return shutil.which("codex")


def _find_openclaw_cli() -> Optional[str]:
    """Find the `openclaw` CLI binary."""
    return shutil.which("openclaw")


def login_via_codex_cli(timeout: int = 300) -> bool:
    """Run `codex login` interactively and wait for completion.

    Returns True if the login process exited successfully.
    """
    codex = _find_codex_cli()
    if not codex:
        raise RuntimeError(
            "Codex CLI not found. Install it first:\n"
            "  brew install openai-codex  (macOS)\n"
            "  npm i -g @openai/codex     (npm)\n\n"
            "Or use an API key instead:\n"
            "  nadirclaw auth add -p openai"
        )

    logger.info("Delegating login to Codex CLI: %s", codex)
    try:
        result = subprocess.run(
            [codex, "login"],
            timeout=timeout,
            check=False,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Codex login timed out after {timeout}s")


def login_via_openclaw(timeout: int = 300) -> bool:
    """Run `openclaw models auth login --provider openai-codex` interactively.

    Returns True if the login process exited successfully.
    """
    openclaw = _find_openclaw_cli()
    if not openclaw:
        raise RuntimeError("OpenClaw CLI not found.")

    logger.info("Delegating login to OpenClaw CLI: %s", openclaw)
    try:
        result = subprocess.run(
            [openclaw, "models", "auth", "login", "--provider", "openai-codex"],
            timeout=timeout,
            check=False,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"OpenClaw login timed out after {timeout}s")


def login_openai(timeout: int = 300) -> Optional[dict]:
    """Run the OpenAI Codex login flow.

    Strategy:
      1. Try `codex login` (official Codex CLI) — most reliable.
      2. Fall back to `openclaw models auth login --provider openai-codex`.
      3. After external login, read the stored credentials.

    Returns dict with: access_token, refresh_token, expires_at — or None.
    """
    # Try Codex CLI first
    codex = _find_codex_cli()
    openclaw = _find_openclaw_cli()

    if codex:
        success = login_via_codex_cli(timeout=timeout)
    elif openclaw:
        success = login_via_openclaw(timeout=timeout)
    else:
        raise RuntimeError(
            "Neither Codex CLI nor OpenClaw CLI found.\n\n"
            "Install the Codex CLI to enable OAuth login:\n"
            "  brew install openai-codex  (macOS)\n"
            "  npm i -g @openai/codex     (npm)\n\n"
            "Or use an API key instead:\n"
            "  nadirclaw auth add -p openai"
        )

    if not success:
        return None

    # Read the credentials that were just stored
    # Wait a moment for file writes to complete
    time.sleep(0.5)

    # Try reading from OpenClaw auth-profiles (most likely location)
    profile = read_openclaw_codex_credentials()
    if profile:
        return {
            "access_token": profile["access"],
            "refresh_token": profile.get("refresh", ""),
            # OpenClaw stores expires in milliseconds
            "expires_at": int(profile.get("expires", 0)) // 1000,
        }

    # Try reading from Codex CLI storage
    codex_creds = read_codex_cli_credentials()
    if codex_creds:
        return codex_creds

    logger.warning(
        "Login appeared to succeed but could not find stored credentials. "
        "The token may be stored in a format NadirClaw doesn't recognize yet."
    )
    return None
