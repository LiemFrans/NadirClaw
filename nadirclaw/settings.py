"""Minimal env-based configuration for NadirClaw."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from ~/.nadirclaw/.env if it exists
_nadirclaw_dir = Path.home() / ".nadirclaw"
_env_file = _nadirclaw_dir / ".env"
if _env_file.exists():
    load_dotenv(_env_file)
else:
    # Fallback to current directory .env
    load_dotenv()


class Settings:
    """All configuration from environment variables."""

    @staticmethod
    def _split_models(raw: str) -> list[str]:
        return [m.strip() for m in (raw or "").split(",") if m.strip()]

    @staticmethod
    def _dedupe_keep_order(models: list[str]) -> list[str]:
        seen = set()
        out: list[str] = []
        for m in models:
            if m in seen:
                continue
            seen.add(m)
            out.append(m)
        return out

    def _tier_models(self, single_env: str, multi_env: str, fallback: list[str]) -> list[str]:
        raw_multi = os.getenv(multi_env, "").strip()
        if raw_multi:
            return self._dedupe_keep_order(self._split_models(raw_multi))

        raw_single = os.getenv(single_env, "").strip()
        if raw_single:
            return self._dedupe_keep_order(self._split_models(raw_single))

        return self._dedupe_keep_order(fallback)

    @property
    def AUTH_TOKEN(self) -> str:
        return os.getenv("NADIRCLAW_AUTH_TOKEN", "")

    @property
    def SIMPLE_MODEL(self) -> str:
        """Model for simple prompts. Falls back to last model in MODELS list."""
        models = self.SIMPLE_MODELS
        return models[0] if models else "gemini-3-flash-preview"

    @property
    def SIMPLE_MODELS(self) -> list[str]:
        """Ordered failover models for simple tier."""
        fallback = [self.MODELS[-1]] if self.MODELS else ["gemini-3-flash-preview"]
        return self._tier_models("NADIRCLAW_SIMPLE_MODEL", "NADIRCLAW_SIMPLE_MODELS", fallback)

    @property
    def COMPLEX_MODEL(self) -> str:
        """Model for complex prompts. Falls back to first model in MODELS list."""
        models = self.COMPLEX_MODELS
        return models[0] if models else "openai-codex/gpt-5.3-codex"

    @property
    def COMPLEX_MODELS(self) -> list[str]:
        """Ordered failover models for complex tier."""
        fallback = [self.MODELS[0]] if self.MODELS else ["openai-codex/gpt-5.3-codex"]
        return self._tier_models("NADIRCLAW_COMPLEX_MODEL", "NADIRCLAW_COMPLEX_MODELS", fallback)

    @property
    def MODELS(self) -> list[str]:
        raw = os.getenv(
            "NADIRCLAW_MODELS",
            "openai-codex/gpt-5.3-codex,gemini-3-flash-preview",
        )
        return [m.strip() for m in raw.split(",") if m.strip()]

    @property
    def ANTHROPIC_API_KEY(self) -> str:
        return os.getenv("ANTHROPIC_API_KEY", "")

    @property
    def OPENAI_API_KEY(self) -> str:
        return os.getenv("OPENAI_API_KEY", "")

    @property
    def GEMINI_API_KEY(self) -> str:
        return os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")

    @property
    def OLLAMA_API_BASE(self) -> str:
        return os.getenv("OLLAMA_API_BASE", "http://localhost:11434")

    @property
    def CONFIDENCE_THRESHOLD(self) -> float:
        return float(os.getenv("NADIRCLAW_CONFIDENCE_THRESHOLD", "0.06"))

    @property
    def PORT(self) -> int:
        return int(os.getenv("NADIRCLAW_PORT", "8856"))

    @property
    def LOG_RAW(self) -> bool:
        """When True, log full raw request messages and response content."""
        return os.getenv("NADIRCLAW_LOG_RAW", "").lower() in ("1", "true", "yes")

    @property
    def LOG_DIR(self) -> Path:
        return Path(os.getenv("NADIRCLAW_LOG_DIR", "~/.nadirclaw/logs")).expanduser()

    @property
    def CREDENTIALS_FILE(self) -> Path:
        return Path.home() / ".nadirclaw" / "credentials.json"

    @property
    def REASONING_MODEL(self) -> str:
        """Model for reasoning tasks. Falls back to COMPLEX_MODEL."""
        models = self.REASONING_MODELS
        return models[0] if models else self.COMPLEX_MODEL

    @property
    def REASONING_MODELS(self) -> list[str]:
        """Ordered failover models for reasoning tier."""
        return self._tier_models(
            "NADIRCLAW_REASONING_MODEL",
            "NADIRCLAW_REASONING_MODELS",
            self.COMPLEX_MODELS,
        )

    @property
    def FREE_MODEL(self) -> str:
        """Free fallback model. Falls back to SIMPLE_MODEL."""
        models = self.FREE_MODELS
        return models[0] if models else self.SIMPLE_MODEL

    @property
    def FREE_MODELS(self) -> list[str]:
        """Ordered failover models for free tier."""
        return self._tier_models(
            "NADIRCLAW_FREE_MODEL",
            "NADIRCLAW_FREE_MODELS",
            self.SIMPLE_MODELS,
        )

    @property
    def has_explicit_tiers(self) -> bool:
        """True if SIMPLE_MODEL and COMPLEX_MODEL are explicitly set via env."""
        return bool(
            (os.getenv("NADIRCLAW_SIMPLE_MODEL") or os.getenv("NADIRCLAW_SIMPLE_MODELS"))
            and (os.getenv("NADIRCLAW_COMPLEX_MODEL") or os.getenv("NADIRCLAW_COMPLEX_MODELS"))
        )

    @property
    def tier_models(self) -> list[str]:
        """Deduplicated list of all tier models in failover order."""
        merged = [
            *self.COMPLEX_MODELS,
            *self.SIMPLE_MODELS,
            *self.REASONING_MODELS,
            *self.FREE_MODELS,
        ]
        return self._dedupe_keep_order(merged)


settings = Settings()
