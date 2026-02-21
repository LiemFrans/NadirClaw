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
    def _parse_model_list(raw: str) -> list[str]:
        """Parse comma-separated model list, trimming whitespace."""
        return [m.strip() for m in raw.split(",") if m.strip()]

    @property
    def AUTH_TOKEN(self) -> str:
        return os.getenv("NADIRCLAW_AUTH_TOKEN", "")

    @property
    def SIMPLE_MODEL(self) -> str:
        """Model for simple prompts. Falls back to last model in MODELS list."""
        return self.SIMPLE_MODELS[0]

    @property
    def SIMPLE_MODELS(self) -> list[str]:
        """Priority-ordered simple models (comma-separated env supported)."""
        explicit = os.getenv("NADIRCLAW_SIMPLE_MODEL", "")
        parsed = self._parse_model_list(explicit)
        if parsed:
            return parsed
        models = self.MODELS
        return [models[-1]] if models else ["gemini-3-flash-preview"]

    @property
    def COMPLEX_MODEL(self) -> str:
        """Model for complex prompts. Falls back to first model in MODELS list."""
        return self.COMPLEX_MODELS[0]

    @property
    def COMPLEX_MODELS(self) -> list[str]:
        """Priority-ordered complex models (comma-separated env supported)."""
        explicit = os.getenv("NADIRCLAW_COMPLEX_MODEL", "")
        parsed = self._parse_model_list(explicit)
        if parsed:
            return parsed
        models = self.MODELS
        return [models[0]] if models else ["openai-codex/gpt-5.3-codex"]

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
        return self.REASONING_MODELS[0]

    @property
    def REASONING_MODELS(self) -> list[str]:
        """Priority-ordered reasoning models (comma-separated env supported)."""
        explicit = os.getenv("NADIRCLAW_REASONING_MODEL", "")
        parsed = self._parse_model_list(explicit)
        if parsed:
            return parsed
        return list(self.COMPLEX_MODELS)

    @property
    def FREE_MODEL(self) -> str:
        """Free fallback model. Falls back to SIMPLE_MODEL."""
        return self.FREE_MODELS[0]

    @property
    def FREE_MODELS(self) -> list[str]:
        """Priority-ordered free models (comma-separated env supported)."""
        explicit = os.getenv("NADIRCLAW_FREE_MODEL", "")
        parsed = self._parse_model_list(explicit)
        if parsed:
            return parsed
        return list(self.SIMPLE_MODELS)

    @property
    def has_explicit_tiers(self) -> bool:
        """True if SIMPLE_MODEL and COMPLEX_MODEL are explicitly set via env."""
        return bool(
            os.getenv("NADIRCLAW_SIMPLE_MODEL") and os.getenv("NADIRCLAW_COMPLEX_MODEL")
        )

    @property
    def tier_models(self) -> list[str]:
        """Deduplicated list of configured tier models (all priority pools)."""
        ordered = [
            *self.COMPLEX_MODELS,
            *self.SIMPLE_MODELS,
            *self.REASONING_MODELS,
            *self.FREE_MODELS,
        ]
        seen = set()
        unique = []
        for model in ordered:
            if model not in seen:
                seen.add(model)
                unique.append(model)
        return unique


settings = Settings()
