"""Configuration loading for wiro-gateway.

Reads from process env first, then from a `.env` next to the launched binary
or in the current working directory. Values are validated with pydantic so
misconfiguration fails fast at startup rather than at first request.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env from the cwd, then from the package directory, so `wiro-gateway start`
# works whether or not the user is inside the project folder.
_env_candidates = [Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"]
for p in _env_candidates:
    if p.exists():
        load_dotenv(p, override=False)
        break


_VALID_EFFORTS = ("none", "low", "medium", "high")


class Settings(BaseSettings):
    """Runtime configuration. All fields are overridable via env vars."""

    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    # Wiro credentials
    wiro_api_key: str = Field(default="", alias="WIRO_API_KEY")
    wiro_api_secret: str = Field(default="", alias="WIRO_API_SECRET")
    wiro_model: str = Field(default="qwen/qwen3-8-27b-uncensored", alias="WIRO_MODEL")
    wiro_api_base: str = Field(default="https://api.wiro.ai/v1", alias="WIRO_API_BASE")
    wiro_ws_base: str = Field(default="wss://socket.wiro.ai/v1", alias="WIRO_WS_BASE")

    # Gateway
    gateway_host: str = Field(default="127.0.0.1", alias="GATEWAY_HOST")
    gateway_port: int = Field(default=8765, alias="GATEWAY_PORT")

    # Polling
    wiro_poll_interval_sec: float = Field(default=1.0, alias="WIRO_POLL_INTERVAL_SEC")
    wiro_poll_timeout_sec: int = Field(default=600, alias="WIRO_POLL_TIMEOUT_SEC")

    # --- Reasoning effort (Wiro Qwen3.8-27B-Uncensored) ---
    # Default effort used when a request body doesn't ask for one. Accepts
    # none/low/medium/high (aliases: off/disabled/minimal/med/max/maximum).
    wiro_default_reasoning_effort: str = Field(default="none", alias="WIRO_DEFAULT_REASONING_EFFORT")
    # If true, the preset (low/medium/high) ALWAYS wins over the request body's
    # own temperature/top_p/top_k/max_tokens. If false, request body wins and we
    # only stamp the thinking toggles. Default true so the Wiro-recommended
    # sampling actually applies when thinking is on.
    wiro_preset_overrides_sampling: bool = Field(default=True, alias="WIRO_PRESET_OVERRIDES_SAMPLING")
    # Fallback explicit toggle: when true, reasoning effort auto-enables
    # enableThinking. When false, even effort=medium leaves thinking off
    # (useful for fine-grained "I want reasoning_effort but no chain of thought").
    wiro_enable_thinking_on_effort: bool = Field(default=True, alias="WIRO_ENABLE_THINKING_ON_EFFORT")

    @field_validator("wiro_default_reasoning_effort")
    @classmethod
    def _validate_effort(cls, v: str) -> str:
        s = (v or "").strip().lower()
        if s in _VALID_EFFORTS:
            return s
        alias_map = {
            "off": "none", "disabled": "none",
            "minimal": "low", "med": "medium", "max": "high", "maximum": "high",
        }
        if s in alias_map:
            return alias_map[s]
        raise ValueError(
            f"WIRO_DEFAULT_REASONING_EFFORT must be one of {_VALID_EFFORTS} (got {v!r})"
        )

    def require_credentials(self) -> None:
        """Raise if Wiro credentials are missing/look like placeholders."""
        if not self.wiro_api_key or not self.wiro_api_secret:
            raise RuntimeError(
                "WIRO_API_KEY and WIRO_API_SECRET are required. Run `wiro-gateway init` "
                "or copy .env.example to .env and fill them in."
            )
        if self.wiro_api_key.startswith("{{") or self.wiro_api_secret.startswith("{{"):
            raise RuntimeError(
                "Wiro credentials still contain template placeholders. Replace them."
            )


def get_settings() -> Settings:
    """Return a fresh Settings snapshot reflecting the current environment."""
    return Settings()
