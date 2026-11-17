"""Environment-backed configuration for optional Phase 5 inference."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


class AgentConfigurationError(ValueError):
    """The optional assistant is unavailable or incorrectly configured."""


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    try:
        value = default if raw is None else int(raw)
    except ValueError as exc:
        raise AgentConfigurationError(f"{name} must be an integer from {minimum} to {maximum}.") from exc
    if not minimum <= value <= maximum:
        raise AgentConfigurationError(f"{name} must be an integer from {minimum} to {maximum}.")
    return value


def _bounded_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    try:
        value = default if raw is None else float(raw)
    except ValueError as exc:
        raise AgentConfigurationError(f"{name} must be a number from {minimum} to {maximum}.") from exc
    if not minimum <= value <= maximum:
        raise AgentConfigurationError(f"{name} must be a number from {minimum} to {maximum}.")
    return value


@dataclass(frozen=True)
class AgentConfig:
    provider: str
    model: str
    api_key: str | None = field(repr=False, compare=False)
    request_timeout_seconds: float = 45.0
    max_tool_rounds: int = 2
    max_model_requests: int = 3
    max_completion_tokens: int = 1200
    max_history_messages: int = 12
    max_tool_result_chars: int = 12000

    @classmethod
    def from_environment(cls, *, load_env_file: bool = True) -> "AgentConfig":
        if load_env_file:
            try:
                from dotenv import load_dotenv
            except ImportError:
                # python-dotenv belongs to the optional assistant extra.
                # Core dashboard use continues with process environment only.
                pass
            else:
                load_dotenv(override=False)
        rounds = _bounded_int("AGENT_MAX_TOOL_ROUNDS", 2, 1, 4)
        requests = _bounded_int("AGENT_MAX_MODEL_REQUESTS", rounds + 1, 1, 5)
        if requests > rounds + 1:
            raise AgentConfigurationError("AGENT_MAX_MODEL_REQUESTS cannot exceed AGENT_MAX_TOOL_ROUNDS + 1.")
        provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
        model = os.getenv("LLM_MODEL", "openai/gpt-oss-20b").strip()
        if not provider or not model:
            raise AgentConfigurationError("LLM_PROVIDER and LLM_MODEL must not be empty.")
        api_key = os.getenv("GROQ_API_KEY", "").strip() or None
        if api_key and api_key.casefold() in {"your_groq_api_key_here", "your_api_key_here", "replace_me"}:
            api_key = None
        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            request_timeout_seconds=_bounded_float("LLM_REQUEST_TIMEOUT_SECONDS", 45.0, 1.0, 180.0),
            max_tool_rounds=rounds,
            max_model_requests=requests,
            max_completion_tokens=_bounded_int("AGENT_MAX_COMPLETION_TOKENS", 1200, 128, 8192),
            max_history_messages=_bounded_int("AGENT_MAX_HISTORY_MESSAGES", 12, 0, 24),
            max_tool_result_chars=_bounded_int("AGENT_MAX_TOOL_RESULT_CHARS", 12000, 1000, 30000),
        )

    @property
    def available(self) -> bool:
        return self.provider == "groq" and bool(self.api_key)


def provider_setup_message(config: AgentConfig) -> str | None:
    if config.provider != "groq":
        return f"Unsupported LLM_PROVIDER {config.provider!r}. This release supports groq only."
    if not config.api_key:
        return "AI Assistant is unavailable until GROQ_API_KEY is set. Add your own key to the environment or local .env file, then restart Streamlit."
    return None
