"""Provider-neutral chat contract and the official Groq SDK adapter."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from aiqon.agent.config import AgentConfig, AgentConfigurationError


class ProviderError(RuntimeError):
    """Sanitized provider failure safe to display to a user."""

    def __init__(self, message: str, *, code: str = "provider_error"):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class NormalizedToolCall:
    call_id: str | None
    name: str | None
    arguments: str | None


@dataclass(frozen=True)
class NormalizedResponse:
    content: str | None
    tool_calls: tuple[NormalizedToolCall, ...] = ()


class ChatProvider(Protocol):
    def complete(self, *, messages: list[dict], tools: list[dict], model: str,
                 max_completion_tokens: int) -> NormalizedResponse: ...


class GroqChatProvider:
    """Groq chat-completions adapter with retries disabled and bounded timeout."""

    def __init__(self, api_key: str, timeout_seconds: float, *, client=None):
        if not api_key:
            raise AgentConfigurationError("GROQ_API_KEY is required to create the Groq provider.")
        if client is None:
            try:
                from groq import Groq
            except ImportError as exc:
                raise AgentConfigurationError("Install the optional assistant extra with `python -m pip install \".[assistant]\"`.") from exc
            client = Groq(api_key=api_key, timeout=timeout_seconds, max_retries=0)
        self._client = client

    def complete(self, *, messages, tools, model, max_completion_tokens) -> NormalizedResponse:
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                parallel_tool_calls=False,
                max_completion_tokens=max_completion_tokens,
            )
            if not getattr(response, "choices", None):
                raise ProviderError("Groq returned no completion choices.", code="malformed_response")
            message = response.choices[0].message
            calls = []
            for item in (getattr(message, "tool_calls", None) or ()):
                function = getattr(item, "function", None)
                calls.append(NormalizedToolCall(
                    call_id=getattr(item, "id", None),
                    name=getattr(function, "name", None),
                    arguments=getattr(function, "arguments", None),
                ))
            content = getattr(message, "content", None)
            if content is not None and not isinstance(content, str):
                raise ProviderError("Groq returned a malformed text response.", code="malformed_response")
            if not content and not calls:
                raise ProviderError("Groq returned an empty response.", code="empty_response")
            return NormalizedResponse(content=content, tool_calls=tuple(calls))
        except ProviderError:
            raise
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            class_name = type(exc).__name__
            if status == 401:
                raise ProviderError("Groq rejected the API key (401). Check GROQ_API_KEY.", code="authentication") from None
            if status == 403:
                raise ProviderError("Groq denied access (403). Check account and model access.", code="permission") from None
            if status == 429:
                raise ProviderError("Groq rate limit or quota was reached (429). No automatic retry was made.", code="rate_limit") from None
            if class_name in {"APITimeoutError", "TimeoutException"}:
                raise ProviderError("Groq request timed out. Try again later.", code="timeout") from None
            if class_name in {"APIConnectionError", "ConnectError", "ReadError", "WriteError"}:
                raise ProviderError("Could not reach Groq. Check network access and try again.", code="network") from None
            if status is not None:
                raise ProviderError(f"Groq returned HTTP {status}. Check the model and request configuration.", code="provider_error") from None
            raise ProviderError("Groq request failed unexpectedly. Check provider configuration and try again.", code="provider_error") from None


def create_provider(config: AgentConfig, *, client=None) -> ChatProvider:
    """Construct a provider only when explicitly requested by the assistant UI."""
    if config.provider != "groq":
        raise AgentConfigurationError(f"Unsupported LLM_PROVIDER {config.provider!r}. This release supports groq only.")
    if not config.api_key:
        raise AgentConfigurationError("GROQ_API_KEY is not configured.")
    return GroqChatProvider(config.api_key, config.request_timeout_seconds, client=client)
