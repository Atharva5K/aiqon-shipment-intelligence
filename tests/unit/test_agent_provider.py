from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from aiqon.agent.config import AgentConfig, AgentConfigurationError, provider_setup_message
from aiqon.agent.provider import (GroqChatProvider, NormalizedResponse,
    ProviderError, create_provider)
from aiqon.agent.tools import TOOL_DEFINITIONS


def _config(**updates):
    values = dict(provider="groq", model="openai/gpt-oss-20b", api_key="key-for-test-only",
                  request_timeout_seconds=17, max_tool_rounds=2, max_model_requests=3,
                  max_completion_tokens=600, max_history_messages=12, max_tool_result_chars=12000)
    values.update(updates)
    return AgentConfig(**values)


class FakeCompletions:
    def __init__(self, response=None, error=None):
        self.response, self.error = response, error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def _response(content="text", tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_config_defaults_and_api_key_is_not_in_repr(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    config = AgentConfig.from_environment(load_env_file=False)
    assert config.provider == "groq"
    assert config.model == "openai/gpt-oss-20b"
    assert not config.available
    assert "key-for-test" not in repr(_config())
    assert "GROQ_API_KEY" in provider_setup_message(config)


def test_config_works_without_optional_python_dotenv(monkeypatch):
    monkeypatch.setitem(sys.modules, "dotenv", None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    config = AgentConfig.from_environment(load_env_file=True)
    assert not config.available


def test_config_reads_model_and_bounds_rounds_and_requests(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_MODEL", "custom/model")
    monkeypatch.setenv("GROQ_API_KEY", "secret-test-value")
    monkeypatch.setenv("AGENT_MAX_TOOL_ROUNDS", "2")
    monkeypatch.delenv("AGENT_MAX_MODEL_REQUESTS", raising=False)
    config = AgentConfig.from_environment(load_env_file=False)
    assert config.model == "custom/model"
    assert config.max_tool_rounds == 2 and config.max_model_requests == 3
    assert "secret-test-value" not in repr(config)
    monkeypatch.setenv("AGENT_MAX_TOOL_ROUNDS", "99")
    with pytest.raises(AgentConfigurationError):
        AgentConfig.from_environment(load_env_file=False)


def test_env_example_placeholder_is_not_treated_as_a_credential(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "your_groq_api_key_here")
    config = AgentConfig.from_environment(load_env_file=False)
    assert config.api_key is None
    assert not config.available


def test_factory_requires_key_and_rejects_unimplemented_provider():
    with pytest.raises(AgentConfigurationError, match="GROQ_API_KEY"):
        create_provider(_config(api_key=None))
    with pytest.raises(AgentConfigurationError, match="supports groq"):
        create_provider(_config(provider="openrouter"))


def test_groq_factory_uses_configured_timeout_and_zero_retries(monkeypatch):
    import groq

    completions = FakeCompletions(_response("ready"))
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    captured = {}

    def fake_groq(**kwargs):
        captured.update(kwargs)
        return client

    monkeypatch.setattr(groq, "Groq", fake_groq)
    provider = create_provider(_config())
    assert captured["timeout"] == 17
    assert captured["max_retries"] == 0
    result = provider.complete(messages=[{"role": "user", "content": "hi"}],
        tools=TOOL_DEFINITIONS, model="configured-model", max_completion_tokens=321)
    assert result == NormalizedResponse("ready")
    call = completions.calls[0]
    assert call["model"] == "configured-model"
    assert call["max_completion_tokens"] == 321
    assert call["tools"] is TOOL_DEFINITIONS
    assert call["parallel_tool_calls"] is False
    assert call["tool_choice"] == "auto"


def test_groq_adapter_normalizes_tool_calls_and_text():
    tool = SimpleNamespace(id="call-abc", function=SimpleNamespace(name="get_eta_prediction", arguments='{"shipment_id":"SHP-0001"}'))
    completion = FakeCompletions(_response(None, [tool]))
    provider = GroqChatProvider("test", 5, client=SimpleNamespace(chat=SimpleNamespace(completions=completion)))
    result = provider.complete(messages=[], tools=[], model="m", max_completion_tokens=100)
    assert result.content is None
    assert result.tool_calls[0].call_id == "call-abc"
    assert result.tool_calls[0].name == "get_eta_prediction"
    assert result.tool_calls[0].arguments == '{"shipment_id":"SHP-0001"}'


@pytest.mark.parametrize(("status", "code"), [(401, "authentication"), (403, "permission"), (429, "rate_limit"), (503, "provider_error")])
def test_groq_status_errors_are_sanitized_and_not_retried(status, code):
    error = RuntimeError("provider exception contains internal detail")
    error.status_code = status
    completions = FakeCompletions(error=error)
    provider = GroqChatProvider("test", 5, client=SimpleNamespace(chat=SimpleNamespace(completions=completions)))
    with pytest.raises(ProviderError) as raised:
        provider.complete(messages=[], tools=[], model="m", max_completion_tokens=10)
    assert raised.value.code == code
    assert "internal detail" not in str(raised.value)
    assert len(completions.calls) == 1


@pytest.mark.parametrize(("error_type", "code"), [("APITimeoutError", "timeout"), ("APIConnectionError", "network")])
def test_groq_network_and_timeout_errors_are_sanitized(error_type, code):
    error = type(error_type, (RuntimeError,), {})("secret transport details")
    provider = GroqChatProvider("test", 5, client=SimpleNamespace(chat=SimpleNamespace(
        completions=FakeCompletions(error=error))))
    with pytest.raises(ProviderError) as raised:
        provider.complete(messages=[], tools=[], model="m", max_completion_tokens=10)
    assert raised.value.code == code
    assert "secret transport" not in str(raised.value)


def test_empty_or_malformed_groq_responses_fail_safely():
    empty = GroqChatProvider("test", 5, client=SimpleNamespace(chat=SimpleNamespace(
        completions=FakeCompletions(_response(None, None)))))
    with pytest.raises(ProviderError, match="empty"):
        empty.complete(messages=[], tools=[], model="m", max_completion_tokens=10)
    malformed = GroqChatProvider("test", 5, client=SimpleNamespace(chat=SimpleNamespace(
        completions=FakeCompletions(SimpleNamespace(choices=[])))))
    with pytest.raises(ProviderError, match="no completion"):
        malformed.complete(messages=[], tools=[], model="m", max_completion_tokens=10)
