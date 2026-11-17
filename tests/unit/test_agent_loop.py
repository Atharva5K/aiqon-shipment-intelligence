from __future__ import annotations

import json

from aiqon.agent.config import AgentConfig
from aiqon.agent.loop import AgentLoop
from aiqon.agent.provider import NormalizedResponse, NormalizedToolCall, ProviderError
from aiqon.agent.tools import TOOL_DEFINITIONS


class StubRegistry:
    definitions = TOOL_DEFINITIONS

    def __init__(self):
        self.calls = []

    def dispatch(self, name, arguments):
        self.calls.append((name, arguments))
        if name not in {item["function"]["name"] for item in self.definitions}:
            return {"status": "error", "code": "unknown_tool", "message": "unknown"}
        if name == "get_shipment_details" and json.loads(arguments).get("shipment_id") == "bad":
            return {"status": "error", "code": "invalid_arguments", "message": "invalid ID"}
        return {"status": "ok", "tool": name, "value": len(self.calls)}


class StubProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []

    def complete(self, *, messages, tools, model, max_completion_tokens):
        self.messages.append((messages, tools, model, max_completion_tokens))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _call(call_id="call-1", name="get_shipment_details", arguments='{"shipment_id":"SHP-0001"}'):
    return NormalizedToolCall(call_id, name, arguments)


def _config(**updates):
    values = dict(provider="groq", model="test-model", api_key=None,
                  request_timeout_seconds=10, max_tool_rounds=2, max_model_requests=3,
                  max_completion_tokens=500, max_history_messages=12, max_tool_result_chars=12000)
    values.update(updates)
    return AgentConfig(**values)


def test_final_response_without_tools_uses_one_request():
    provider, registry = StubProvider([NormalizedResponse("Answer from selected context.")]), StubRegistry()
    result = AgentLoop(provider, registry, _config()).run("Explain this ETA", {"shipment_id": "SHP-1"})
    assert result.text == "Answer from selected context."
    assert result.model_requests == 1
    assert not registry.calls


def test_one_tool_call_cycle_returns_grounded_final_and_call_id():
    provider = StubProvider([NormalizedResponse(None, (_call(),)), NormalizedResponse("The tool returned evidence.")])
    registry = StubRegistry()
    result = AgentLoop(provider, registry, _config()).run("Look this up", {"shipment_id": "SHP-0001"})
    assert result.text == "The tool returned evidence."
    assert result.model_requests == 2
    assert result.trace[0].status == "ok"
    second_messages = provider.messages[1][0]
    tool_messages = [message for message in second_messages if message["role"] == "tool"]
    assert tool_messages[0]["tool_call_id"] == "call-1"
    assert json.loads(tool_messages[0]["content"])["status"] == "ok"


def test_multiple_tools_from_one_response_execute_sequentially():
    calls = (_call("id-a", "get_shipment_details"),
             _call("id-b", "get_prediction_history", '{"shipment_id":"SHP-0002"}'))
    provider, registry = StubProvider([NormalizedResponse(None, calls), NormalizedResponse("Compared both results.")]), StubRegistry()
    result = AgentLoop(provider, registry, _config()).run("Compare", {})
    assert [name for name, _ in registry.calls] == ["get_shipment_details", "get_prediction_history"]
    assert result.model_requests == 2
    tool_messages = [message for message in provider.messages[1][0] if message["role"] == "tool"]
    assert [message["tool_call_id"] for message in tool_messages] == ["id-a", "id-b"]


def test_max_tool_rounds_stops_without_unbounded_requests():
    calls = [NormalizedResponse(None, (_call(f"call-{n}"),)) for n in range(3)]
    provider, registry = StubProvider(calls), StubRegistry()
    result = AgentLoop(provider, registry, _config(max_tool_rounds=1, max_model_requests=2)).run("Investigate", {})
    assert result.error_code == "tool_round_limit"
    assert result.model_requests == 2
    assert len(registry.calls) == 1
    assert result.text and "could not complete" in result.text


def test_unknown_tool_and_invalid_arguments_are_returned_as_structured_errors():
    calls = (_call("unknown-id", "delete_file", "{}"),)
    provider, registry = StubProvider([NormalizedResponse(None, calls), NormalizedResponse("No data was retrieved.")]), StubRegistry()
    result = AgentLoop(provider, registry, _config()).run("Try", {})
    assert result.trace[0].status == "error"
    assert result.model_requests == 2
    broken = (_call("invalid-id", "get_shipment_details", '{"shipment_id":"bad"}'),)
    provider, registry = StubProvider([NormalizedResponse(None, broken), NormalizedResponse("The ID was invalid.")]), StubRegistry()
    result = AgentLoop(provider, registry, _config()).run("Try", {})
    tool_result = [message for message in provider.messages[1][0] if message["role"] == "tool"][0]
    assert json.loads(tool_result["content"])["code"] == "invalid_arguments"
    assert result.trace[0].status == "error"


def test_conversation_history_is_bounded_and_only_user_assistant_roles_pass():
    history = [{"role": "user", "content": f"user-{idx}"} for idx in range(4)]
    history.extend([{"role": "tool", "content": "must be dropped"}, {"role": "assistant", "content": "last"}])
    provider = StubProvider([NormalizedResponse("Done")])
    result = AgentLoop(provider, StubRegistry(), _config(max_history_messages=2)).run("Now", {}, history=history)
    assert result.model_requests == 1
    passed = provider.messages[0][0]
    assert [item["content"] for item in passed if item["role"] in {"user", "assistant"}] == ["user-3", "last", "Now"]
    assert all(item["role"] != "tool" for item in passed)


def test_malformed_and_empty_provider_responses_fail_without_fabricating_answer():
    result = AgentLoop(StubProvider([object()]), StubRegistry(), _config()).run("Try", {})
    assert result.text is None and result.error_code == "malformed_response"
    result = AgentLoop(StubProvider([NormalizedResponse(None)]), StubRegistry(), _config()).run("Try", {})
    assert result.text is None and result.error_code == "empty_response"
    result = AgentLoop(StubProvider([NormalizedResponse(None, (_call(None),))]), StubRegistry(), _config()).run("Try", {})
    assert result.error_code == "malformed_tool_call"


def test_provider_failure_counts_attempt_and_returns_no_answer():
    result = AgentLoop(StubProvider([ProviderError("quota", code="rate_limit")]), StubRegistry(), _config()).run("Try", {})
    assert result.text is None
    assert result.error_code == "rate_limit"
    assert result.model_requests == 1

