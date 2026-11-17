"""Bounded sequential tool-calling loop independent of the LLM provider."""
from __future__ import annotations

import json
from dataclasses import dataclass

from aiqon.agent.config import AgentConfig
from aiqon.agent.prompts import SYSTEM_PROMPT
from aiqon.agent.provider import ChatProvider, NormalizedResponse
from aiqon.agent.tools import ToolRegistry


@dataclass(frozen=True)
class ToolTrace:
    name: str
    status: str
    summary: str


@dataclass(frozen=True)
class AgentResult:
    text: str | None
    trace: tuple[ToolTrace, ...]
    model_requests: int
    error_code: str | None = None


class AgentLoop:
    def __init__(self, provider: ChatProvider, registry: ToolRegistry, config: AgentConfig):
        self.provider = provider
        self.registry = registry
        self.config = config

    @staticmethod
    def _history(messages: list[dict], maximum: int) -> list[dict]:
        safe = []
        for item in messages:
            if not isinstance(item, dict) or item.get("role") not in {"user", "assistant"}:
                continue
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                safe.append({"role": item["role"], "content": content[:3000]})
        return safe[-maximum:] if maximum else []

    def run(self, user_message: str, context: dict, *, history: list[dict] | None = None) -> AgentResult:
        if not isinstance(user_message, str) or not user_message.strip() or len(user_message) > 3000:
            return AgentResult(None, (), 0, "invalid_message")
        context_json = json.dumps(context, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        if len(context_json) > 12000:
            return AgentResult(None, (), 0, "context_too_large")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": "Selected shipment/checkpoint context from deterministic Python services (JSON):\n" + context_json},
            *self._history(history or [], self.config.max_history_messages),
            {"role": "user", "content": user_message.strip()},
        ]
        trace: list[ToolTrace] = []
        requests = 0
        rounds_used = 0

        while requests < self.config.max_model_requests:
            requests += 1
            try:
                response = self.provider.complete(
                    messages=messages, tools=self.registry.definitions,
                    model=self.config.model,
                    max_completion_tokens=self.config.max_completion_tokens)
            except Exception as exc:
                code = getattr(exc, "code", "provider_error")
                message = str(exc)[:600] if getattr(exc, "code", None) else "The assistant provider failed. No answer was generated."
                return AgentResult(None, tuple(trace), requests, str(code))
            if not isinstance(response, NormalizedResponse):
                return AgentResult(None, tuple(trace), requests, "malformed_response")
            calls = response.tool_calls
            if not calls:
                if response.content and response.content.strip():
                    return AgentResult(response.content[:12000], tuple(trace), requests)
                return AgentResult(None, tuple(trace), requests, "empty_response")

            # The configured GPT-OSS 20B model does not support parallel tool
            # execution. Even if a response contains several calls, execute
            # each on this thread and append each result before requesting again.
            if rounds_used >= self.config.max_tool_rounds:
                return AgentResult(self._limit_message(trace), tuple(trace), requests, "tool_round_limit")
            rounds_used += 1
            if len(calls) > 6:
                trace.append(ToolTrace("tool_call_batch", "limit", "At most six calls are accepted per model response."))
                return AgentResult(self._limit_message(trace), tuple(trace), requests, "tool_call_limit")

            ids = [call.call_id for call in calls]
            if any(not isinstance(call_id, str) or not call_id or len(call_id) > 128 for call_id in ids) or len(set(ids)) != len(ids):
                return AgentResult(None, tuple(trace), requests, "malformed_tool_call")
            if any(not isinstance(call.name, str) or not call.name or len(call.name) > 80
                   or not isinstance(call.arguments, str) for call in calls):
                return AgentResult(None, tuple(trace), requests, "malformed_tool_call")
            assistant_calls = []
            for call in calls:
                assistant_calls.append({"id": call.call_id, "type": "function", "function": {
                    "name": call.name, "arguments": call.arguments}})
            messages.append({"role": "assistant", "content": response.content,
                             "tool_calls": assistant_calls})
            for call in calls:
                result = self.registry.dispatch(call.name, call.arguments)
                status = str(result.get("status", "error"))
                trace.append(ToolTrace(call.name, status,
                    "Completed." if status == "ok" else str(result.get("message", "Tool returned an error."))[:180]))
                content = self._bounded_json(result)
                messages.append({"role": "tool", "tool_call_id": call.call_id,
                                 "name": str(call.name or "unknown")[:80], "content": content})

        return AgentResult(self._limit_message(trace), tuple(trace), requests, "request_limit")

    def _bounded_json(self, result: dict) -> str:
        encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        if len(encoded) <= self.config.max_tool_result_chars:
            return encoded
        preview_length = max(100, self.config.max_tool_result_chars - 220)
        wrapper = {"status": "truncated", "message": "Tool result exceeded the configured size limit; only this JSON preview is available.",
                   "preview": encoded[:preview_length]}
        return json.dumps(wrapper, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _limit_message(trace: list[ToolTrace]) -> str:
        completed = [step.name for step in trace if step.status == "ok"]
        if completed:
            names = ", ".join(completed)
            return f"I reached the configured investigation limit after retrieving evidence from: {names}. I could not complete the full response, so I will not infer missing values."
        return "I reached the configured investigation limit before retrieving evidence. I cannot provide a grounded answer from the available results."
