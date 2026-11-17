"""Explicit opt-in live API smoke test; ordinary pytest never spends quota."""
from __future__ import annotations

import os
import json

import pytest

from aiqon.agent.config import AgentConfig
from aiqon.agent.provider import create_provider
from aiqon.agent.tools import TOOL_DEFINITIONS


pytestmark = pytest.mark.skipif(
    os.getenv("AIQON_RUN_LIVE_GROQ_TESTS") != "1",
    reason="Set AIQON_RUN_LIVE_GROQ_TESTS=1 to opt in to a real Groq API request.",
)


def test_live_groq_chat_completion_smoke():
    config = AgentConfig.from_environment()
    if not config.api_key:
        pytest.skip("A valid GROQ_API_KEY is required for the opt-in smoke test.")
    provider = create_provider(config)
    response = provider.complete(messages=[
        {"role": "system", "content": "Reply with one short word."},
        {"role": "user", "content": "Say ready."},
    ], tools=[], model=config.model, max_completion_tokens=32)
    assert response.content
    assert not response.tool_calls


def test_live_groq_tool_call_smoke():
    """One opt-in request verifies tool-call normalization without executing it."""
    config = AgentConfig.from_environment()
    if not config.api_key:
        pytest.skip("A valid GROQ_API_KEY is required for the opt-in smoke test.")
    provider = create_provider(config)
    function = next(item for item in TOOL_DEFINITIONS
                    if item["function"]["name"] == "get_shipment_details")
    response = provider.complete(messages=[
        {"role": "system", "content": "You must call the get_shipment_details function. Do not answer in text."},
        {"role": "user", "content": "Call get_shipment_details for the synthetic shipment SHP-0001."},
    ], tools=[function], model=config.model, max_completion_tokens=128)
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.name == "get_shipment_details"
    assert json.loads(call.arguments) == {"shipment_id": "SHP-0001"}
