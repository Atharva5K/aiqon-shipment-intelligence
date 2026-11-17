from __future__ import annotations

import json

from aiqon.agent.config import AgentConfig
from aiqon.agent.loop import AgentLoop
from aiqon.agent.provider import NormalizedResponse, NormalizedToolCall
from aiqon.agent.tools import AgentToolService, ToolRegistry
from aiqon.dashboard.config import DashboardPaths
from aiqon.dashboard.data_loader import load_dashboard_data
from aiqon.dashboard.services.dashboard_service import DashboardService
from aiqon.dashboard.services.explanation_service import load_explanations
from aiqon.dashboard.services.prediction_adapter import Phase3PredictionAdapter
from aiqon.dashboard.services.scenario_service import ScenarioService


class DeterministicTestProvider:
    """Test-only provider stub; never created by application runtime code."""
    def __init__(self):
        self.calls = 0

    def complete(self, *, messages, tools, model, max_completion_tokens):
        self.calls += 1
        if self.calls == 1:
            return NormalizedResponse(None, (NormalizedToolCall("call-attribution", "get_delay_attribution",
                '{"shipment_id":"SHP-0002"}'),))
        result = next(item for item in messages if item.get("role") == "tool")
        data = json.loads(result["content"])
        return NormalizedResponse(f"Evidence status: {data['prediction_driver_attribution']['status']}; checkpoint: {data['checkpoint']}.")


def test_agent_loop_uses_real_phase4_services_without_live_provider():
    paths = DashboardPaths()
    data = load_dashboard_data(paths.root)
    predictor = Phase3PredictionAdapter.load(paths.artifacts / "phase3_bundle.joblib")
    dashboard = DashboardService(data, predictor)
    explanations = load_explanations(paths.metrics, paths.figures)
    tools = AgentToolService(dashboard, explanations, ScenarioService(predictor),
        selected_shipment_id="SHP-0002", selected_checkpoint="departure")
    provider = DeterministicTestProvider()
    config = AgentConfig(provider="groq", model="test-only-model", api_key=None,
        max_tool_rounds=2, max_model_requests=3, max_completion_tokens=400,
        max_history_messages=12, max_tool_result_chars=12000)
    result = AgentLoop(provider, ToolRegistry(tools), config).run(
        "Explain the prediction drivers.", {"shipment_id": "SHP-0002", "checkpoint": "departure"})
    assert result.text == "Evidence status: available; checkpoint: departure."
    assert result.model_requests == provider.calls == 2
    assert result.trace[0].name == "get_delay_attribution"
    assert "retrospective" not in result.text.lower()
