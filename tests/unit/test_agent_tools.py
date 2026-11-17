from __future__ import annotations

import json

import pandas as pd
import pytest

from aiqon.agent.tools import AgentToolService, TOOL_DEFINITIONS, ToolRegistry
from aiqon.dashboard.config import DashboardPaths
from aiqon.dashboard.data_loader import load_dashboard_data
from aiqon.dashboard.services.dashboard_service import DashboardService
from aiqon.dashboard.services.explanation_service import ExplanationBundle, load_explanations
from aiqon.dashboard.services.prediction_adapter import Phase3PredictionAdapter
from aiqon.dashboard.services.scenario_service import ScenarioService


@pytest.fixture(scope="module")
def agent_services():
    paths = DashboardPaths()
    data = load_dashboard_data(paths.root)
    predictor = Phase3PredictionAdapter.load(paths.artifacts / "phase3_bundle.joblib")
    dashboard = DashboardService(data, predictor)
    explanations = load_explanations(paths.metrics, paths.figures)
    scenarios = ScenarioService(predictor)
    service = AgentToolService(dashboard, explanations, scenarios,
                               selected_shipment_id="SHP-0002", selected_checkpoint="departure")
    return service, ToolRegistry(service)


def test_registry_exposes_exactly_six_tools_and_rejects_unknown_or_extra_args(agent_services):
    _, registry = agent_services
    names = [item["function"]["name"] for item in registry.definitions]
    assert names == ["get_shipment_details", "get_eta_prediction", "get_prediction_history",
                     "get_delay_attribution", "query_shipments", "run_what_if_simulation"]
    unknown = registry.dispatch("delete_records", "{}")
    assert unknown["status"] == "error" and unknown["code"] == "unknown_tool"
    invalid = registry.dispatch("get_shipment_details", '{"shipment_id":"SHP-0002","extra":true}')
    assert invalid["code"] == "invalid_arguments"


def test_shipment_details_and_eta_are_json_compatible_and_use_context(agent_services):
    _, registry = agent_services
    details = registry.dispatch("get_shipment_details", '{"shipment_id":"SHP-0002"}')
    assert details["status"] == "ok"
    assert details["shipment"]["volume"]["unit"] == "unspecified in Shipment schema"
    assert "scheduled_delivery_at_utc" in details["shipment"]
    eta = registry.dispatch("get_eta_prediction", '{"shipment_id":"SHP-0002"}')
    assert eta["status"] == "ok"
    assert eta["prediction"]["checkpoint"] == "departure"
    assert eta["prediction"]["predicted_remaining_hours"] >= 0
    json.dumps(details, allow_nan=False)
    json.dumps(eta, allow_nan=False)
    missing = registry.dispatch("get_shipment_details", '{"shipment_id":"SHP-DOES-NOT-EXIST"}')
    assert missing["status"] == "error"


def test_prediction_history_has_all_checkpoints_and_no_actual_outcomes(agent_services):
    _, registry = agent_services
    result = registry.dispatch("get_prediction_history", '{"shipment_id":"SHP-0002"}')
    assert result["status"] == "ok"
    assert [row["checkpoint"] for row in result["predictions"]] == ["departure", "port_arrival", "customs_cleared"]
    forbidden = {"actual_delivery_at", "actual_delay_hours", "actual_delayed", "actual_significant_delay",
                 "remaining_hours_actual", "significant_delay_label"}
    assert not forbidden.intersection(result)
    assert all(not forbidden.intersection(row) for row in result["predictions"])
    assert result["selected_checkpoint"] == "departure"
    assert result["temporal_scope"]["as_of_checkpoints"] == ["departure"]
    assert result["temporal_scope"]["later_checkpoints"] == ["port_arrival", "customs_cleared"]
    assert [row["available_as_of_selected_checkpoint"] for row in result["predictions"]] == [True, False, False]
    assert all(row["temporal_label"] == expected for row, expected in zip(result["predictions"], [
        "available_as_of_selected_checkpoint", "retrospective_later_checkpoint", "retrospective_later_checkpoint"]))
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(("checkpoint", "as_of", "later", "flags"), [
    ("departure", ["departure"], ["port_arrival", "customs_cleared"], [True, False, False]),
    ("port_arrival", ["departure", "port_arrival"], ["customs_cleared"], [True, True, False]),
    ("customs_cleared", ["departure", "port_arrival", "customs_cleared"], [], [True, True, True]),
])
def test_prediction_history_marks_temporal_scope_for_selected_checkpoint(agent_services, checkpoint, as_of, later, flags):
    service, _ = agent_services
    scoped = AgentToolService(service.dashboard, service.explanations, service.scenarios,
        selected_shipment_id=service.selected_shipment_id, selected_checkpoint=checkpoint)
    result = ToolRegistry(scoped).dispatch("get_prediction_history", '{"shipment_id":"SHP-0002"}')
    assert result["status"] == "ok"
    assert result["selected_checkpoint"] == checkpoint
    assert result["selected_checkpoint_timestamp_utc"].endswith("Z")
    assert result["temporal_scope"]["as_of_checkpoints"] == as_of
    assert result["temporal_scope"]["later_checkpoints"] == later
    assert [row["available_as_of_selected_checkpoint"] for row in result["predictions"]] == flags
    assert len(result["predictions"]) == 3
    json.dumps(result, allow_nan=False)


def test_attribution_keeps_prediction_driver_shap_and_hindsight_distinct(agent_services):
    _, registry = agent_services
    result = registry.dispatch("get_delay_attribution", '{"shipment_id":"SHP-0002"}')
    assert result["status"] == "ok"
    assert result["checkpoint"] == "departure"
    assert result["prediction_driver_attribution"]["status"] == "available"
    assert result["shap_residual_attribution"]["status"] == "available"
    assert result["retrospective_cause_attribution"]["status"] == "not_exposed"
    json.dumps(result, allow_nan=False)


def test_missing_attribution_is_reported_without_fabrication(agent_services):
    service, _ = agent_services
    empty = ExplanationBundle(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None,
                              {"prediction_driver": "driver file absent", "shap_local": "local file absent"})
    registry = ToolRegistry(AgentToolService(service.dashboard, empty, service.scenarios,
        selected_shipment_id=service.selected_shipment_id,
        selected_checkpoint=service.selected_checkpoint))
    result = registry.dispatch("get_delay_attribution", '{"shipment_id":"SHP-0002"}')
    assert result["prediction_driver_attribution"]["status"] == "unavailable"
    assert result["prediction_driver_attribution"]["rows"] == []
    assert result["shap_residual_attribution"]["status"] == "unavailable"
    assert result["shap_residual_attribution"]["rows"] == []


def test_query_filters_are_bounded_and_never_use_actual_fields(agent_services):
    _, registry = agent_services
    result = registry.dispatch("query_shipments", json.dumps({"filters": {
        "checkpoint": "departure", "predicted_delay_above_hours": 5,
        "significant_delay_probability_above": 0.1, "route_id": "nhava_sheva_inland_a",
        "limit": 3}}))
    assert result["status"] == "ok"
    assert result["checkpoint"] == "departure"
    assert len(result["results"]) <= 3
    assert all(row["predicted_total_delay_hours"] > 5 for row in result["results"])
    assert all("actual_delay_hours" not in row for row in result["results"])
    single_id = registry.dispatch("query_shipments", '{"filters":{"shipment_ids":["SHP-0002"],"limit":3}}')
    assert single_id["status"] == "ok"
    assert all(row["shipment_id"] == "SHP-0002" for row in single_id["results"])
    invalid = registry.dispatch("query_shipments", '{"filters":{"significant_delay_probability_above":1.5}}')
    assert invalid["code"] == "invalid_arguments"
    too_many = registry.dispatch("query_shipments", '{"filters":{"limit":26}}')
    assert too_many["code"] == "invalid_arguments"
    json.dumps(result, allow_nan=False)


def test_what_if_reuses_scenario_service_and_rejects_unsupported_parameters(agent_services):
    _, registry = agent_services
    result = registry.dispatch("run_what_if_simulation", json.dumps({
        "shipment_id": "SHP-0002", "scenario_parameters": {"port_congestion": 0.95}}))
    assert result["status"] == "ok"
    assert result["label"].startswith("SIMULATED SCENARIO")
    assert result["source_state_unchanged"] is True
    json.dumps(result, allow_nan=False)
    unsupported = registry.dispatch("run_what_if_simulation", json.dumps({
        "shipment_id": "SHP-0002", "checkpoint": "customs_cleared",
        "scenario_parameters": {"port_congestion": 0.95}}))
    assert unsupported["status"] == "error"


@pytest.mark.parametrize("parameters", [{}, {"port_congestion": None, "weather_severity": None}])
def test_what_if_rejects_empty_or_all_null_parameters(agent_services, parameters):
    _, registry = agent_services
    result = registry.dispatch("run_what_if_simulation", json.dumps({
        "shipment_id": "SHP-0002", "scenario_parameters": parameters}))
    assert result["status"] == "error"
    assert result["code"] == "invalid_arguments"


def test_what_if_accepts_multiple_non_null_parameters_and_preserves_source(agent_services):
    _, registry = agent_services
    result = registry.dispatch("run_what_if_simulation", json.dumps({
        "shipment_id": "SHP-0002", "scenario_parameters": {
            "port_congestion": 0.95, "berth_service_hours": 4.0}}))
    assert result["status"] == "ok"
    assert result["source_state_unchanged"] is True

