from __future__ import annotations

from datetime import datetime, timedelta
import hashlib

import pytest
import pandas as pd
import joblib

from aiqon.dashboard.config import DashboardPaths
from aiqon.dashboard.data_loader import DashboardDataError, load_dashboard_data
from aiqon.dashboard.metrics import filter_dashboard_frame, shipment_level_frame
from aiqon.dashboard.services.dashboard_service import DashboardService
from aiqon.dashboard.services.explanation_service import load_explanations
from aiqon.dashboard.services.prediction_adapter import DashboardArtifactError, Phase3PredictionAdapter
from aiqon.dashboard.services.scenario_service import ScenarioParameters, ScenarioService
from aiqon.dashboard.pages.what_if import _first_present
from aiqon.prediction.features import build_features


@pytest.fixture(scope="module")
def paths():
    return DashboardPaths()


@pytest.fixture(scope="module")
def loaded(paths):
    data = load_dashboard_data(paths.root)
    predictor = Phase3PredictionAdapter.load(paths.artifacts / "phase3_bundle.joblib")
    return data, predictor


def test_dashboard_loads_exact_cohort_and_keeps_targets_separate(loaded):
    data, _ = loaded
    assert data.shipment_count == 300
    assert data.snapshot_count == 900
    assert len({shipment.route_id for shipment in data.shipments.values()}) == 3
    assert len(data.actual_events) == 2400
    assert len(data.prediction_targets) == 900
    assert all(isinstance(snapshot.checkpoint_at, datetime)
               and snapshot.checkpoint_at.tzinfo is not None for snapshot in data.snapshots.values())


def test_dashboard_predictions_use_phase3_adapter_and_all_checkpoints(loaded):
    data, predictor = loaded
    service = DashboardService(data, predictor)
    frame = service.prediction_frame()
    assert len(frame) == 900
    assert set(frame.checkpoint_type) == {"departure", "port_arrival", "customs_cleared"}
    assert frame.groupby("shipment_id").size().eq(3).all()
    assert frame.predicted_remaining_hours.ge(0).all()
    assert frame.probability_delay_over_12h.between(0, 1).all()
    sample = data.snapshots[("SHP-0001", "departure")]
    prediction = predictor.predict(sample, data.shipments["SHP-0001"])
    assert prediction.validated_output is not None or prediction.interval_status.startswith("unavailable")
    assert prediction.predicted_delivery_at == sample.checkpoint_at + timedelta(hours=prediction.predicted_remaining_hours)


def test_checkpoint_visibility_hides_future_actuals(loaded):
    data, predictor = loaded
    service = DashboardService(data, predictor)
    departure = service.visible_actual_events("SHP-0001", "departure")
    customs = service.visible_actual_events("SHP-0001", "customs_cleared")
    retrospective = service.visible_actual_events("SHP-0001", "departure", retrospective=True)
    assert "delivery" not in set(departure.event_type)
    assert "gate_out" not in set(customs.event_type)
    assert "delivery" not in set(customs.event_type)
    assert "delivery" in set(retrospective.event_type)
    assert (departure.actual_at <= data.snapshots[("SHP-0001", "departure")].checkpoint_at).all()


def test_overview_filters_and_shipment_units(loaded):
    data, predictor = loaded
    service = DashboardService(data, predictor)
    full = service.frame()
    one_route = sorted(full.route_id.unique())[0]
    filtered = filter_dashboard_frame(full, route=one_route, checkpoint="customs_cleared")
    shipments = shipment_level_frame(filtered, "customs_cleared")
    assert shipments.shipment_id.is_unique
    assert set(shipments.route_id) == {one_route}
    assert len(shipments) == filtered.shipment_id.nunique()


def test_scenario_is_reproducible_and_does_not_mutate_sources(loaded, tmp_path, monkeypatch):
    data, predictor = loaded
    paths = DashboardPaths()
    scenario = ScenarioService(predictor)
    shipment = data.shipments["SHP-0001"]
    snapshot = data.snapshots[(shipment.shipment_id, "departure")]
    original_snapshot = snapshot.model_dump(mode="json")
    original_route = predictor.routes[shipment.route_id].model_dump(mode="json")
    original_world = predictor.bundle["world"].model_dump(mode="json")
    bundle_fingerprint = joblib.hash(predictor.bundle)
    calibration_fingerprint = joblib.hash(predictor.calibration)
    history = tmp_path / "prediction_history.jsonl"
    history.write_text('{"sentinel":true}\n', encoding="utf-8")
    history_before = history.read_bytes()
    persistent_files = [path for path in (paths.root / "data").rglob("*") if path.is_file()]
    persistent_files.extend(path for path in (
        paths.root / "artifacts" / "prediction_history.jsonl",
        paths.artifacts / "phase3_bundle.joblib",
        paths.artifacts / "phase3_calibration.joblib") if path.is_file())
    persisted_before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in persistent_files}
    params = ScenarioParameters(port_congestion=0.95, berth_capacity=1,
                                berth_service_hours=9.0, inland_traffic=0.9)
    first = scenario.run(snapshot, shipment, params)
    second = scenario.run(snapshot, shipment, params)
    assert first.modified == second.modified
    assert first.stage_deltas_hours == second.stage_deltas_hours
    assert first.source_state_unchanged
    assert first.source_snapshot_sha256_before == first.source_snapshot_sha256_after
    assert snapshot.model_dump(mode="json") == original_snapshot
    assert predictor.routes[shipment.route_id].model_dump(mode="json") == original_route
    assert predictor.bundle["world"].model_dump(mode="json") == original_world
    assert joblib.hash(predictor.bundle) == bundle_fingerprint
    assert joblib.hash(predictor.calibration) == calibration_fingerprint
    assert history.read_bytes() == history_before
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in persistent_files} == persisted_before
    with pytest.raises(ValueError, match="unsupported"):
        scenario.run(data.snapshots[(shipment.shipment_id, "customs_cleared")], shipment,
                     {"port_congestion": 0.9})
    assert scenario.reset_parameters() == ScenarioParameters()


def test_scenario_controls_validate_bounds_and_support(loaded):
    data, predictor = loaded
    service = ScenarioService(predictor)
    shipment = data.shipments["SHP-0001"]
    snapshot = data.snapshots[(shipment.shipment_id, "customs_cleared")]
    support = service.parameter_support(snapshot, shipment)
    assert all(not support[name][0] for name in ("port_congestion", "weather_severity", "berth_capacity", "berth_service_hours", "customs_workload"))
    assert support["inland_traffic"][0]
    with pytest.raises(ValueError):
        ScenarioParameters(port_congestion=1.1)


def test_port_congestion_changes_berth_wait_component(loaded):
    data, predictor = loaded
    shipment = data.shipments["SHP-0001"]
    snapshot = data.snapshots[(shipment.shipment_id, "departure")]
    service = ScenarioService(predictor)
    baseline = service.run(snapshot, shipment, {})
    changed = service.run(snapshot, shipment, {"port_congestion": 0.99})
    assert changed.modified.propagation_components["berth_wait"] != baseline.modified.propagation_components["berth_wait"]


def test_berth_capacity_changes_residual_prediction_when_consumed(loaded):
    data, predictor = loaded
    shipment = data.shipments["SHP-0002"]
    snapshot = data.snapshots[(shipment.shipment_id, "port_arrival")]
    service = ScenarioService(predictor)
    baseline = service.run(snapshot, shipment, {})
    changed = service.run(snapshot, shipment, {"berth_capacity": 1})
    assert changed.modified.ml_residual_hours != baseline.modified.ml_residual_hours


def test_berth_service_time_changes_port_handling_component(loaded):
    data, predictor = loaded
    shipment = data.shipments["SHP-0001"]
    snapshot = data.snapshots[(shipment.shipment_id, "departure")]
    service = ScenarioService(predictor)
    baseline = service.run(snapshot, shipment, {})
    changed = service.run(snapshot, shipment, {"berth_service_hours": 12.0})
    assert changed.modified.propagation_components["port_handling"] != baseline.modified.propagation_components["port_handling"]


def test_customs_workload_changes_processing_component(loaded):
    data, predictor = loaded
    shipment = data.shipments["SHP-0001"]
    snapshot = data.snapshots[(shipment.shipment_id, "departure")]
    service = ScenarioService(predictor)
    baseline = service.run(snapshot, shipment, {})
    changed = service.run(snapshot, shipment, {"customs_workload": 0.99})
    assert changed.modified.propagation_components["customs_processing"] != baseline.modified.propagation_components["customs_processing"]


def test_inland_traffic_changes_inland_travel_component(loaded):
    data, predictor = loaded
    shipment = data.shipments["SHP-0001"]
    snapshot = data.snapshots[(shipment.shipment_id, "customs_cleared")]
    service = ScenarioService(predictor)
    baseline = service.run(snapshot, shipment, {})
    changed = service.run(snapshot, shipment, {"inland_traffic": 0.99})
    assert changed.modified.propagation_components["inland_travel"] != baseline.modified.propagation_components["inland_travel"]


def test_eligible_weather_scenario_reaches_checkpoint_feature(loaded, monkeypatch):
    data, predictor = loaded
    shipment = data.shipments["SHP-0001"]
    snapshot = data.snapshots[(shipment.shipment_id, "port_arrival")]
    service = ScenarioService(predictor)
    assert service.parameter_support(snapshot, shipment)["weather_severity"][0]
    seen = []
    predict = predictor.predict

    def record_features(selected_snapshot, selected_shipment, **kwargs):
        route = kwargs.get("route") or predictor.routes[selected_shipment.route_id]
        world = kwargs.get("world") or predictor.bundle["world"]
        seen.append(build_features(selected_snapshot, selected_shipment, route, world))
        return predict(selected_snapshot, selected_shipment, **kwargs)

    monkeypatch.setattr(predictor, "predict", record_features)
    result = service.run(snapshot, shipment, {"weather_severity": 0.99})
    assert result.parameters["weather_severity"] == 0.99
    assert any(features["observed_port_weather"] == 0.99 for features in seen)
    # The current saved Phase 3 residual bundle has zero split importance for
    # port-weather features, so it correctly leaves the residual unchanged.
    assert result.modified.ml_residual_hours == result.original.ml_residual_hours


def test_zero_observation_and_forecast_values_are_preserved_as_defaults():
    assert _first_present(0.0, 0.7, 0.5) == 0.0
    assert _first_present(None, 0.0, 0.5) == 0.0
    assert _first_present(None, None, 0.5) == 0.5


def test_artifact_compatibility_rejects_missing_bundle(paths):
    with pytest.raises(DashboardArtifactError, match="missing"):
        Phase3PredictionAdapter.load(paths.artifacts / "absent.joblib")


def test_artifact_runtime_incompatibility_is_reported(paths, monkeypatch):
    import aiqon.dashboard.services.prediction_adapter as adapter_module

    monkeypatch.setattr(adapter_module, "validate_artifact_runtime_versions",
                        lambda metadata: (_ for _ in ()).throw(RuntimeError("version mismatch")))
    with pytest.raises(DashboardArtifactError, match="compatible|runtime"):
        Phase3PredictionAdapter.load(paths.artifacts / "phase3_bundle.joblib")


def test_empty_and_missing_dashboard_tables_fail_clearly(monkeypatch, tmp_path):
    import aiqon.dashboard.data_loader as loader_module

    monkeypatch.setattr(loader_module, "_read_table", lambda root, name: pd.DataFrame())
    with pytest.raises(DashboardDataError, match="must not be empty"):
        load_dashboard_data(tmp_path)

    def missing(root, name):
        raise FileNotFoundError(name)

    monkeypatch.setattr(loader_module, "_read_table", missing)
    with pytest.raises(DashboardDataError, match="unavailable"):
        load_dashboard_data(tmp_path)


def test_optional_explanations_load_independently(paths, tmp_path):
    bundle = load_explanations(tmp_path, tmp_path)
    assert bundle.retrospective_cause_attribution.empty
    assert bundle.prediction_driver_attribution.empty
    assert bundle.shap_global.empty
    assert bundle.shap_local.empty
    assert "retrospective" in bundle.unavailable
    assert bundle.shap_figure is None


def test_all_three_explanation_categories_load(paths):
    bundle = load_explanations(paths.metrics, paths.figures)
    assert not bundle.retrospective_cause_attribution.empty
    assert not bundle.prediction_driver_attribution.empty
    assert not bundle.shap_global.empty
    assert not bundle.shap_local.empty


def test_prediction_timeline_rows_have_no_future_actual_events(loaded):
    data, predictor = loaded
    service = DashboardService(data, predictor)
    snapshot = data.snapshots[("SHP-0001", "departure")]
    events = service.visible_actual_events("SHP-0001", "departure")
    assert all(event_time <= snapshot.checkpoint_at for event_time in events.actual_at)

