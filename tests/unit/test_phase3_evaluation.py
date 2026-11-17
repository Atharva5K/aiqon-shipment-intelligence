from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from aiqon.config.loader import load_config, load_routes
from aiqon.evaluation.bootstrap import shipment_bootstrap
from aiqon.evaluation.conformal import finite_sample_quantile, normalized_scores, prediction_interval
from aiqon.evaluation.metrics import probability_metrics, regression_metrics
from aiqon.evaluation.masking import apply_feature_mask, random_mask_features
from aiqon.evaluation.probability import normalized_signed_residuals, significant_delay_probabilities
from aiqon.evaluation.splits import labels_by_shipment, stratified_shipment_split
from aiqon.simulation.engine import simulate_cohort


def _labels(n=300, positives=264):
    return {f"SHP-{i:04d}": int(i <= positives) for i in range(1, n + 1)}


def test_exact_deterministic_stratified_shipment_split():
    labels = _labels()
    first = stratified_shipment_split(labels, 42)
    second = stratified_shipment_split(labels, 42)
    assert first == second
    assert {key: len(ids) for key, ids in first.items()} == {"fit": 210, "calibration": 45, "test": 45}
    assert len(set(first["fit"]) & set(first["calibration"])) == 0
    assert len(set(first["fit"]) & set(first["test"])) == 0
    assert len(set(first["calibration"]) & set(first["test"])) == 0
    assert set(sum(first.values(), [])) == set(labels)


def test_combined_experiment_fingerprint_tracks_data_configs_routes_and_model(tmp_path):
    from aiqon.evaluation.splits import experiment_fingerprints
    for folder, names in (("data/raw", ("shipments", "events", "environment_actuals", "forecasts")),
                          ("data/processed", ("prediction_snapshots", "snapshot_targets"))):
        directory = tmp_path / folder
        directory.mkdir(parents=True)
        for name in names:
            (directory / f"{name}.csv").write_text(f"initial:{name}", encoding="utf-8")
    config = tmp_path / "simulation.yaml"; config.write_text("simulation: v1", encoding="utf-8")
    routes = tmp_path / "routes.yaml"; routes.write_text("routes: v1", encoding="utf-8")
    model = {"n_estimators": 100, "learning_rate": .05, "max_depth": 2,
             "min_samples_leaf": 8, "random_state": 42}
    base = experiment_fingerprints(tmp_path, config, routes, model, 42)
    routes.write_text("routes: v2", encoding="utf-8")
    assert experiment_fingerprints(tmp_path, config, routes, model, 42)["experiment_fingerprint_sha256"] != base["experiment_fingerprint_sha256"]
    routes.write_text("routes: v1", encoding="utf-8")
    config.write_text("simulation: v2", encoding="utf-8")
    assert experiment_fingerprints(tmp_path, config, routes, model, 42)["experiment_fingerprint_sha256"] != base["experiment_fingerprint_sha256"]
    config.write_text("simulation: v1", encoding="utf-8")
    (tmp_path / "data/processed/prediction_snapshots.csv").write_text("processed: changed", encoding="utf-8")
    changed = experiment_fingerprints(tmp_path, config, routes, model, 42)
    assert changed["experiment_fingerprint_sha256"] != base["experiment_fingerprint_sha256"]
    assert changed["raw_dataset_fingerprint_sha256"] == base["raw_dataset_fingerprint_sha256"]
    assert changed["processed_dataset_fingerprint_sha256"] != base["processed_dataset_fingerprint_sha256"]


def test_label_consistency_and_exact_stratification_failures_are_clear():
    ships = pd.DataFrame({"shipment_id": ["a"]})
    targets = pd.DataFrame({"shipment_id": ["a", "a", "a"],
                            "significant_delay_label": [1, 1, 0]})
    with pytest.raises(ValueError, match="consistent across checkpoints"):
        labels_by_shipment(ships, targets)
    with pytest.raises(ValueError, match="only one class"):
        stratified_shipment_split({f"SHP-{i:04d}": 1 for i in range(300)}, 42)
    with pytest.raises(ValueError, match="exact stratified 70/15/15"):
        stratified_shipment_split(_labels(300, 299), 42)


def test_normalized_conformal_finite_sample_quantile_and_bounds():
    scores, scale = normalized_scores([10, 20], [8, 23], [0, 2], 1.0)
    assert scale.tolist() == [1, 2]
    assert scores.tolist() == pytest.approx([2, 1.5])
    assert finite_sample_quantile([1, 2, 3, 4, 5], .5) == 3
    assert np.isinf(finite_sample_quantile([1, 2], .9))
    lo, hi = prediction_interval([1, 5], [0, 2], 2, 1)
    assert np.all(lo >= 0) and np.all(hi >= lo)


def test_calibration_signed_residual_probability_uses_total_delivery_threshold():
    checkpoint = datetime(2025, 1, 1, tzinfo=timezone.utc)
    scheduled = checkpoint + timedelta(hours=10)
    residuals = normalized_signed_residuals([12, 8], [10, 10], [1, 2], 1)
    assert residuals.tolist() == pytest.approx([2, -1])
    p = significant_delay_probabilities([10], [1], [checkpoint], [scheduled], residuals, 12, 1)
    # Threshold is scheduled + 12h = 22h after checkpoint; corrected outcomes are 12h and 9h.
    assert p.tolist() == [0.0]


def test_significant_delay_probability_uses_strict_boundary_on_calibration_residuals():
    checkpoint = datetime(2025, 1, 1, tzinfo=timezone.utc)
    scheduled = checkpoint + timedelta(hours=10)
    residuals = normalized_signed_residuals([22.0, 22.1], [10.0, 10.0], [1.0, 1.0], 1.0)
    probability = significant_delay_probabilities([10.0], [1.0], [checkpoint], [scheduled],
        residuals, threshold_hours=12.0, minimum_scale_hours=1.0)
    # A corrected delivery exactly at the threshold is excluded; the 0.1h excess is included.
    assert probability.tolist() == [0.5]


def test_probability_metrics_handle_one_class_without_auc_claims():
    result = probability_metrics([1, 1, 1], [.9, .2, .8], .5)
    assert result["roc_auc"] is None
    assert "only one class" in result["roc_auc_unavailable_reason"]
    assert result["brier_score"] >= 0


def test_regression_metrics_use_carry_forward_as_improvement_reference():
    metrics = regression_metrics([10, 20], [11, 19], [15, 15])
    assert metrics["mae_hours"] == 1
    assert metrics["improvement_over_carry_forward_hours"] == 4


def test_bootstrap_samples_whole_shipments_and_reuses_paired_draws():
    frame = pd.DataFrame([
        {"shipment_id": f"s{i}", "checkpoint_type": cp, "actual": float(i),
         "carry_forward": float(i + 1), "model_a": float(i + 2), "model_b": float(i + 3)}
        for i in range(6) for cp in ("departure", "port_arrival", "customs_cleared")
    ])
    first, draws1 = shipment_bootstrap(frame, ("carry_forward", "model_a", "model_b"),
        ("departure", "port_arrival", "customs_cleared"), 20, 9)
    second, draws2 = shipment_bootstrap(frame, ("carry_forward", "model_a", "model_b"),
        ("departure", "port_arrival", "customs_cleared"), 20, 9)
    assert first == second and draws1 == draws2
    assert all(len(draw) == 6 for draw in draws1)
    for sampled_ids in draws1:
        sampled_frame = frame.set_index("shipment_id").loc[sampled_ids]
        assert all((sampled_frame.checkpoint_type == cp).sum() == len(sampled_ids)
                   for cp in ("departure", "port_arrival", "customs_cleared"))
    assert first["paired_comparisons"] is True


def test_masking_preserves_feature_schema_updates_flags_and_does_not_mutate_source():
    frame = pd.DataFrame({"actual_transit_hours": [5., 6.], "actual_transit_hours__missing": [0, 0],
                          "transit_deviation_congestion_interaction": [1., 2.],
                          "transit_deviation_congestion_interaction__missing": [0, 0], "route_id": ["a", "b"]})
    masked = apply_feature_mask(frame, ["actual_transit_hours"])
    assert list(masked.columns) == list(frame.columns)
    assert masked["actual_transit_hours"].isna().all()
    assert (masked["actual_transit_hours__missing"] == 1).all()
    assert masked["transit_deviation_congestion_interaction"].isna().all()
    assert frame["actual_transit_hours"].tolist() == [5., 6.]
    assert random_mask_features(frame, ["actual_transit_hours"], .5, 4).columns.tolist() == frame.columns.tolist()


def test_counterfactual_neutralization_is_opt_in_and_keeps_cohort_seeded():
    config = load_config("configs/simulation.yaml")
    routes = tuple(load_routes(config.routes.config_path).routes)
    config = config.model_copy(update={"data": config.data.model_copy(update={"n_shipments": 9})})
    original = simulate_cohort(config, routes)
    baseline_events = [event.model_dump() for event in original.events]
    repeated = simulate_cohort(config, routes)
    assert baseline_events == [event.model_dump() for event in repeated.events]
    counterfactual = simulate_cohort(config, routes, neutralize_mechanism="berth_slot")
    counterfactual_repeat = simulate_cohort(config, routes, neutralize_mechanism="berth_slot")
    assert [event.model_dump() for event in original.events] == baseline_events
    assert counterfactual.events != original.events
    assert counterfactual.event_rows == counterfactual_repeat.event_rows
    assert len(counterfactual.shipments) == len(original.shipments)


def test_amplification_controlled_examples_have_positive_input_and_expected_regimes():
    from aiqon.evaluation.phase3 import _amplification_experiments
    config = load_config("configs/simulation.yaml")
    route = load_routes(config.routes.config_path).routes[-1]
    result = _amplification_experiments(load_routes(config.routes.config_path).routes, config.world)
    assert all(x["injected_delay_hours_positive"] for x in result["results"].values())
    assert result["results"]["busy_window"]["amplification_factor"] > 1
    assert result["results"]["quiet_window"]["amplification_factor"] <= 1


def test_prediction_driver_attribution_rejects_oracle_fields(monkeypatch):
    import aiqon.evaluation.phase3 as phase3
    monkeypatch.setattr(phase3, "estimate_remaining_duration", lambda *args, **kwargs: (10.0, {}))
    base = {"snapshot": SimpleNamespace(forecasts=(), checkpoint_type="departure"),
            "shipment": SimpleNamespace(shipment_id="s"), "route": object(), "world": object()}
    rows = phase3._prediction_driver_attribution(None, [base], {"port_congestion": .5})
    assert rows[0]["forecast_used"] is False
    with pytest.raises(ValueError, match="as-of inputs only"):
        phase3._prediction_driver_attribution(None, [{**base, "actual_delivery_at": datetime.now(timezone.utc)}], {})


def _selected_port_forecast_case():
    from aiqon.evaluation.phase3 import _neutralize_selected_forecast
    from aiqon.prediction.propagation import estimate_remaining_duration
    from aiqon.prediction.training import _parse_snapshot, _read_table
    from aiqon.schemas.shipment import Shipment
    from aiqon.schemas.snapshots import Forecast
    config = load_config("configs/simulation.yaml")
    routes = {r.route_id: r for r in load_routes(config.routes.config_path).routes}
    shipments = {r["shipment_id"]: Shipment.model_validate(r)
                 for r in _read_table(".", "shipments").to_dict("records")}
    for _, row in _read_table(".", "prediction_snapshots").iterrows():
        snapshot = _parse_snapshot(row)
        route = routes[snapshot.route_id]
        trace = {}
        estimate_remaining_duration(snapshot, route, config.world, forecast_trace=trace)
        request = trace["port_congestion"]
        valid_at = request["requested_at"].replace(minute=0, second=0, microsecond=0)
        issued_at = min(snapshot.checkpoint_at, valid_at - timedelta(seconds=1))
        forecast = Forecast(location_id=route.destination_port, forecast_issued_at=issued_at,
            forecast_valid_at=valid_at, forecast_variable="port_congestion", forecast_value=.99)
        unused_weather = Forecast(location_id=route.destination_port, forecast_issued_at=issued_at,
            forecast_valid_at=valid_at, forecast_variable="port_weather", forecast_value=.99)
        snapshot = snapshot.model_copy(update={"forecasts": (forecast, unused_weather)})
        high = snapshot
        low = _neutralize_selected_forecast(high, forecast, .10)
        high_value = estimate_remaining_duration(high, route, config.world)[0]
        low_value = estimate_remaining_duration(low, route, config.world)[0]
        if abs(high_value - low_value) > 1e-9:
            return high, shipments[snapshot.shipment_id], route, config.world, forecast
    raise AssertionError("Dataset has no consumed port-congestion vintage with nonzero propagation sensitivity")


def test_driver_attribution_uses_exact_eligible_vintage_and_mutates_only_that_forecast():
    from aiqon.evaluation.phase3 import _neutralize_selected_forecast, _prediction_driver_attribution
    from aiqon.prediction.propagation import estimate_remaining_duration, select_eligible_forecast
    snapshot, shipment, route, world, selected = _selected_port_forecast_case()
    before = snapshot.model_dump(mode="python")
    trace = {}
    original, _ = estimate_remaining_duration(snapshot, route, world, forecast_trace=trace)
    assert trace["port_congestion"]["forecast"] is selected
    assert selected.forecast_issued_at <= snapshot.checkpoint_at
    assert select_eligible_forecast(snapshot, "port_congestion", trace["port_congestion"]["requested_at"],
                                    trace["port_congestion"]["location_id"]) is selected

    after_issue = selected.model_copy(update={"forecast_issued_at": snapshot.checkpoint_at + timedelta(hours=1)})
    with pytest.raises(ValueError, match="forecast issued after checkpoint"):
        select_eligible_forecast(snapshot.model_copy(update={"forecasts": (after_issue,)}),
            "port_congestion", trace["port_congestion"]["requested_at"], selected.location_id)
    wrong_valid = selected.model_copy(update={"forecast_valid_at": selected.forecast_valid_at + timedelta(hours=2)})
    assert select_eligible_forecast(snapshot.model_copy(update={"forecasts": (wrong_valid,)}),
        "port_congestion", trace["port_congestion"]["requested_at"], selected.location_id) is None
    wrong_location = selected.model_copy(update={"location_id": "unrelated_location"})
    assert select_eligible_forecast(snapshot.model_copy(update={"forecasts": (wrong_location,)}),
        "port_congestion", trace["port_congestion"]["requested_at"], selected.location_id) is None

    neutralized = _neutralize_selected_forecast(snapshot, selected, .10)
    diff = [name for name in before if before[name] != neutralized.model_dump(mode="python")[name]]
    assert diff == ["forecasts"]
    assert neutralized.forecasts[1] == snapshot.forecasts[1]
    for field in ("observed_event_times", "observed_stage_fields", "current_environment", "scheduled_delivery_at"):
        assert getattr(neutralized, field) == getattr(snapshot, field)
    changed_values = [f.forecast_value for f in neutralized.forecasts]
    original_value = selected.forecast_value
    expected, _ = estimate_remaining_duration(neutralized, route, world)
    rows = _prediction_driver_attribution(None, [{"snapshot": snapshot, "shipment": shipment,
        "route": route, "world": world}], {"port_congestion": .10, "port_weather": 0.0})
    used = next(r for r in rows if r["component"] == "port_congestion")
    unsupported = next(r for r in rows if r["component"] == "port_weather")
    assert used["forecast_used"] is True
    assert used["forecast_issue_time"] == selected.forecast_issued_at
    assert used["forecast_valid_time"] == selected.forecast_valid_at
    assert used["forecast_location"] == selected.location_id
    assert used["original_forecast_value"] == original_value
    assert used["original_propagation_remaining"] == pytest.approx(max(0, original))
    assert used["neutralized_propagation_remaining"] == pytest.approx(max(0, expected))
    assert used["effect_hours"] == pytest.approx(max(0, original) - max(0, expected))
    assert abs(used["effect_hours"]) > 1e-9
    assert unsupported["forecast_used"] is False
    assert unsupported["unavailable_reason"] == "unsupported_not_consumed_by_propagation"
    assert snapshot.model_dump(mode="python") == before
    assert selected.forecast_value == original_value
    assert len(changed_values) == len(snapshot.forecasts)


def test_observation_missingness_rebuilds_ml_and_propagation_inputs_without_mutation():
    from aiqon.evaluation.phase3 import _masked_partition
    from aiqon.prediction.features import FEATURE_COLUMNS
    snapshot, shipment, route, world, _ = _selected_port_forecast_case()
    checkpoint = datetime(2025, 1, 1, 0, 42, tzinfo=timezone.utc)
    snapshot = snapshot.model_copy(update={
        "checkpoint_type": "port_arrival", "checkpoint_at": checkpoint,
        "observed_event_times": {"departure_at": checkpoint - timedelta(hours=1),
                                 "port_arrival_at": checkpoint},
        "observed_stage_fields": {"transit_duration_hours": 1.0, "observed_queue_length": 0.0},
        "current_environment": {"port_congestion": .99}, "forecasts": (),
        "feature_missingness": {},
    })
    route = route.model_copy(update={"berth_capacity": 1, "berth_slot_hours": tuple(range(24)),
        "congestion": route.congestion.model_copy(update={"baseline": .1}),
        "customs": route.customs.model_copy(update={"opening_hour_utc": 0, "closing_hour_utc": 23,
            "operating_days": ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")})})
    original_dump = snapshot.model_dump(mode="python")
    original_meta = {"shipment_id": shipment.shipment_id, "checkpoint_type": "port_arrival",
        "remaining_hours_actual": 22.0, "propagation_remaining": 0.0,
        "scheduled_remaining": 20.0, "carry_remaining": 21.0,
        "snapshot": snapshot, "shipment": shipment, "route": route, "world": world}
    original_features = pd.DataFrame([__import__("aiqon.prediction.features", fromlist=["build_features"])
        .build_features(snapshot, shipment, route, world)], columns=FEATURE_COLUMNS)
    before_propagation = original_meta["propagation_remaining"]
    from aiqon.prediction.propagation import estimate_remaining_duration
    original_propagation = estimate_remaining_duration(snapshot, route, world)[0]
    X_masked, masked_meta = _masked_partition([original_meta], ["observed_port_congestion"])
    assert pd.isna(X_masked.loc[0, "observed_port_congestion"])
    assert X_masked.loc[0, "observed_port_congestion__missing"] == 1
    assert X_masked.loc[0, "arrival_congestion_interaction__missing"] == 1
    assert masked_meta[0]["snapshot"].current_environment.get("port_congestion") is None
    assert masked_meta[0]["propagation_remaining"] != pytest.approx(max(0.0, original_propagation))
    assert original_features.loc[0, "observed_port_congestion"] == .99
    assert snapshot.model_dump(mode="python") == original_dump
    assert original_meta["snapshot"] is snapshot
    assert original_meta["propagation_remaining"] == before_propagation
    assert original_meta["remaining_hours_actual"] == 22.0


def test_forecast_missingness_removes_only_selected_component_from_propagation():
    from aiqon.evaluation.phase3 import _masked_partition
    from aiqon.prediction.propagation import estimate_remaining_duration
    snapshot, shipment, route, world, selected = _selected_port_forecast_case()
    env = dict(snapshot.current_environment)
    env.pop("port_congestion", None)
    snapshot = snapshot.model_copy(update={"current_environment": env})
    original_forecasts = snapshot.forecasts
    before = snapshot.model_dump(mode="python")
    meta = {"shipment_id": shipment.shipment_id, "checkpoint_type": snapshot.checkpoint_type,
        "remaining_hours_actual": 10.0, "propagation_remaining": 0.0, "scheduled_remaining": 0.0,
        "carry_remaining": 0.0, "snapshot": snapshot, "shipment": shipment,
        "route": route, "world": world}
    original = estimate_remaining_duration(snapshot, route, world)[0]
    X_masked, masked_meta = _masked_partition([meta], ["forecast_port_congestion"])
    assert X_masked.loc[0, "forecast_port_congestion__missing"] == 1
    assert pd.isna(X_masked.loc[0, "forecast_port_congestion"])
    assert all(f.forecast_variable != "port_congestion" for f in masked_meta[0]["snapshot"].forecasts)
    assert [f for f in masked_meta[0]["snapshot"].forecasts] == [f for f in original_forecasts
        if f.forecast_variable != "port_congestion"]
    masked = estimate_remaining_duration(masked_meta[0]["snapshot"], route, world)[0]
    assert masked != pytest.approx(original)
    assert selected.forecast_variable == "port_congestion"
    assert snapshot.model_dump(mode="python") == before
    assert meta["snapshot"] is snapshot and meta["remaining_hours_actual"] == 10.0


def test_retrospective_attribution_reruns_full_cohort_and_does_not_mutate_source(tmp_path, monkeypatch):
    import aiqon.evaluation.phase3 as phase3
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    shipment_frame = pd.DataFrame([{"shipment_id": "s", "route_id": "nhava_sheva_inland_a"}])
    shipment_path = raw / "shipments.csv"
    shipment_frame.to_csv(shipment_path, index=False)
    fri = datetime(2025, 1, 3, 20, tzinfo=timezone.utc)
    mon = datetime(2025, 1, 6, 8, tzinfo=timezone.utc)
    rows = [
        {"shipment_id": "s", "event_type": "departure", "actual_at": fri.isoformat()},
        {"shipment_id": "s", "event_type": "port_arrival", "actual_at": fri.isoformat()},
        {"shipment_id": "s", "event_type": "berth_start", "actual_at": fri.isoformat(),
         "slot_wait_hours": 2.0, "queue_wait_hours": 1.0},
        {"shipment_id": "s", "event_type": "port_complete", "actual_at": fri.isoformat()},
        {"shipment_id": "s", "event_type": "customs_start", "actual_at": mon.isoformat()},
        {"shipment_id": "s", "event_type": "customs_complete", "actual_at": mon.isoformat()},
        {"shipment_id": "s", "event_type": "gate_out", "actual_at": mon.isoformat(), "gate_wait_hours": 4.0},
        {"shipment_id": "s", "event_type": "delivery", "actual_at": (mon + timedelta(hours=4)).isoformat()},
    ]
    event_path = raw / "events.csv"
    pd.DataFrame(rows).to_csv(event_path, index=False)
    original_bytes = event_path.read_bytes()
    config = load_config("configs/simulation.yaml")
    routes = tuple(load_routes(config.routes.config_path).routes)
    calls = []
    def fake_simulate(config, routes, *, neutralize_mechanism=None, injected_transit_delay_hours=0.0):
        calls.append((config.data.random_seed, neutralize_mechanism, len(routes)))
        return SimpleNamespace(events=[SimpleNamespace(shipment_id="s", event_type="delivery",
            actual_at=(mon + timedelta(hours=4) if neutralize_mechanism is None else mon + timedelta(hours=2)) )])
    monkeypatch.setattr(phase3, "simulate_cohort", fake_simulate)
    result = phase3._retrospective_attribution(tmp_path, config, routes, "fingerprint", 42)
    assert result["label"] == "Retrospective cause attribution"
    assert {call[1] for call in calls} == {None, "berth_slot", "vessel_bunching", "weekend_customs", "gate_cutoff"}
    assert all(seed == 42 and route_count == 3 for seed, _, route_count in calls)
    assert all(row["retrospective_cause_effect_hours"] == 2 for row in result["results"])
    assert result["baseline_equivalence"]["status"] == "passed"
    assert result["baseline_equivalence"]["maximum_absolute_delta_seconds"] == 0
    assert event_path.read_bytes() == original_bytes


def test_retrospective_attribution_fails_before_intervention_on_baseline_mismatch(tmp_path, monkeypatch):
    import aiqon.evaluation.phase3 as phase3
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    pd.DataFrame([{"shipment_id": "s", "route_id": "nhava_sheva_inland_a"}]).to_csv(raw / "shipments.csv", index=False)
    fri = datetime(2025, 1, 3, 20, tzinfo=timezone.utc)
    mon = datetime(2025, 1, 6, 8, tzinfo=timezone.utc)
    rows = [
        {"shipment_id": "s", "event_type": "departure", "actual_at": fri.isoformat()},
        {"shipment_id": "s", "event_type": "port_arrival", "actual_at": fri.isoformat()},
        {"shipment_id": "s", "event_type": "berth_start", "actual_at": fri.isoformat(), "slot_wait_hours": 2.0},
        {"shipment_id": "s", "event_type": "port_complete", "actual_at": fri.isoformat()},
        {"shipment_id": "s", "event_type": "customs_start", "actual_at": mon.isoformat()},
        {"shipment_id": "s", "event_type": "customs_complete", "actual_at": mon.isoformat()},
        {"shipment_id": "s", "event_type": "gate_out", "actual_at": mon.isoformat()},
        {"shipment_id": "s", "event_type": "delivery", "actual_at": (mon + timedelta(hours=4)).isoformat()},
    ]
    pd.DataFrame(rows).to_csv(raw / "events.csv", index=False)
    config = load_config("configs/simulation.yaml")
    routes = tuple(load_routes(config.routes.config_path).routes)
    calls = []
    def mismatch(config, routes, *, neutralize_mechanism=None, injected_transit_delay_hours=0.0):
        calls.append(neutralize_mechanism)
        return SimpleNamespace(events=[SimpleNamespace(shipment_id="s", event_type="delivery", actual_at=mon)])
    monkeypatch.setattr(phase3, "simulate_cohort", mismatch)
    with pytest.raises(ValueError, match="Counterfactual baseline mismatch for s"):
        phase3._retrospective_attribution(tmp_path, config, routes, "fingerprint", 42)
    assert calls == [None]


def test_phase3_stress_scenarios_are_separate_and_weather_limit_is_explicit():
    from aiqon.evaluation.phase3 import _fit_models, _stress_test
    from aiqon.prediction.features import FEATURE_COLUMNS
    config = load_config("configs/simulation.yaml")
    routes = tuple(load_routes(config.routes.config_path).routes)
    # A deterministic tiny fit pipeline is sufficient to verify held-out stress harness isolation.
    X = pd.DataFrame({col: [0.0] * 18 for col in FEATURE_COLUMNS})
    for col in ("route_id", "shipment_type", "checkpoint_type", "origin_condition", "observed_gate_state"):
        X[col] = "fixture"
    meta = [{"remaining_hours_actual": float(i + 1), "propagation_remaining": 4.0} for i in range(18)]
    models, _ = _fit_models(X, meta, config.evaluation.phase3.model)
    first = _stress_test(".", config, routes, models, 4200, config.evaluation.phase3)
    second = _stress_test(".", config, routes, models, 4200, config.evaluation.phase3)
    assert first == second
    assert len(first["scenario_families"]) == 4
    assert len(first["prediction_rows"]) == 4 * config.evaluation.phase3.stress_shipment_count * 3
    assert first["unrepresentable_scenarios"][0]["scenario_family"] == "weather_disruption_plus_reduced_recovery_capacity"


def test_configured_phase3_hyperparameters_apply_to_both_training_paths():
    from aiqon.evaluation.phase3 import _fit_models, _fit_random_mask_model
    from aiqon.prediction.features import FEATURE_COLUMNS
    config = load_config("configs/simulation.yaml")
    model_config = config.evaluation.phase3.model.model_copy(update={
        "n_estimators": 7, "learning_rate": .13, "max_depth": 3,
        "min_samples_leaf": 2, "random_seed": 99})
    X = pd.DataFrame({column: [0.0, 1.0, 2.0, 3.0] for column in FEATURE_COLUMNS})
    for column in ("route_id", "shipment_type", "checkpoint_type", "origin_condition", "observed_gate_state"):
        X[column] = "fixture"
    meta = [{"remaining_hours_actual": float(i + 5), "propagation_remaining": 2.0} for i in range(4)]
    models, settings = _fit_models(X, meta, model_config)
    random_model = _fit_random_mask_model(X, np.arange(4, dtype=float), settings)
    expected = {"n_estimators": 7, "learning_rate": .13, "max_depth": 3,
                "min_samples_leaf": 2, "random_state": 99}
    assert settings["random_seed"] == 99
    for pipeline in (*models.values(), random_model):
        assert {name: pipeline.named_steps["model"].get_params()[name] for name in expected} == expected


def test_shap_explains_gradient_boosting_residual_output_only():
    import shap
    from sklearn.ensemble import GradientBoostingRegressor
    rng = np.random.default_rng(11)
    X = rng.normal(size=(24, 3))
    residual_target = X[:, 0] - 2 * X[:, 2]
    residual_model = GradientBoostingRegressor(n_estimators=10, max_depth=2, random_state=11).fit(X, residual_target)
    explainer = shap.TreeExplainer(residual_model)
    values = np.asarray(explainer.shap_values(X))
    reconstructed = float(np.asarray(explainer.expected_value).reshape(-1)[0]) + values.sum(axis=1)
    assert values.shape == X.shape
    assert np.allclose(reconstructed, residual_model.predict(X), atol=1e-6)

