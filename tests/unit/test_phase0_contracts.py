from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from aiqon.checkpoints import (CHECKPOINT_DEFINITIONS, CHECKPOINTS, FEATURE_ALLOWLISTS,
                               validate_forecast_issue_time, validate_snapshot_features)
from aiqon.config.loader import RouteProfile, load_config, load_routes
from aiqon.ground_truth import final_delay_hours, is_significant_delay, remaining_duration_hours
from aiqon.schemas.events import ActualEnvironment, ActualEvent
from aiqon.schemas.predictions import PredictionOutput
from aiqon.schemas.shipment import Shipment
from aiqon.schemas.snapshots import CheckpointSnapshot, Forecast

ROOT = Path(__file__).resolve().parents[2]
UTC = timezone.utc


def test_default_configuration_and_routes_load_and_validate():
    config = load_config(ROOT / "configs/simulation.yaml")
    routes = load_routes(ROOT / config.routes.config_path)
    assert config.data.n_shipments == 300
    assert config.data.checkpoint_names == CHECKPOINTS
    assert len(routes.routes) == 3
    assert [(route.route_id, route.scheduled_transit_hours, route.berth_capacity,
             route.berth_slot_hours, route.congestion.baseline, route.congestion.volatility,
             route.customs.opening_hour_utc, route.customs.closing_hour_utc,
             route.customs.base_processing_hours, route.inland.travel_hours,
             route.inland.gate_cutoff_hour_utc) for route in routes.routes] == [
        ("nhava_sheva_inland_a", 96, 2, (0, 12), 0.45, 0.18, 8, 18, 8, 18, 16),
        ("mundra_inland_b", 84, 3, (0, 8, 16), 0.35, 0.15, 7, 19, 7, 22, 17),
        ("synthetic_corridor_c", 108, 1, (0, 12), 0.55, 0.22, 9, 17, 10, 26, 15),
    ]


def test_invalid_configuration_has_clear_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("project: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid simulation configuration"):
        load_config(path)


def test_invalid_timezone_is_rejected(tmp_path):
    path = tmp_path / "bad_timezone.yaml"
    path.write_text("""project:\n  name: sample\n  timezone: Not/A_Real_Zone\ndata:\n  n_shipments: 3\n  random_seed: 42\n  checkpoint_names: [departure, port_arrival, customs_cleared]\nprediction:\n  significant_delay_threshold_hours: 12\n  conformal_coverage: 0.9\nevaluation:\n  n_splits: 5\n  n_repeats: 3\n  bootstrap_samples: 1000\n  primary_metric: mae\nroutes:\n  config_path: configs/routes.yaml\n""", encoding="utf-8")
    with pytest.raises(ValueError, match="valid IANA timezone"):
        load_config(path)


def test_shipment_schema_and_schedule_order():
    base = datetime(2026, 1, 1, tzinfo=UTC)
    s = Shipment(shipment_id="S1", route_id="R1", origin="O", destination_port="P",
                 final_destination="D", shipment_type="container", volume=1,
                 scheduled_departure_at=base, scheduled_port_arrival_at=base+timedelta(hours=2),
                 scheduled_customs_clearance_at=base+timedelta(hours=3),
                 scheduled_delivery_at=base+timedelta(hours=4), simulation_seed=42, scenario_id="normal")
    assert s.scheduled_departure_at.utcoffset() == timedelta(0)
    with pytest.raises(ValidationError, match="chronological"):
        Shipment(**{**s.model_dump(), "scheduled_port_arrival_at": base-timedelta(hours=1)})


def test_event_environment_forecast_snapshot_and_prediction_schemas():
    t = datetime(2026, 1, 1, tzinfo=UTC)
    event = ActualEvent(shipment_id="S1", event_id="E1", event_type="departure", actual_at=t,
                        stage="origin", observable_after="departure")
    env = ActualEnvironment(location_id="O", valid_at=t, congestion_actual=0.2)
    forecast = Forecast(location_id="P", forecast_issued_at=t, forecast_valid_at=t+timedelta(hours=1),
                        forecast_variable="port_weather", forecast_value=0.1)
    snapshot = CheckpointSnapshot(shipment_id="S1", checkpoint_type="departure", checkpoint_at=t,
                                  scheduled_delivery_at=t+timedelta(days=1), route_id="R1",
                                  observed_event_times={"departure_at": t}, forecasts=(forecast,))
    prediction = PredictionOutput(shipment_id="S1", checkpoint_type="departure", prediction_created_at=t,
        predicted_remaining_hours=24, predicted_delivery_at=t+timedelta(days=1), predicted_total_delay_hours=0,
        probability_delay_over_12h=0.2, remaining_hours_lower=20, remaining_hours_upper=28,
        model_version="contract-only", feature_schema_version="v1")
    assert event.event_type == "departure" and env.location_id == "O"
    assert snapshot.shipment_id == prediction.shipment_id


@pytest.mark.parametrize("event_type,visibility", [
    ("departure", "departure"),
    ("port_arrival", "port_arrival"),
    ("berth_start", "customs_cleared"),
    ("port_complete", "customs_cleared"),
    ("customs_start", "customs_cleared"),
    ("customs_complete", "customs_cleared"),
    ("gate_out", "ground_truth_only"),
    ("delivery", "ground_truth_only"),
])
def test_actual_event_visibility_contract_accepts_valid_combinations(event_type, visibility):
    ActualEvent(shipment_id="S1", event_id="E1", event_type=event_type,
        actual_at=datetime(2026, 1, 1, tzinfo=UTC), stage="stage", observable_after=visibility)


@pytest.mark.parametrize("event_type,visibility", [
    ("departure", "port_arrival"),
    ("port_arrival", "departure"),
    ("customs_complete", "port_arrival"),
    ("berth_start", "port_arrival"),
    ("delivery", "departure"),
    ("gate_out", "customs_cleared"),
    ("delivery", "customs_cleared"),
])
def test_actual_event_visibility_contract_rejects_invalid_combinations(event_type, visibility):
    with pytest.raises(ValidationError, match="must have observable_after"):
        ActualEvent(shipment_id="S1", event_id="E1", event_type=event_type,
            actual_at=datetime(2026, 1, 1, tzinfo=UTC), stage="stage", observable_after=visibility)


def test_timestamps_must_be_timezone_aware():
    naive = datetime(2026, 1, 1)
    with pytest.raises(ValidationError, match="timezone-aware"):
        Forecast(location_id="P", forecast_issued_at=naive, forecast_valid_at=naive,
                 forecast_variable="weather", forecast_value=0.1)


def test_forecast_valid_time_must_follow_issue_time():
    issued = datetime(2026, 1, 1, tzinfo=UTC)
    valid_forecast = Forecast(location_id="P", forecast_issued_at=issued,
        forecast_valid_at=issued+timedelta(seconds=1), forecast_variable="port_weather", forecast_value=0.2)
    non_utc = timezone(timedelta(hours=5, minutes=30))
    normalized = Forecast(location_id="P",
        forecast_issued_at=datetime(2026, 1, 1, tzinfo=non_utc),
        forecast_valid_at=datetime(2026, 1, 1, 1, tzinfo=non_utc),
        forecast_variable="port_weather", forecast_value=0.2)
    assert valid_forecast.forecast_valid_at > valid_forecast.forecast_issued_at
    assert normalized.forecast_issued_at.utcoffset() == timedelta(0)
    assert normalized.forecast_valid_at.utcoffset() == timedelta(0)
    for valid_at in (issued, issued-timedelta(seconds=1)):
        with pytest.raises(ValidationError, match="forecast_valid_at must be later"):
            Forecast(location_id="P", forecast_issued_at=issued, forecast_valid_at=valid_at,
                forecast_variable="port_weather", forecast_value=0.2)


def test_ground_truth_calculations_and_significance_boundary():
    scheduled = datetime(2026, 1, 1, tzinfo=UTC)
    actual = scheduled + timedelta(hours=12, minutes=1)
    checkpoint = scheduled + timedelta(hours=2)
    assert final_delay_hours(actual, scheduled) == pytest.approx(12 + 1/60)
    assert remaining_duration_hours(actual, checkpoint) == pytest.approx(10 + 1/60)
    assert is_significant_delay(actual, scheduled)
    assert not is_significant_delay(scheduled + timedelta(hours=12), scheduled)


def test_ground_truth_rejects_naive_timestamp():
    with pytest.raises(ValueError, match="timezone-aware"):
        final_delay_hours(datetime(2026, 1, 2), datetime(2026, 1, 1, tzinfo=UTC))


def test_three_checkpoint_definitions_are_unique_and_allowlisted():
    assert tuple(d.name for d in CHECKPOINT_DEFINITIONS) == CHECKPOINTS
    assert len({d.name for d in CHECKPOINT_DEFINITIONS}) == 3
    assert set(FEATURE_ALLOWLISTS) == set(CHECKPOINTS)
    for checkpoint in CHECKPOINTS:
        validate_snapshot_features(checkpoint, {"shipment_id", "checkpoint_at"})


@pytest.mark.parametrize("field", ["actual_delivery_at", "actual_customs_complete_at",
                                    "actual_final_delivery_at", "actual_delay_hours"])
def test_forbidden_future_and_target_fields_are_rejected(field):
    with pytest.raises(ValueError, match="Forbidden ground-truth"):
        validate_snapshot_features("departure", {field})


def test_nonallowlisted_field_is_rejected():
    with pytest.raises(ValueError, match="not allowed"):
        validate_snapshot_features("departure", {"unreviewed_future_field"})


def test_forecast_issue_time_validation():
    checkpoint = datetime(2026, 1, 1, tzinfo=UTC)
    validate_forecast_issue_time(checkpoint, checkpoint)
    with pytest.raises(ValueError, match="after the checkpoint"):
        validate_forecast_issue_time(checkpoint+timedelta(seconds=1), checkpoint)


def test_snapshot_rejects_late_forecast_vintage():
    t = datetime(2026, 1, 1, tzinfo=UTC)
    late = Forecast(location_id="P", forecast_issued_at=t+timedelta(minutes=1),
                    forecast_valid_at=t+timedelta(hours=2), forecast_variable="port_weather", forecast_value=0.2)
    with pytest.raises(ValidationError, match="after the checkpoint"):
        CheckpointSnapshot(shipment_id="S1", checkpoint_type="departure", checkpoint_at=t,
                           scheduled_delivery_at=t+timedelta(days=2), route_id="R1",
                           observed_event_times={"departure_at": t}, forecasts=(late,))


def test_snapshot_rejects_future_actual_hidden_in_nested_mapping():
    t = datetime(2026, 1, 1, tzinfo=UTC)
    with pytest.raises(ValidationError, match="not allowed in observed_event_times"):
        CheckpointSnapshot(shipment_id="S1", checkpoint_type="departure", checkpoint_at=t,
            scheduled_delivery_at=t+timedelta(days=2), route_id="R1",
            observed_event_times={"actual_delivery_at": t+timedelta(days=3)})


def test_snapshot_rejects_leaking_missingness_flag():
    t = datetime(2026, 1, 1, tzinfo=UTC)
    with pytest.raises(ValidationError, match="Missingness flag"):
        CheckpointSnapshot(shipment_id="S1", checkpoint_type="departure", checkpoint_at=t,
            scheduled_delivery_at=t+timedelta(days=2), route_id="R1",
            observed_event_times={"departure_at": t},
            feature_missingness={"actual_delivery_at": True})


@pytest.mark.parametrize("checkpoint,trigger", [
    ("departure", "departure_at"),
    ("port_arrival", "port_arrival_at"),
    ("customs_cleared", "customs_complete_at"),
])
def test_each_checkpoint_requires_its_trigger_at_checkpoint_time(checkpoint, trigger):
    t = datetime(2026, 1, 1, tzinfo=UTC)
    with pytest.raises(ValidationError, match="requires triggering event"):
        CheckpointSnapshot(shipment_id="S1", checkpoint_type=checkpoint, checkpoint_at=t,
            scheduled_delivery_at=t+timedelta(days=2), route_id="R1")

    with pytest.raises(ValidationError, match="must equal checkpoint_at"):
        CheckpointSnapshot(shipment_id="S1", checkpoint_type=checkpoint, checkpoint_at=t,
            scheduled_delivery_at=t+timedelta(days=2), route_id="R1",
            observed_event_times={trigger: t-timedelta(minutes=1)})


@pytest.mark.parametrize("checkpoint,trigger,earlier", [
    ("departure", "departure_at", {}),
    ("port_arrival", "port_arrival_at", {"departure_at": -timedelta(hours=1)}),
    ("customs_cleared", "customs_complete_at", {
        "departure_at": -timedelta(hours=3), "port_arrival_at": -timedelta(hours=2)}),
])
def test_each_checkpoint_accepts_trigger_at_checkpoint_and_prior_events(checkpoint, trigger, earlier):
    t = datetime(2026, 1, 1, tzinfo=UTC)
    events = {name: t + offset for name, offset in earlier.items()}
    events[trigger] = t
    CheckpointSnapshot(shipment_id="S1", checkpoint_type=checkpoint, checkpoint_at=t,
        scheduled_delivery_at=t+timedelta(days=2), route_id="R1", observed_event_times=events)


@pytest.mark.parametrize("checkpoint,trigger", [
    ("departure", "departure_at"),
    ("port_arrival", "port_arrival_at"),
    ("customs_cleared", "customs_complete_at"),
])
def test_each_checkpoint_rejects_any_observed_timestamp_after_checkpoint(checkpoint, trigger):
    t = datetime(2026, 1, 1, tzinfo=UTC)
    with pytest.raises(ValidationError, match="later than checkpoint_at"):
        CheckpointSnapshot(shipment_id="S1", checkpoint_type=checkpoint, checkpoint_at=t,
            scheduled_delivery_at=t+timedelta(days=2), route_id="R1",
            observed_event_times={trigger: t, "departure_at": t+timedelta(seconds=1)})


@pytest.mark.parametrize("future_event", ["gate_out_at", "delivery_at"])
def test_post_customs_events_cannot_become_predictor_visible(future_event):
    t = datetime(2026, 1, 1, tzinfo=UTC)
    with pytest.raises(ValidationError, match="not allowed in observed_event_times"):
        CheckpointSnapshot(shipment_id="S1", checkpoint_type="customs_cleared", checkpoint_at=t,
            scheduled_delivery_at=t+timedelta(days=2), route_id="R1",
            observed_event_times={"departure_at": t-timedelta(hours=3),
                "port_arrival_at": t-timedelta(hours=2), "customs_complete_at": t,
                future_event: t+timedelta(hours=1)})


def test_snapshot_rejects_naive_nested_event_timestamp_as_validation_error():
    t = datetime(2026, 1, 1, tzinfo=UTC)
    with pytest.raises(ValidationError, match="timezone-aware"):
        CheckpointSnapshot(shipment_id="S1", checkpoint_type="departure", checkpoint_at=t,
            scheduled_delivery_at=t+timedelta(days=2), route_id="R1",
            observed_event_times={"departure_at": datetime(2026, 1, 1)})


def test_snapshot_normalizes_nested_event_timestamp_to_utc():
    t = datetime(2026, 1, 1, tzinfo=UTC)
    non_utc = timezone(timedelta(hours=5, minutes=30))
    snapshot = CheckpointSnapshot(shipment_id="S1", checkpoint_type="departure", checkpoint_at=t,
        scheduled_delivery_at=t+timedelta(days=2), route_id="R1",
        observed_event_times={"departure_at": datetime(2026, 1, 1, 5, 30, tzinfo=non_utc)})
    assert snapshot.observed_event_times["departure_at"] == t
    assert snapshot.observed_event_times["departure_at"].utcoffset() == timedelta(0)


@pytest.mark.parametrize("bad_route,match", [
    ({"berth_slot_hours": []}, "at least 1 item"),
    ({"berth_slot_hours": [8, 8]}, "must be unique"),
    ({"berth_slot_hours": [24]}, "between 0 and 23"),
    ({"congestion": {"baseline": 1.1, "volatility": 0.2}}, "less than or equal to 1"),
    ({"congestion": {"baseline": 0.5, "volatility": -0.1}}, "greater than or equal to 0"),
    ({"customs": {"opening_hour_utc": 18, "closing_hour_utc": 18}}, "earlier than"),
    ({"customs": {"opening_hour_utc": 24}}, "less than or equal to 23"),
    ({"customs": {"closing_hour_utc": 25}}, "less than or equal to 24"),
    ({"customs": {"operating_days": []}}, "must not be empty"),
    ({"inland": {"travel_hours": 0}}, "greater than 0"),
])
def test_invalid_route_profile_values_are_rejected(bad_route, match):
    original = load_routes(ROOT / "configs/routes.yaml").routes[0].model_dump()
    for key, value in bad_route.items():
        if isinstance(value, dict) and isinstance(original.get(key), dict):
            original[key].update(value)
        else:
            original[key] = value
    with pytest.raises(ValidationError, match=match):
        RouteProfile.model_validate(original)


def test_prediction_output_rejects_invalid_interval_and_negative_remaining():
    t = datetime(2026, 1, 1, tzinfo=UTC)
    common = dict(shipment_id="S1", checkpoint_type="departure", prediction_created_at=t,
        predicted_remaining_hours=24, predicted_delivery_at=t+timedelta(days=1),
        predicted_total_delay_hours=0, probability_delay_over_12h=0.5,
        remaining_hours_lower=20, remaining_hours_upper=28, model_version="m1",
        feature_schema_version="v1")
    with pytest.raises(ValidationError, match="lower bound exceeds upper bound"):
        PredictionOutput(**{**common, "remaining_hours_lower": 30})
    with pytest.raises(ValidationError, match="predicted_remaining_hours"):
        PredictionOutput(**{**common, "predicted_remaining_hours": -1})
    with pytest.raises(ValidationError, match="must fall within its interval"):
        PredictionOutput(**{**common, "remaining_hours_lower": 25})


def test_prediction_delivery_timestamp_must_match_created_at_plus_remaining():
    t = datetime(2026, 1, 1, tzinfo=UTC)
    common = dict(shipment_id="S1", checkpoint_type="departure", prediction_created_at=t,
        predicted_remaining_hours=24, predicted_delivery_at=t+timedelta(days=1),
        predicted_total_delay_hours=0, probability_delay_over_12h=0.5,
        remaining_hours_lower=20, remaining_hours_upper=28, model_version="m1",
        feature_schema_version="v1")
    PredictionOutput(**{**common, "predicted_delivery_at": t+timedelta(days=1, milliseconds=500)})
    with pytest.raises(ValidationError, match="within 1 second"):
        PredictionOutput(**{**common, "predicted_delivery_at": t+timedelta(days=1, seconds=2)})

