from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aiqon.config.loader import load_config, load_routes
from aiqon.data_generation.environment import EnvironmentIndex, floor_hour
from aiqon.data_generation.generator import (
    build_summary, extract_journey, generate_dataset, regenerate_shipment,
)
from aiqon.simulation.engine import (
    controlled_propagation_examples,
    interval_intersects_weekend_closure,
    next_customs_start,
    next_gate_out,
    schedule_route_berths,
    simulate_cohort,
    validate_simulation_result,
)

ROOT = Path(__file__).resolve().parents[2]
UTC = timezone.utc


@pytest.fixture
def small_world():
    config = load_config(ROOT / "configs/simulation.yaml")
    routes = tuple(load_routes(ROOT / config.routes.config_path).routes)
    config = config.model_copy(update={
        "data": config.data.model_copy(update={"n_shipments": 12}),
        "world": config.world.model_copy(update={"departure_window_days": 6}),
    })
    return config, routes


def test_small_cohort_has_sequential_events_and_three_target_separated_snapshots(small_world):
    config, routes = small_world
    result = simulate_cohort(config, routes)
    validate_simulation_result(result)
    assert len(result.shipments) == 12
    assert len({shipment.shipment_id for shipment in result.shipments}) == 12
    assert len(result.events) == 12 * 8
    assert len({event.event_id for event in result.events}) == len(result.events)
    assert len(result.snapshots) == 12 * 3
    assert len(result.snapshot_targets) == 12 * 3
    target_fields = {"remaining_hours_actual", "actual_delay_hours", "significant_delay_label"}
    for snapshot in result.snapshots:
        assert not target_fields.intersection(snapshot.model_dump())
        trigger = {
            "departure": "departure_at",
            "port_arrival": "port_arrival_at",
            "customs_cleared": "customs_complete_at",
        }[snapshot.checkpoint_type]
        assert snapshot.observed_event_times[trigger] == snapshot.checkpoint_at
        assert all(f.forecast_issued_at <= snapshot.checkpoint_at for f in snapshot.forecasts)
    for shipment_id in {event.shipment_id for event in result.events}:
        journey = [event for event in result.events if event.shipment_id == shipment_id]
        assert [event.actual_at for event in journey] == sorted(event.actual_at for event in journey)
        assert all(event.actual_at.tzinfo is not None for event in journey)


def test_seeded_output_is_reproducible_and_master_seed_changes_draws(small_world):
    config, routes = small_world
    first = simulate_cohort(config, routes)
    second = simulate_cohort(config, routes)
    assert [s.model_dump() for s in first.shipments] == [s.model_dump() for s in second.shipments]
    assert first.event_rows == second.event_rows
    assert first.environment_rows == second.environment_rows
    assert first.forecast_rows == second.forecast_rows
    different_config = config.model_copy(update={
        "data": config.data.model_copy(update={"random_seed": config.data.random_seed + 1})
    })
    changed = simulate_cohort(different_config, routes)
    assert [s.simulation_seed for s in first.shipments] != [s.simulation_seed for s in changed.shipments]
    assert first.environment_rows != changed.environment_rows
    original_journey = extract_journey(first, "SHP-0001")
    regenerated_journey = regenerate_shipment("SHP-0001", config)
    assert regenerated_journey == original_journey


def test_environment_lookup_uses_hour_of_actual_event(small_world):
    config, routes = small_world
    result = simulate_cohort(config, routes)
    index = EnvironmentIndex(result.environment_rows)
    shipments = {shipment.shipment_id: shipment for shipment in result.shipments}
    arrivals = {event.shipment_id: event for event in result.events if event.event_type == "port_arrival"}
    snapshots = {(snapshot.shipment_id, snapshot.checkpoint_type): snapshot for snapshot in result.snapshots}
    checked_shift = False
    for shipment_id, event in arrivals.items():
        shipment = shipments[shipment_id]
        actual_hour = floor_hour(event.actual_at)
        scheduled_hour = floor_hour(shipment.scheduled_port_arrival_at)
        expected = index.at(shipment.destination_port, event.actual_at)
        snapshot = snapshots[(shipment_id, "port_arrival")]
        assert snapshot.current_environment["port_congestion"] == expected["congestion_actual"]
        if actual_hour != scheduled_hour:
            checked_shift = True
            assert actual_hour == floor_hour(event.actual_at)
            break
    assert checked_shift, "small deterministic fixture should include transit timing shifts"


def test_route_global_berth_scheduler_accounts_for_busy_berth_and_slot():
    day = datetime(2025, 1, 1, tzinfo=UTC)
    assignments = schedule_route_berths(
        [("A", day), ("B", day + timedelta(hours=1))],
        berth_capacity=1,
        slot_hours=(0, 12),
        service_durations={"A": 18.0, "B": 1.0},
    )
    assert assignments["A"].berth_start_at == day
    assert assignments["B"].berth_start_at == day + timedelta(days=1)
    assert assignments["B"].slot_wait_hours == pytest.approx(11.0)
    assert assignments["B"].queue_wait_hours == pytest.approx(12.0)
    assert assignments["B"].berth_id == assignments["A"].berth_id


def test_customs_weekly_closure_and_gate_cutoff_defer_operations(small_world):
    _, routes = small_world
    route = routes[0]  # Mon-Fri customs profile
    saturday = datetime(2025, 1, 4, 10, tzinfo=UTC)
    assert next_customs_start(saturday, route) == datetime(2025, 1, 6, 8, tzinfo=UTC)
    late = datetime(2025, 1, 1, 17, tzinfo=UTC)
    assert next_gate_out(late, 16) == datetime(2025, 1, 2, 16, tzinfo=UTC)
    before_cutoff = datetime(2025, 1, 1, 15, tzinfo=UTC)
    assert next_gate_out(before_cutoff, 16) == before_cutoff


def test_weekend_customs_closure_attribution_uses_wait_interval(small_world):
    _, routes = small_world
    weekday_only = routes[0]
    saturday_open = routes[1]

    friday_evening = datetime(2025, 1, 3, 20, tzinfo=UTC)
    monday_open = datetime(2025, 1, 6, 8, tzinfo=UTC)
    assert interval_intersects_weekend_closure(friday_evening, monday_open, weekday_only)

    saturday_closed = datetime(2025, 1, 4, 10, tzinfo=UTC)
    assert interval_intersects_weekend_closure(saturday_closed, monday_open, weekday_only)

    tuesday_after_close = datetime(2025, 1, 7, 20, tzinfo=UTC)
    wednesday_open = datetime(2025, 1, 8, 8, tzinfo=UTC)
    assert not interval_intersects_weekend_closure(tuesday_after_close, wednesday_open, weekday_only)

    saturday_open_time = datetime(2025, 1, 4, 10, tzinfo=UTC)
    same_day_start = next_customs_start(saturday_open_time, saturday_open)
    assert same_day_start == saturday_open_time
    assert not interval_intersects_weekend_closure(saturday_open_time, same_day_start, saturday_open)

    saturday_after_close = datetime(2025, 1, 4, 20, tzinfo=UTC)
    monday_mundra_open = next_customs_start(saturday_after_close, saturday_open)
    assert monday_mundra_open == datetime(2025, 1, 6, 7, tzinfo=UTC)
    assert interval_intersects_weekend_closure(saturday_after_close, monday_mundra_open, saturday_open)


def test_seeded_service_multipliers_and_controlled_amplification_examples(small_world):
    config, routes = small_world
    result = simulate_cohort(config, routes)
    multipliers = [row["berth_service_multiplier"] for row in result.journey_metrics]
    assert all(0.85 <= value <= 1.15 for value in multipliers)
    assert len(set(multipliers)) > 1
    control_route = next(route for route in routes if route.berth_capacity == 1 and 0 in route.berth_slot_hours)
    examples = controlled_propagation_examples(control_route, config.world)
    assert examples["busy_window"]["amplification_factor"] > 1
    assert examples["quiet_window"]["amplification_factor"] <= 1


def test_summary_reports_required_class_balance_and_mechanism_diagnostics(small_world):
    config, routes = small_world
    result = simulate_cohort(config, routes)
    summary = build_summary(result, config, routes)
    assert summary["counts"]["unique_shipments"] == 12
    assert summary["counts"]["checkpoint_snapshots"] == 36
    assert 0 <= summary["counts"]["significant_delay_prevalence"] <= 1
    assert sum(summary["delay_class_counts_hours"].values()) == 12
    assert set(summary["mechanism_affected_journeys"]) >= {
        "berth_slot_wait_affected", "vessel_bunching_or_berth_queue",
        "weekend_customs_closure", "inland_gate_cutoff_miss",
    }
    assert summary["controlled_propagation_examples"]["busy_window"]["amplification_factor"] > 1


def test_small_dataset_exports_consistently_and_summary_is_machine_readable(small_world, tmp_path):
    config, _ = small_world
    generated = generate_dataset(config, config.routes.config_path, tmp_path)
    summary = generated["summary"]
    assert summary["counts"]["unique_shipments"] == 12
    assert summary["counts"]["checkpoint_snapshots"] == 36
    assert summary["storage_format"] == "parquet"
    for path in summary["output_files"].values():
        assert (tmp_path / path).is_file()
    assert (tmp_path / summary["summary_path"]).is_file()


def test_customs_snapshot_omits_unmeasured_queue_and_documents_constant_truck_availability(small_world):
    config, routes = small_world
    result = simulate_cohort(config, routes)
    port_snapshots = [s for s in result.snapshots if s.checkpoint_type == "port_arrival"]
    customs_snapshots = [s for s in result.snapshots if s.checkpoint_type == "customs_cleared"]
    assert all(s.current_environment["queue_state"] == s.observed_stage_fields["observed_queue_length"]
               for s in port_snapshots)
    assert all("queue_state" not in s.current_environment for s in customs_snapshots)
    assert all(s.observed_stage_fields["truck_availability"] == 1.0 for s in customs_snapshots)


def test_regenerated_journey_contains_all_requested_records(small_world):
    config, routes = small_world
    complete = simulate_cohort(config, routes)
    journey = extract_journey(complete, "SHP-0001")
    assert journey.shipment.shipment_id == "SHP-0001"
    assert len(journey.events) == 8
    assert len(journey.snapshots) == 3
    assert len(journey.snapshot_targets) == 3
    assert journey.forecast_records
    assert journey.environment_records
    assert journey.journey_metrics["shipment_id"] == "SHP-0001"

