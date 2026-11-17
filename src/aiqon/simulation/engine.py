"""Sequential shipment journey simulation and route-global berth scheduling."""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Iterable

from aiqon.config.loader import RouteProfile, SimulationConfig
from aiqon.data_generation.environment import EnvironmentIndex, floor_hour, generate_environment
from aiqon.data_generation.forecasts import generate_checkpoint_forecasts
from aiqon.data_generation.randomness import derive_seed, rng_for
from aiqon.schemas.events import ActualEvent, CheckpointName
from aiqon.schemas.shipment import Shipment
from aiqon.schemas.snapshots import CheckpointSnapshot
from aiqon.ground_truth import final_delay_hours, is_significant_delay, remaining_duration_hours

UTC = timezone.utc
HOURS_PER_DAY = 24
EVENT_VISIBILITY = {
    "departure": "departure",
    "port_arrival": "port_arrival",
    "berth_start": "customs_cleared",
    "port_complete": "customs_cleared",
    "customs_start": "customs_cleared",
    "customs_complete": "customs_cleared",
    "gate_out": "ground_truth_only",
    "delivery": "ground_truth_only",
}


def hours_between(start: datetime, end: datetime) -> float:
    return (end - start).total_seconds() / 3600.0


def next_berth_slot(arrival_at: datetime, slot_hours: Iterable[int]) -> datetime:
    """Slots are exact UTC hours each day; arriving after one waits for the next."""
    slots = tuple(sorted(slot_hours))
    if not slots:
        raise ValueError("At least one berth slot is required")
    day = arrival_at.replace(hour=0, minute=0, second=0, microsecond=0)
    for day_offset in range(3):
        candidate_day = day + timedelta(days=day_offset)
        for hour in slots:
            candidate = candidate_day + timedelta(hours=hour)
            if candidate >= arrival_at:
                return candidate
    raise RuntimeError("Could not find a berth slot within the search horizon")


@dataclass(frozen=True)
class BerthAssignment:
    shipment_id: str
    berth_id: int
    arrival_at: datetime
    berth_start_at: datetime
    berth_complete_at: datetime
    slot_wait_hours: float
    queue_wait_hours: float
    queue_length_at_arrival: int
    available_berths_at_arrival: int
    slot_at_or_after_arrival: datetime
    service_duration_hours: float


def schedule_route_berths(
    arrivals: Iterable[tuple[str, datetime]],
    berth_capacity: int,
    slot_hours: Iterable[int],
    service_durations: dict[str, float],
    *, ignore_slot_windows: bool = False,
) -> dict[str, BerthAssignment]:
    """Schedule route vessels globally by actual arrival and deterministic ID tie-break."""
    ordered = sorted(arrivals, key=lambda item: (item[1], item[0]))
    if not ordered or berth_capacity < 1:
        return {}
    first_arrival = ordered[0][1]
    heap = [(first_arrival - timedelta(days=1), berth_id) for berth_id in range(berth_capacity)]
    heapq.heapify(heap)
    prior: list[BerthAssignment] = []
    assignments: dict[str, BerthAssignment] = {}
    for shipment_id, arrival_at in ordered:
        slot_at_or_after_arrival = arrival_at if ignore_slot_windows else next_berth_slot(arrival_at, slot_hours)
        candidates = [
            (max(arrival_at, free_at) if ignore_slot_windows else
             next_berth_slot(max(arrival_at, free_at), slot_hours), berth_id, free_at)
            for free_at, berth_id in heap
        ]
        start_at, berth_id, selected_free = min(candidates, key=lambda item: (item[0], item[1]))
        slot_wait = 0.0 if ignore_slot_windows else max(0.0, hours_between(arrival_at, slot_at_or_after_arrival))
        total_wait = max(0.0, hours_between(arrival_at, start_at))
        queue_wait = max(0.0, total_wait - slot_wait)
        service_duration = float(service_durations[shipment_id])
        complete_at = start_at + timedelta(hours=service_duration)
        queue_length = sum(1 for item in prior if item.berth_start_at > arrival_at)
        available = sum(1 for free_at, _ in heap if free_at <= arrival_at)
        assignment = BerthAssignment(
            shipment_id=shipment_id,
            berth_id=berth_id,
            arrival_at=arrival_at,
            berth_start_at=start_at,
            berth_complete_at=complete_at,
            slot_wait_hours=slot_wait,
            queue_wait_hours=queue_wait,
            queue_length_at_arrival=queue_length,
            available_berths_at_arrival=available,
            slot_at_or_after_arrival=slot_at_or_after_arrival,
            service_duration_hours=service_duration,
        )
        prior.append(assignment)
        assignments[shipment_id] = assignment
        heap.remove((selected_free, berth_id))
        heapq.heapify(heap)
        heapq.heappush(heap, (complete_at, berth_id))
    return assignments


def next_customs_start(eligible_at: datetime, route: RouteProfile) -> datetime:
    """Return the next start instant inside the configured weekly operating window."""
    operating_days = {day.lower() for day in route.customs.operating_days}
    day_start = eligible_at.replace(hour=0, minute=0, second=0, microsecond=0)
    opening = route.customs.opening_hour_utc
    closing = route.customs.closing_hour_utc
    for offset in range(15):
        candidate_day = day_start + timedelta(days=offset)
        weekday = candidate_day.strftime("%A").lower()
        if weekday not in operating_days:
            continue
        opens_at = candidate_day + timedelta(hours=opening)
        closes_at = candidate_day + timedelta(hours=closing)
        candidate = max(eligible_at, opens_at)
        if candidate < closes_at:
            return candidate
    raise RuntimeError("No customs operating window found within two weeks")


def interval_intersects_weekend_closure(
    port_complete_at: datetime, customs_start_at: datetime, route: RouteProfile
) -> bool:
    """Whether the actual customs wait overlaps a closed Saturday or Sunday."""
    if customs_start_at <= port_complete_at:
        return False
    operating_days = {day.lower() for day in route.customs.operating_days}
    day = port_complete_at.replace(hour=0, minute=0, second=0, microsecond=0)
    while day < customs_start_at:
        next_day = day + timedelta(days=1)
        if day.weekday() >= 5 and day.strftime("%A").lower() not in operating_days:
            if max(port_complete_at, day) < min(customs_start_at, next_day):
                return True
        day = next_day
    return False


def next_gate_out(customs_complete_at: datetime, cutoff_hour: int) -> datetime:
    """Cut-off is the day's last dispatch time; late-ready freight waits to next day's cut-off."""
    cutoff = customs_complete_at.replace(hour=cutoff_hour, minute=0, second=0, microsecond=0)
    if customs_complete_at <= cutoff:
        return customs_complete_at
    return cutoff + timedelta(days=1)


@dataclass
class SimulationResult:
    shipments: list[Shipment]
    events: list[ActualEvent]
    event_rows: list[dict]
    environment_rows: list[dict]
    forecast_rows: list[dict]
    snapshots: list[CheckpointSnapshot]
    snapshot_targets: list[dict]
    journey_metrics: list[dict]
    forecast_diagnostics: list[dict]


def validate_simulation_result(result: SimulationResult) -> None:
    """Fail before export if event, target, duration, or checkpoint contracts drift."""
    shipment_ids = [shipment.shipment_id for shipment in result.shipments]
    if len(shipment_ids) != len(set(shipment_ids)):
        raise ValueError("Shipment IDs must be unique")
    event_ids = [event.event_id for event in result.events]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("Event IDs must be unique")
    by_shipment: dict[str, list[ActualEvent]] = defaultdict(list)
    for event in result.events:
        by_shipment[event.shipment_id].append(event)
    for shipment_id in shipment_ids:
        journey = by_shipment.get(shipment_id, [])
        if len(journey) != 8:
            raise ValueError(f"{shipment_id} must have exactly 8 actual events")
        if [event.actual_at for event in journey] != sorted(event.actual_at for event in journey):
            raise ValueError(f"Actual events are not chronological for {shipment_id}")
        if any(event.actual_at.tzinfo is None or event.actual_at.utcoffset() is None for event in journey):
            raise ValueError(f"Actual event timestamps must be aware for {shipment_id}")
    nonnegative_metrics = (
        "transit_duration_hours", "berth_slot_wait_hours", "berth_queue_wait_hours",
        "berth_service_duration_hours", "customs_closure_wait_hours",
        "customs_processing_duration_hours", "gate_wait_hours", "inland_travel_duration_hours",
    )
    for metric in result.journey_metrics:
        for field in nonnegative_metrics:
            if metric[field] < 0:
                raise ValueError(f"{field} must be nonnegative for {metric['shipment_id']}")
        if not 0.85 <= metric["berth_service_multiplier"] <= 1.15:
            raise ValueError(f"Berth service multiplier out of bounds for {metric['shipment_id']}")
    if len(result.snapshots) != 3 * len(result.shipments):
        raise ValueError("Exactly three checkpoint snapshots are required per shipment")
    snapshot_keys = [(snapshot.shipment_id, snapshot.checkpoint_type) for snapshot in result.snapshots]
    if len(snapshot_keys) != len(set(snapshot_keys)):
        raise ValueError("Snapshot shipment/checkpoint keys must be unique")
    expected_targets = set(snapshot_keys)
    target_keys = [(row["shipment_id"], row["checkpoint_type"]) for row in result.snapshot_targets]
    if len(target_keys) != len(set(target_keys)) or set(target_keys) != expected_targets:
        raise ValueError("Snapshot targets must have one record per unique shipment/checkpoint")


def _shipment_cohort(config: SimulationConfig, routes: tuple[RouteProfile, ...]) -> tuple[list[Shipment], dict[str, dict]]:
    n = config.data.n_shipments
    if n < len(routes):
        raise ValueError("n_shipments must be at least the number of configured routes")
    shipments: list[Shipment] = []
    random_variables: dict[str, dict] = {}
    start = config.world.start_at_utc.astimezone(UTC)
    for index in range(n):
        shipment_id = f"SHP-{index + 1:04d}"
        route = routes[index % len(routes)]
        shipment_seed = derive_seed(config.data.random_seed, "shipment", shipment_id)
        departure_rng = rng_for(shipment_seed, "departure")
        departure_minutes = int(departure_rng.integers(0, config.world.departure_window_days * 24 * 60))
        scheduled_departure = start + timedelta(minutes=departure_minutes)
        cohort_rng = rng_for(shipment_seed, "attributes")
        volume = float(cohort_rng.uniform(1.0, 5.0))
        perturb_rng = rng_for(shipment_seed, "transit-perturbation")
        transit_perturbation = float(perturb_rng.uniform(
            -config.world.transit_perturbation_hours,
            config.world.transit_perturbation_hours,
        ))
        service_rng = rng_for(shipment_seed, "berth-service")
        service_multiplier = float(service_rng.uniform(0.85, 1.15))
        transit_hours = max(0.0, route.scheduled_transit_hours + transit_perturbation)
        scheduled_port = scheduled_departure + timedelta(hours=route.scheduled_transit_hours)
        scheduled_customs = scheduled_port + timedelta(
            hours=config.world.berth_service_base_hours + route.customs.base_processing_hours
        )
        scheduled_delivery = scheduled_customs + timedelta(hours=route.inland.travel_hours)
        shipment = Shipment(
            shipment_id=shipment_id,
            route_id=route.route_id,
            origin=route.origin,
            destination_port=route.destination_port,
            final_destination=route.final_destination,
            shipment_type=("standard", "priority", "refrigerated")[index % 3],
            volume=volume,
            scheduled_departure_at=scheduled_departure,
            scheduled_port_arrival_at=scheduled_port,
            scheduled_customs_clearance_at=scheduled_customs,
            scheduled_delivery_at=scheduled_delivery,
            simulation_seed=shipment_seed,
            scenario_id="synthetic_baseline",
        )
        shipments.append(shipment)
        random_variables[shipment_id] = {
            "transit_perturbation_hours": transit_perturbation,
            "transit_hours": transit_hours,
            "berth_service_multiplier": service_multiplier,
        }
    return shipments, random_variables


def _event(
    shipment: Shipment,
    event_number: int,
    event_type: str,
    actual_at: datetime,
    stage: str,
    scheduled_at: datetime | None = None,
    cause: str | None = None,
    details: dict | None = None,
) -> tuple[ActualEvent, dict]:
    delay = hours_between(scheduled_at, actual_at) if scheduled_at is not None else None
    event = ActualEvent(
        shipment_id=shipment.shipment_id,
        event_id=f"{shipment.shipment_id}-E{event_number:02d}",
        event_type=event_type,
        scheduled_at=scheduled_at,
        actual_at=actual_at,
        stage=stage,
        event_cause=cause,
        delay_hours=delay,
        observable_after=EVENT_VISIBILITY[event_type],
    )
    row = event.model_dump(mode="python")
    row.update({
        "berth_id": None,
        "next_eligible_slot_at": None,
        "slot_wait_hours": None,
        "queue_wait_hours": None,
        "queue_length_at_arrival": None,
        "available_berths_at_arrival": None,
        "transit_perturbation_hours": None,
        "berth_service_duration_hours": None,
        "berth_service_multiplier": None,
        "customs_eligibility_at": None,
        "customs_closure_wait_hours": None,
        "customs_processing_duration_hours": None,
        "customs_workload_actual": None,
        "customs_processing_multiplier": None,
        "gate_wait_hours": None,
        "next_dispatch_at": None,
        "inland_travel_duration_hours": None,
        "inland_traffic_actual": None,
        "inland_travel_multiplier": None,
    })
    if details:
        row.update(details)
    return event, row


def _build_snapshots(
    shipment: Shipment,
    route: RouteProfile,
    variables: dict,
    timeline: dict,
    berth: BerthAssignment,
    environment: EnvironmentIndex,
    master_seed: int,
    config: SimulationConfig,
) -> tuple[list[CheckpointSnapshot], list[dict], list[dict], list[dict]]:
    departure_at = timeline["departure"]
    port_arrival_at = timeline["port_arrival"]
    customs_complete_at = timeline["customs_complete"]
    origin_env = environment.at(route.origin, departure_at)
    port_env = environment.at(route.destination_port, port_arrival_at)
    customs_env = environment.at(route.destination_port, customs_complete_at)
    inland_now = environment.at(route.final_destination, customs_complete_at)
    shared_observed = {"cargo_category": shipment.shipment_type, "volume": shipment.volume}

    checkpoint_rows = (
        ("departure", departure_at, {"departure_at": departure_at},
         {**shared_observed, "origin_condition": "observed_at_origin"},
         {"origin_congestion": origin_env["congestion_actual"], "origin_weather": origin_env["weather_actual"]}),
        ("port_arrival", port_arrival_at,
         {"departure_at": departure_at, "port_arrival_at": port_arrival_at},
         {**shared_observed,
          "transit_duration_hours": timeline["transit_duration_hours"],
          "transit_disruption_indicator": variables["transit_perturbation_hours"],
          "observed_queue_length": berth.queue_length_at_arrival,
          "berth_capacity_available": berth.available_berths_at_arrival},
         {"origin_congestion": origin_env["congestion_actual"],
          "origin_weather": origin_env["weather_actual"],
          "port_congestion": port_env["congestion_actual"],
          "port_weather": port_env["weather_actual"],
          "berth_capacity": port_env["berth_capacity_actual"],
          "queue_state": berth.queue_length_at_arrival}),
        ("customs_cleared", customs_complete_at,
         {"departure_at": departure_at, "port_arrival_at": port_arrival_at,
          "customs_complete_at": customs_complete_at},
         {**shared_observed,
          "transit_duration_hours": timeline["transit_duration_hours"],
          "transit_disruption_indicator": variables["transit_perturbation_hours"],
          "observed_queue_length": berth.queue_length_at_arrival,
          "berth_capacity_available": berth.available_berths_at_arrival,
          "customs_processing_duration_hours": timeline["customs_processing_duration_hours"],
          "truck_availability": 1.0},
         {"origin_congestion": origin_env["congestion_actual"],
          "origin_weather": origin_env["weather_actual"],
          "port_congestion": customs_env["congestion_actual"],
          "port_weather": customs_env["weather_actual"],
          "berth_capacity": customs_env["berth_capacity_actual"],
          "customs_workload": customs_env["customs_workload_actual"],
          "inland_traffic": inland_now["inland_traffic_actual"],
          "gate_state": "ready_before_cutoff" if timeline["gate_wait_hours"] == 0 else "cutoff_passed",
          "truck_availability": 1.0}),
    )

    snapshots: list[CheckpointSnapshot] = []
    target_rows: list[dict] = []
    forecast_rows: list[dict] = []
    forecast_diagnostics: list[dict] = []
    for checkpoint, checkpoint_at, observed_events, observed_stage, current_environment in checkpoint_rows:
        forecasts, diagnostics = generate_checkpoint_forecasts(
            shipment, route, checkpoint, checkpoint_at, environment,
            master_seed, config.world.berth_service_base_hours,
        )
        snapshot = CheckpointSnapshot(
            shipment_id=shipment.shipment_id,
            checkpoint_type=checkpoint,
            checkpoint_at=checkpoint_at,
            scheduled_delivery_at=shipment.scheduled_delivery_at,
            route_id=shipment.route_id,
            observed_event_times=observed_events,
            observed_stage_fields=observed_stage,
            current_environment=current_environment,
            forecasts=forecasts,
        )
        snapshots.append(snapshot)
        for forecast in forecasts:
            record = forecast.model_dump(mode="python")
            record.update({"shipment_id": shipment.shipment_id, "checkpoint_type": checkpoint})
            forecast_rows.append(record)
        forecast_diagnostics.extend(diagnostics)
        delay = final_delay_hours(timeline["delivery"], shipment.scheduled_delivery_at)
        target_rows.append({
            "shipment_id": shipment.shipment_id,
            "checkpoint_type": checkpoint,
            "remaining_hours_actual": remaining_duration_hours(timeline["delivery"], checkpoint_at),
            "actual_delay_hours": delay,
            "significant_delay_label": is_significant_delay(
                timeline["delivery"], shipment.scheduled_delivery_at,
                config.prediction.significant_delay_threshold_hours,
            ),
        })
    return snapshots, target_rows, forecast_rows, forecast_diagnostics


def simulate_cohort(config: SimulationConfig, routes: Iterable[RouteProfile], *,
                    neutralize_mechanism: str | None = None,
                    injected_transit_delay_hours: float = 0.0) -> SimulationResult:
    """Generate ground truth; optional neutralization is only for matched retrospective analysis.

    The default path is unchanged. Counterfactual modes neutralize one already
    realized mechanism while simulating the complete cohort so route-global
    berth competition and common random streams remain intact.
    """
    allowed_neutralizations = {None, "berth_slot", "vessel_bunching", "weekend_customs", "gate_cutoff"}
    if neutralize_mechanism not in allowed_neutralizations:
        raise ValueError(f"Unknown mechanism neutralization: {neutralize_mechanism!r}")
    if injected_transit_delay_hours < 0:
        raise ValueError("injected_transit_delay_hours must be nonnegative")
    routes = tuple(routes)
    if neutralize_mechanism == "weekend_customs":
        routes = tuple(route.model_copy(update={
            "customs": route.customs.model_copy(update={"operating_days": (
                "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            )})
        }) for route in routes)
    if len(routes) != 3:
        raise ValueError("Phase 1 requires exactly three configured routes")
    route_by_id = {route.route_id: route for route in routes}
    shipments, random_variables = _shipment_cohort(config, routes)
    if injected_transit_delay_hours:
        for values in random_variables.values():
            values["transit_hours"] += injected_transit_delay_hours
            values["transit_perturbation_hours"] += injected_transit_delay_hours
    environment_rows = generate_environment(
        routes, config.world, config.data.random_seed, config.data.n_shipments,
    )
    environment = EnvironmentIndex(environment_rows)
    arrivals_by_route: dict[str, list[tuple[str, datetime]]] = {route.route_id: [] for route in routes}
    for shipment in shipments:
        vars_ = random_variables[shipment.shipment_id]
        arrival = shipment.scheduled_departure_at + timedelta(hours=vars_["transit_hours"])
        arrivals_by_route[shipment.route_id].append((shipment.shipment_id, arrival))

    berth_assignments: dict[str, BerthAssignment] = {}
    for route in routes:
        service_durations = {
            shipment_id: config.world.berth_service_base_hours
            * random_variables[shipment_id]["berth_service_multiplier"]
            for shipment_id, _ in arrivals_by_route[route.route_id]
        }
        effective_capacity = (len(arrivals_by_route[route.route_id])
                              if neutralize_mechanism == "vessel_bunching"
                              else route.berth_capacity)
        berth_assignments.update(schedule_route_berths(
            arrivals_by_route[route.route_id], effective_capacity,
            route.berth_slot_hours, service_durations,
            ignore_slot_windows=neutralize_mechanism == "berth_slot",
        ))

    all_events: list[ActualEvent] = []
    event_rows: list[dict] = []
    snapshots: list[CheckpointSnapshot] = []
    targets: list[dict] = []
    forecast_rows: list[dict] = []
    forecast_diagnostics: list[dict] = []
    journey_metrics: list[dict] = []
    for shipment in shipments:
        route = route_by_id[shipment.route_id]
        vars_ = random_variables[shipment.shipment_id]
        berth = berth_assignments[shipment.shipment_id]
        departure_at = shipment.scheduled_departure_at
        port_arrival_at = departure_at + timedelta(hours=vars_["transit_hours"])
        port_complete_at = berth.berth_complete_at

        customs_start_at = next_customs_start(port_complete_at, route)
        customs_wait = hours_between(port_complete_at, customs_start_at)
        customs_start_env = environment.at(route.destination_port, customs_start_at)
        customs_workload = float(customs_start_env["customs_workload_actual"] or 0.0)
        customs_processing_multiplier = 1.0 + 0.35 * customs_workload
        customs_processing = route.customs.base_processing_hours * customs_processing_multiplier
        customs_complete_at = customs_start_at + timedelta(hours=customs_processing)
        gate_out_at = (customs_complete_at if neutralize_mechanism == "gate_cutoff"
                       else next_gate_out(customs_complete_at, route.inland.gate_cutoff_hour_utc))
        gate_wait = hours_between(customs_complete_at, gate_out_at)
        inland_env = environment.at(route.final_destination, gate_out_at)
        inland_traffic = float(inland_env["inland_traffic_actual"] or 0.0)
        inland_travel_multiplier = 1.0 + 0.20 * inland_traffic
        inland_duration = route.inland.travel_hours * inland_travel_multiplier
        delivery_at = gate_out_at + timedelta(hours=inland_duration)
        timeline = {
            "departure": departure_at,
            "port_arrival": port_arrival_at,
            "port_complete": port_complete_at,
            "customs_start": customs_start_at,
            "customs_complete": customs_complete_at,
            "gate_out": gate_out_at,
            "delivery": delivery_at,
            "transit_duration_hours": hours_between(departure_at, port_arrival_at),
            "customs_processing_duration_hours": customs_processing,
            "gate_wait_hours": gate_wait,
            "inland_travel_duration_hours": inland_duration,
        }

        event_specs = [
            ("departure", departure_at, "origin", shipment.scheduled_departure_at,
             "scheduled_origin_departure", {"transit_perturbation_hours": vars_["transit_perturbation_hours"]}),
            ("port_arrival", port_arrival_at, "transit", shipment.scheduled_port_arrival_at,
             "seeded_transit_perturbation", {"transit_perturbation_hours": vars_["transit_perturbation_hours"]}),
            ("berth_start", berth.berth_start_at, "port", next_berth_slot(
                shipment.scheduled_port_arrival_at, route.berth_slot_hours),
             "berth_slot_wait" + ("+berth_queue" if berth.queue_wait_hours > 0 else ""), {
                 "berth_id": berth.berth_id,
                 "next_eligible_slot_at": berth.slot_at_or_after_arrival,
                 "slot_wait_hours": berth.slot_wait_hours,
                 "queue_wait_hours": berth.queue_wait_hours,
                 "queue_length_at_arrival": berth.queue_length_at_arrival,
                 "available_berths_at_arrival": berth.available_berths_at_arrival,
                 "berth_service_duration_hours": berth.service_duration_hours,
                 "berth_service_multiplier": vars_["berth_service_multiplier"],
             }),
            ("port_complete", port_complete_at, "port", None, "berth_service", {}),
            ("customs_start", customs_start_at, "customs", None,
             "customs_closure_wait" if customs_wait > 0 else "customs_open", {
                 "customs_eligibility_at": port_complete_at,
                 "customs_closure_wait_hours": customs_wait,
             }),
            ("customs_complete", customs_complete_at, "customs", shipment.scheduled_customs_clearance_at,
             "workload_adjusted_processing", {
                 "customs_processing_duration_hours": customs_processing,
                 "customs_workload_actual": customs_workload,
                 "customs_processing_multiplier": customs_processing_multiplier,
             }),
            ("gate_out", gate_out_at, "inland_gate", None,
             "gate_cutoff_wait" if gate_wait > 0 else "ready_before_cutoff", {
                 "gate_wait_hours": gate_wait,
                 "next_dispatch_at": gate_out_at,
                 "inland_travel_duration_hours": inland_duration,
                 "inland_traffic_actual": inland_traffic,
                 "inland_travel_multiplier": inland_travel_multiplier,
             }),
            ("delivery", delivery_at, "inland", shipment.scheduled_delivery_at,
             "traffic_adjusted_inland_travel", {
                 "inland_travel_duration_hours": inland_duration,
                 "inland_traffic_actual": inland_traffic,
                 "inland_travel_multiplier": inland_travel_multiplier,
             }),
        ]
        for number, spec in enumerate(event_specs, 1):
            event, row = _event(shipment, number, *spec)
            all_events.append(event)
            event_rows.append(row)

        journey_snapshots, journey_targets, journey_forecasts, diagnostics = _build_snapshots(
            shipment, route, vars_, timeline, berth, environment,
            config.data.random_seed, config,
        )
        snapshots.extend(journey_snapshots)
        targets.extend(journey_targets)
        forecast_rows.extend(journey_forecasts)
        forecast_diagnostics.extend(diagnostics)
        journey_metrics.append({
            "shipment_id": shipment.shipment_id,
            "route_id": route.route_id,
            "transit_perturbation_hours": vars_["transit_perturbation_hours"],
            "total_delay_hours": final_delay_hours(delivery_at, shipment.scheduled_delivery_at),
            "transit_duration_hours": timeline["transit_duration_hours"],
            "berth_slot_wait_hours": berth.slot_wait_hours,
            "berth_queue_wait_hours": berth.queue_wait_hours,
            "berth_service_duration_hours": berth.service_duration_hours,
            "customs_closure_wait_hours": customs_wait,
            "customs_processing_duration_hours": customs_processing,
            "gate_wait_hours": gate_wait,
            "inland_travel_duration_hours": inland_duration,
            "berth_slot_wait_affected": berth.slot_wait_hours > 0,
            "queue_affected": berth.queue_wait_hours > 0,
            "customs_closure_affected": customs_wait > 0,
            "weekend_customs_closure_affected": interval_intersects_weekend_closure(
                port_complete_at, customs_start_at, route,
            ),
            "gate_cutoff_affected": gate_wait > 0,
            "berth_service_multiplier": vars_["berth_service_multiplier"],
            "actual_delivery_at": delivery_at,
        })

    result = SimulationResult(
        shipments=shipments,
        events=all_events,
        event_rows=event_rows,
        environment_rows=environment_rows,
        forecast_rows=forecast_rows,
        snapshots=snapshots,
        snapshot_targets=targets,
        journey_metrics=journey_metrics,
        forecast_diagnostics=forecast_diagnostics,
    )
    validate_simulation_result(result)
    return result


def controlled_propagation_examples(route: RouteProfile, world) -> dict:
    """Paired quiet/busy berth examples using common service durations and downstream rules."""
    day = datetime(2025, 1, 1, tzinfo=UTC)
    duration = world.berth_service_base_hours

    def downstream_after_port(port_complete_at: datetime) -> dict:
        customs_start = next_customs_start(port_complete_at, route)
        customs_complete = customs_start + timedelta(hours=route.customs.base_processing_hours)
        gate_out = next_gate_out(customs_complete, route.inland.gate_cutoff_hour_utc)
        return {"customs_start_at": customs_start, "customs_complete_at": customs_complete,
                "gate_out_at": gate_out,
                "delivery_at": gate_out + timedelta(hours=route.inland.travel_hours)}

    def pair(name: str, baseline_arrivals: list[tuple[str, datetime]], target_id: str,
             injected_hours: float, baseline_target_at: datetime) -> dict:
        service_durations = {key: duration for key, _ in baseline_arrivals}
        before = schedule_route_berths(
            baseline_arrivals, route.berth_capacity, route.berth_slot_hours, service_durations,
        )
        perturbed_arrivals = [
            (key, arrival + timedelta(hours=injected_hours) if key == target_id else arrival)
            for key, arrival in baseline_arrivals
        ]
        after = schedule_route_berths(
            perturbed_arrivals, route.berth_capacity, route.berth_slot_hours, service_durations,
        )
        baseline_downstream = downstream_after_port(before[target_id].berth_complete_at)
        perturbed_downstream = downstream_after_port(after[target_id].berth_complete_at)
        baseline_delivery = baseline_downstream["delivery_at"]
        perturbed_delivery = perturbed_downstream["delivery_at"]
        outcome_delta = hours_between(baseline_delivery, perturbed_delivery)
        return {
            "scenario": name,
            "route_id": route.route_id,
            "target_baseline_arrival_at": baseline_target_at,
            "target_perturbed_arrival_at": baseline_target_at + timedelta(hours=injected_hours),
            "injected_transit_delay_hours": injected_hours,
            "baseline_berth_start_at": before[target_id].berth_start_at,
            "perturbed_berth_start_at": after[target_id].berth_start_at,
            "baseline_port_complete_at": before[target_id].berth_complete_at,
            "perturbed_port_complete_at": after[target_id].berth_complete_at,
            "baseline_queue_wait_hours": before[target_id].queue_wait_hours,
            "perturbed_queue_wait_hours": after[target_id].queue_wait_hours,
            "baseline_berth_slot_wait_hours": before[target_id].slot_wait_hours,
            "perturbed_berth_slot_wait_hours": after[target_id].slot_wait_hours,
            "baseline_customs_start_at": baseline_downstream["customs_start_at"],
            "perturbed_customs_start_at": perturbed_downstream["customs_start_at"],
            "baseline_customs_complete_at": baseline_downstream["customs_complete_at"],
            "perturbed_customs_complete_at": perturbed_downstream["customs_complete_at"],
            "baseline_gate_out_at": baseline_downstream["gate_out_at"],
            "perturbed_gate_out_at": perturbed_downstream["gate_out_at"],
            "baseline_delivery_at": baseline_delivery,
            "perturbed_delivery_at": perturbed_delivery,
            "final_delivery_change_hours": outcome_delta,
            "final_delay_difference_hours": outcome_delta,
            "amplification_factor": outcome_delta / injected_hours,
            "common_service_duration_hours": duration,
        }

    if 0 not in route.berth_slot_hours or 12 not in route.berth_slot_hours:
        raise ValueError("Controlled examples require route slots at 00:00 and 12:00 UTC")
    quiet = pair("quiet_slot_absorption", [("target", day + timedelta(hours=1))],
                 "target", 1.0, day + timedelta(hours=1))
    busy = pair("busy_queue_with_slot_crossing", [
        ("Z-target", day + timedelta(hours=11)),
        ("A-competitor", day + timedelta(hours=13)),
    ], "Z-target", 2.0, day + timedelta(hours=11))
    return {"quiet_window": quiet, "busy_window": busy}
