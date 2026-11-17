"""Checkpoint-aware features sourced only from snapshot and manifest schedules."""
from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any

from aiqon.checkpoints import (
    OBSERVED_EVENT_ALLOWLISTS, OBSERVED_STAGE_ALLOWLISTS,
    CURRENT_ENVIRONMENT_ALLOWLISTS, FORECAST_VARIABLE_ALLOWLISTS,
    validate_snapshot_features,
)
from aiqon.schemas.shipment import Shipment
from aiqon.schemas.snapshots import CheckpointSnapshot
from .propagation import estimate_customs_timing, estimate_berth_timing, select_eligible_forecast

NUMERIC = (
    "volume", "scheduled_delivery_slack_hours", "scheduled_transit_hours",
    "scheduled_customs_hours", "scheduled_inland_hours", "route_berth_capacity",
    "route_congestion_baseline", "route_congestion_volatility", "elapsed_hours",
    "accumulated_schedule_deviation_hours", "actual_transit_hours",
    "observed_queue_length", "berth_capacity_available", "actual_customs_hours",
    "observed_truck_availability", "observed_origin_congestion", "observed_origin_weather",
    "observed_port_congestion", "observed_port_weather", "observed_queue_state",
    "observed_customs_workload", "observed_inland_traffic",
    "forecast_transit_congestion", "forecast_port_congestion", "forecast_port_weather",
    "forecast_customs_workload", "forecast_inland_traffic", "forecast_lead_hours",
    "forecast_issue_age_hours", "forecast_uncertainty", "arrival_hour_utc",
    "arrival_congestion_interaction", "transit_deviation_congestion_interaction",
    "expected_customs_open", "expected_customs_completion_hour", "customs_gate_cutoff_slack_hours",
    "estimated_customs_start_hour", "estimated_customs_wait_hours",
)
CATEGORICAL = ("route_id", "shipment_type", "checkpoint_type", "origin_condition", "observed_gate_state")
BASE_FEATURE_COLUMNS = CATEGORICAL + NUMERIC
MISSINGNESS_COLUMNS = tuple(f"{name}__missing" for name in BASE_FEATURE_COLUMNS)
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + MISSINGNESS_COLUMNS
FORBIDDEN = {"actual_delivery_at", "actual_delay_hours", "remaining_hours_actual", "significant_delay_label",
             "predicted_remaining_hours", "residual_target", "actual_gate_out_at", "actual_customs_complete_at"}


def _hours(start: datetime, end: datetime) -> float:
    return (end - start).total_seconds() / 3600


def _stage_schedule(shipment: Shipment, checkpoint: str) -> datetime:
    return {"departure": shipment.scheduled_departure_at,
            "port_arrival": shipment.scheduled_port_arrival_at,
            "customs_cleared": shipment.scheduled_customs_clearance_at}[checkpoint]


def build_features(snapshot: CheckpointSnapshot, shipment: Shipment, route: Any, world=None) -> dict[str, Any]:
    """Build fixed-schema features; schedule/config fallbacks never use future actuals."""
    if snapshot.shipment_id != shipment.shipment_id or snapshot.route_id != shipment.route_id:
        raise ValueError("snapshot and shipment manifest identity mismatch")
    cp = snapshot.checkpoint_type
    validate_snapshot_features(cp, set(snapshot.model_fields_set))
    if set(snapshot.observed_event_times) - OBSERVED_EVENT_ALLOWLISTS[cp]:
        raise ValueError("observed event outside checkpoint allowlist")
    if set(snapshot.observed_stage_fields) - OBSERVED_STAGE_ALLOWLISTS[cp]:
        raise ValueError("stage field outside checkpoint allowlist")
    if set(snapshot.current_environment) - CURRENT_ENVIRONMENT_ALLOWLISTS[cp]:
        raise ValueError("environment field outside checkpoint allowlist")
    milestone = _stage_schedule(shipment, cp)
    events, stage, env = snapshot.observed_event_times, snapshot.observed_stage_fields, snapshot.current_environment
    dep = events.get("departure_at")
    arrival = events.get("port_arrival_at")
    transit_dev = (_hours(shipment.scheduled_port_arrival_at, arrival) if arrival else 0.0)
    expected_arrival = arrival or (snapshot.checkpoint_at + timedelta(hours=route.scheduled_transit_hours))
    if world is None:
        # Compatibility fallback for direct callers without global world config:
        # use the scheduled inter-stage allowance after subtracting configured
        # customs processing, never a realized future duration.
        stage_allowance = (shipment.scheduled_customs_clearance_at -
                           shipment.scheduled_port_arrival_at).total_seconds() / 3600
        handling_hours = max(0.0, stage_allowance - route.customs.base_processing_hours)
        world = SimpleNamespace(berth_service_base_hours=handling_hours)
    port_complete, customs_start = estimate_customs_timing(snapshot, route, world)
    expected_customs_start = customs_start or snapshot.checkpoint_at
    for f in snapshot.forecasts:
        if f.forecast_variable not in FORECAST_VARIABLE_ALLOWLISTS[cp]:
            raise ValueError(f"forecast variable {f.forecast_variable!r} is not allowed at {cp}")
        if f.forecast_issued_at > snapshot.checkpoint_at:
            raise ValueError("forecast issued after checkpoint")
    queue_length = stage.get("observed_queue_length",0) if cp=="port_arrival" else 0
    preliminary_wait,_,_,_=estimate_berth_timing(expected_arrival,route,world,queue_length,route.congestion.baseline)
    estimated_berth_start=expected_arrival+timedelta(hours=preliminary_wait)
    expected_delivery = expected_customs_start + timedelta(hours=route.customs.base_processing_hours + route.inland.travel_hours)
    if cp == "customs_cleared":
        expected_delivery = snapshot.checkpoint_at + timedelta(hours=route.inland.travel_hours)
    target_time = {"transit_congestion": expected_arrival, "port_congestion": estimated_berth_start,
                   "port_weather": estimated_berth_start,
                   "customs_workload": expected_customs_start+timedelta(hours=route.customs.base_processing_hours),
                   "inland_traffic": expected_delivery}
    forecast_locations={"port_congestion":route.destination_port,"port_weather":route.destination_port,
                        "transit_congestion":route.destination_port,"customs_workload":route.destination_port,
                        "inland_traffic":route.final_destination}
    selected={variable:forecast for variable in FORECAST_VARIABLE_ALLOWLISTS[cp]
              if (forecast:=select_eligible_forecast(snapshot,variable,target_time[variable],forecast_locations[variable])) is not None}
    out: dict[str, Any] = {key: None for key in FEATURE_COLUMNS}
    out.update(route_id=route.route_id, shipment_type=shipment.shipment_type, checkpoint_type=cp,
               origin_condition=stage.get("origin_condition"), volume=shipment.volume,
               scheduled_delivery_slack_hours=_hours(snapshot.checkpoint_at, shipment.scheduled_delivery_at),
               scheduled_transit_hours=route.scheduled_transit_hours,
               scheduled_customs_hours=route.customs.base_processing_hours,
               scheduled_inland_hours=route.inland.travel_hours, route_berth_capacity=route.berth_capacity,
               route_congestion_baseline=route.congestion.baseline,
               route_congestion_volatility=route.congestion.volatility,
               elapsed_hours=_hours(dep, snapshot.checkpoint_at) if dep else 0.0,
               accumulated_schedule_deviation_hours=_hours(milestone, snapshot.checkpoint_at),
               actual_transit_hours=stage.get("transit_duration_hours"),
               observed_queue_length=stage.get("observed_queue_length"),
               berth_capacity_available=stage.get("berth_capacity_available"),
               actual_customs_hours=stage.get("customs_processing_duration_hours"),
               observed_truck_availability=stage.get("truck_availability"),
               observed_origin_congestion=env.get("origin_congestion"), observed_origin_weather=env.get("origin_weather"),
               observed_port_congestion=env.get("port_congestion"), observed_port_weather=env.get("port_weather"),
               observed_queue_state=env.get("queue_state"), observed_customs_workload=env.get("customs_workload"),
               observed_inland_traffic=env.get("inland_traffic"), observed_gate_state=env.get("gate_state"))
    for variable, forecast in selected.items():
        out[f"forecast_{variable}"] = forecast.forecast_value
    if selected:
        nearest = min(selected.items(), key=lambda kv: abs((kv[1].forecast_valid_at - target_time[kv[0]]).total_seconds()))[1]
        out["forecast_lead_hours"] = _hours(snapshot.checkpoint_at, nearest.forecast_valid_at)
        out["forecast_issue_age_hours"] = _hours(nearest.forecast_issued_at, snapshot.checkpoint_at)
        out["forecast_uncertainty"] = nearest.uncertainty
    out["arrival_hour_utc"] = expected_arrival.hour + expected_arrival.minute / 60
    out["arrival_congestion_interaction"] = (out["arrival_hour_utc"] * out["forecast_port_congestion"]
                                               if out["forecast_port_congestion"] is not None else None)
    out["transit_deviation_congestion_interaction"] = (transit_dev * out["forecast_port_congestion"]
                                                        if out["forecast_port_congestion"] is not None else None)
    if customs_start is not None:
        days = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
        out["expected_customs_open"] = int(days[port_complete.weekday()] in route.customs.operating_days and
            route.customs.opening_hour_utc <= port_complete.hour < route.customs.closing_hour_utc)
        out["estimated_customs_start_hour"] = customs_start.hour + customs_start.minute / 60
        out["estimated_customs_wait_hours"] = _hours(port_complete, customs_start)
        customs_completion = customs_start + timedelta(hours=route.customs.base_processing_hours)
        out["expected_customs_completion_hour"] = customs_completion.hour
        out["customs_gate_cutoff_slack_hours"] = route.inland.gate_cutoff_hour_utc - customs_completion.hour
    else:
        # Customs has already cleared at this checkpoint; these future-stage
        # timing features are inapplicable and are explicitly marked missing.
        out["expected_customs_open"] = None
        out["estimated_customs_start_hour"] = None
        out["estimated_customs_wait_hours"] = None
        out["expected_customs_completion_hour"] = None
        out["customs_gate_cutoff_slack_hours"] = None
    source_for = {
        "origin_condition": "origin_condition", "actual_transit_hours": "transit_duration_hours",
        "observed_queue_length": "observed_queue_length", "berth_capacity_available": "berth_capacity_available",
        "actual_customs_hours": "customs_processing_duration_hours", "observed_truck_availability": "truck_availability",
        "observed_origin_congestion": "origin_congestion", "observed_origin_weather": "origin_weather",
        "observed_port_congestion": "port_congestion", "observed_port_weather": "port_weather",
        "observed_queue_state": "queue_state", "observed_customs_workload": "customs_workload",
        "observed_inland_traffic": "inland_traffic", "observed_gate_state": "gate_state",
    }
    for variable in FORECAST_VARIABLE_ALLOWLISTS[cp]:
        source_for[f"forecast_{variable}"] = variable
    for name in BASE_FEATURE_COLUMNS:
        source = source_for.get(name, name)
        explicit_missing = snapshot.feature_missingness.get(source, False)
        out[f"{name}__missing"] = int(bool(explicit_missing) or out[name] is None)
    if FORBIDDEN & set(out):
        raise ValueError("target leakage detected in constructed feature names")
    return out


def validate_feature_names(names) -> None:
    forbidden = set(names) & FORBIDDEN
    if forbidden or set(names) != set(FEATURE_COLUMNS):
        raise ValueError(f"invalid feature schema; forbidden={sorted(forbidden)}")
