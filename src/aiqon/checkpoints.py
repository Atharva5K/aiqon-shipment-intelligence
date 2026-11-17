"""Checkpoint information rules and explicit feature allowlists."""

from dataclasses import dataclass
from typing import FrozenSet

CHECKPOINTS = ("departure", "port_arrival", "customs_cleared")


@dataclass(frozen=True)
class CheckpointDefinition:
    name: str
    trigger: str
    happened: tuple[str, ...]
    observable: tuple[str, ...]
    unavailable: tuple[str, ...]
    usable_actual_events: tuple[str, ...]
    usable_forecasts: str
    hidden_outcomes: tuple[str, ...]


CHECKPOINT_DEFINITIONS = (
    CheckpointDefinition("departure", "Origin departure event has occurred.",
        ("origin departure",),
        ("shipment and route metadata", "scheduled timestamps", "actual departure time",
         "origin observations", "forecasts issued by checkpoint", "published schedules/calendars"),
        ("future vessel arrival", "future berth and customs outcomes", "future actual conditions"),
        ("departure",), "Only vintages issued at or before checkpoint time.",
        ("actual port arrival", "berth completion and service draw", "customs outcomes",
         "final delivery and derived targets")),
    CheckpointDefinition("port_arrival", "Vessel has arrived, before berth/port handling completes.",
        ("origin departure", "port arrival"),
        ("departure information", "actual transit duration", "arrival congestion/queue and berth availability",
         "forecasts issued by checkpoint"),
        ("berth completion", "customs outcomes", "future actual weather/traffic"),
        ("departure", "port_arrival"), "Only vintages issued at or before checkpoint time.",
        ("berth service draw/completion", "customs start/completion", "gate/delivery outcomes",
         "final delivery and derived targets")),
    CheckpointDefinition("customs_cleared", "Customs clearance completion event has occurred.",
        ("origin departure", "port arrival", "customs completion"),
        ("previously observed events", "actual customs duration", "current inland/gate/traffic observations",
         "forecasts issued by checkpoint"),
        ("future gate-out and delivery", "future actual traffic/weather"),
        ("departure", "port_arrival", "customs_complete"),
        "Only vintages issued at or before checkpoint time.",
        ("actual gate-out", "actual delivery and derived targets")),
)

COMMON_ALLOWED: FrozenSet[str] = frozenset({
    "shipment_id", "checkpoint_type", "checkpoint_at", "scheduled_delivery_at", "route_id",
    "observed_event_times", "observed_stage_fields", "current_environment", "forecasts",
    "feature_missingness",
})
FEATURE_ALLOWLISTS: dict[str, FrozenSet[str]] = {name: COMMON_ALLOWED for name in CHECKPOINTS}
OBSERVED_EVENT_ALLOWLISTS: dict[str, FrozenSet[str]] = {
    "departure": frozenset({"departure_at"}),
    "port_arrival": frozenset({"departure_at", "port_arrival_at"}),
    "customs_cleared": frozenset({"departure_at", "port_arrival_at", "customs_complete_at"}),
}
OBSERVED_STAGE_ALLOWLISTS: dict[str, FrozenSet[str]] = {
    "departure": frozenset({"cargo_category", "volume", "origin_condition"}),
    "port_arrival": frozenset({"cargo_category", "volume", "origin_condition",
        "transit_duration_hours", "transit_disruption_indicator", "observed_queue_length",
        "berth_capacity_available"}),
    "customs_cleared": frozenset({"cargo_category", "volume", "origin_condition",
        "transit_duration_hours", "transit_disruption_indicator", "observed_queue_length",
        "berth_capacity_available", "customs_processing_duration_hours", "truck_availability"}),
}
CURRENT_ENVIRONMENT_ALLOWLISTS: dict[str, FrozenSet[str]] = {
    "departure": frozenset({"origin_congestion", "origin_weather"}),
    "port_arrival": frozenset({"origin_congestion", "origin_weather", "port_congestion",
        "port_weather", "berth_capacity", "queue_state"}),
    "customs_cleared": frozenset({"origin_congestion", "origin_weather", "port_congestion",
        "port_weather", "berth_capacity", "queue_state", "customs_workload",
        "inland_traffic", "gate_state", "truck_availability"}),
}
FORECAST_VARIABLE_ALLOWLISTS: dict[str, FrozenSet[str]] = {
    "departure": frozenset({"transit_congestion", "port_congestion", "port_weather",
        "customs_workload", "inland_traffic"}),
    "port_arrival": frozenset({"port_congestion", "port_weather", "customs_workload", "inland_traffic"}),
    "customs_cleared": frozenset({"inland_traffic"}),
}
FORBIDDEN_FEATURES: FrozenSet[str] = frozenset({
    "actual_delivery_at", "actual_delay_hours", "remaining_hours_actual", "significant_delay_label",
    "future_actual_events", "future_environment_actuals", "actual_customs_start_at",
    "actual_customs_complete_at", "actual_port_complete_at", "actual_gate_out_at",
    "actual_final_delivery_at", "realized_berth_service_multiplier",
})


def validate_snapshot_features(checkpoint: str, fields: set[str] | frozenset[str]) -> None:
    """Reject unknown checkpoints, unallowlisted keys and known truth leakage."""
    if checkpoint not in FEATURE_ALLOWLISTS:
        raise ValueError(f"Unknown checkpoint: {checkpoint}")
    forbidden = fields & FORBIDDEN_FEATURES
    if forbidden:
        raise ValueError(f"Forbidden ground-truth feature(s): {sorted(forbidden)}")
    extra = fields - FEATURE_ALLOWLISTS[checkpoint]
    if extra:
        raise ValueError(f"Feature(s) not allowed at {checkpoint}: {sorted(extra)}")


def validate_forecast_issue_time(forecast_issued_at, checkpoint_at) -> None:
    """Require timezone-aware forecast/checkpoint times and an eligible vintage."""
    from .schemas.common import require_aware_utc
    issued = require_aware_utc(forecast_issued_at)
    checkpoint = require_aware_utc(checkpoint_at)
    if issued > checkpoint:
        raise ValueError("forecast_issued_at is after the checkpoint; forecast is not usable")


def validate_snapshot(snapshot) -> None:
    """Validate the complete as-of feature contract, including nested maps."""
    checkpoint = snapshot.checkpoint_type
    validate_snapshot_features(checkpoint, set(snapshot.model_fields_set) & COMMON_ALLOWED)
    nested_contracts = (
        ("observed_event_times", OBSERVED_EVENT_ALLOWLISTS[checkpoint]),
        ("observed_stage_fields", OBSERVED_STAGE_ALLOWLISTS[checkpoint]),
        ("current_environment", CURRENT_ENVIRONMENT_ALLOWLISTS[checkpoint]),
    )
    for field_name, allowed in nested_contracts:
        supplied = set(getattr(snapshot, field_name))
        extra = supplied - allowed
        if extra:
            raise ValueError(f"Field(s) not allowed in {field_name} at {checkpoint}: {sorted(extra)}")
    from .schemas.common import require_aware_utc
    checkpoint_at = require_aware_utc(snapshot.checkpoint_at)
    observed_times = {
        event_name: require_aware_utc(event_at)
        for event_name, event_at in snapshot.observed_event_times.items()
    }
    for event_name, event_at in observed_times.items():
        if event_at > checkpoint_at:
            raise ValueError(
                f"Observed event timestamp {event_name} is later than checkpoint_at"
            )
    trigger_by_checkpoint = {
        "departure": "departure_at",
        "port_arrival": "port_arrival_at",
        "customs_cleared": "customs_complete_at",
    }
    trigger_name = trigger_by_checkpoint[checkpoint]
    if trigger_name not in observed_times:
        raise ValueError(f"Checkpoint {checkpoint} requires triggering event {trigger_name}")
    if observed_times[trigger_name] != checkpoint_at:
        raise ValueError(
            f"Triggering event {trigger_name} must equal checkpoint_at for {checkpoint}"
        )
    missingness_extra = set(snapshot.feature_missingness) - (
        OBSERVED_EVENT_ALLOWLISTS[checkpoint] | OBSERVED_STAGE_ALLOWLISTS[checkpoint]
        | CURRENT_ENVIRONMENT_ALLOWLISTS[checkpoint]
    )
    if missingness_extra:
        raise ValueError(f"Missingness flag(s) refer to unavailable fields at {checkpoint}: {sorted(missingness_extra)}")
    for forecast in snapshot.forecasts:
        validate_forecast_issue_time(forecast.forecast_issued_at, snapshot.checkpoint_at)
    forecast_variables = {forecast.forecast_variable for forecast in snapshot.forecasts}
    extra_forecasts = forecast_variables - FORECAST_VARIABLE_ALLOWLISTS[checkpoint]
    if extra_forecasts:
        raise ValueError(f"Forecast variable(s) not allowed at {checkpoint}: {sorted(extra_forecasts)}")
