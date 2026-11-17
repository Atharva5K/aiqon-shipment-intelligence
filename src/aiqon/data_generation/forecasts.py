"""Checkpoint-scoped, imperfect profile-based forecast generation."""

from __future__ import annotations

from datetime import datetime, timedelta

from aiqon.checkpoints import FORECAST_VARIABLE_ALLOWLISTS
from aiqon.config.loader import RouteProfile
from aiqon.data_generation.environment import EnvironmentIndex
from aiqon.data_generation.randomness import rng_for
from aiqon.schemas.events import CheckpointName
from aiqon.schemas.shipment import Shipment
from aiqon.schemas.snapshots import Forecast


def _future_at(checkpoint_at: datetime, target: datetime) -> datetime:
    return max(target, checkpoint_at + timedelta(hours=1))


def generate_checkpoint_forecasts(
    shipment: Shipment,
    route: RouteProfile,
    checkpoint: CheckpointName,
    checkpoint_at: datetime,
    environment: EnvironmentIndex,
    master_seed: int,
    berth_service_base_hours: float,
) -> tuple[tuple[Forecast, ...], list[dict]]:
    """Create independent forecasts; actual values are used only for diagnostics."""
    port_time = shipment.scheduled_port_arrival_at
    customs_time = shipment.scheduled_customs_clearance_at
    delivery_time = shipment.scheduled_delivery_at
    if checkpoint == "port_arrival":
        port_time = checkpoint_at + timedelta(hours=berth_service_base_hours)
        customs_time = port_time + timedelta(hours=route.customs.base_processing_hours)
        delivery_time = customs_time + timedelta(hours=route.inland.travel_hours)
    elif checkpoint == "customs_cleared":
        delivery_time = checkpoint_at + timedelta(hours=route.inland.travel_hours)

    specifications: dict[str, tuple[str, datetime, float]] = {
        "transit_congestion": (route.destination_port, port_time, route.congestion.baseline),
        "port_congestion": (route.destination_port, port_time, route.congestion.baseline),
        "port_weather": (route.destination_port, port_time, 0.30),
        "customs_workload": (
            route.destination_port, customs_time,
            0.28 + 0.62 * route.congestion.baseline,
        ),
        "inland_traffic": (
            route.final_destination, delivery_time,
            0.20 + 0.60 * route.congestion.baseline,
        ),
    }
    forecasts: list[Forecast] = []
    diagnostics: list[dict] = []
    for variable in sorted(FORECAST_VARIABLE_ALLOWLISTS[checkpoint]):
        location_id, target_time, profile_mean = specifications[variable]
        valid_at = _future_at(checkpoint_at, target_time)
        lead_hours = max(0.0, (valid_at - checkpoint_at).total_seconds() / 3600)
        uncertainty = 0.035 + 0.0025 * lead_hours
        rng = rng_for(master_seed, "forecast", shipment.shipment_id, checkpoint, variable)
        value = float(min(1.0, max(0.0, profile_mean + rng.normal(0.0, uncertainty))))
        forecast = Forecast(
            location_id=location_id,
            forecast_issued_at=checkpoint_at,
            forecast_valid_at=valid_at,
            forecast_variable=variable,
            forecast_value=value,
            uncertainty=uncertainty,
            source="synthetic_route_profile",
        )
        forecasts.append(forecast)
        actual_row = environment.at(location_id, valid_at)
        actual_key = {
            "transit_congestion": "congestion_actual",
            "port_congestion": "congestion_actual",
            "port_weather": "weather_actual",
            "customs_workload": "customs_workload_actual",
            "inland_traffic": "inland_traffic_actual",
        }[variable]
        actual_value = actual_row.get(actual_key)
        if actual_value is not None:
            diagnostics.append({
                "shipment_id": shipment.shipment_id,
                "checkpoint_type": checkpoint,
                "forecast_variable": variable,
                "lead_hours": lead_hours,
                "forecast_value": value,
                "actual_value_for_evaluation": float(actual_value),
                "absolute_error": abs(value - float(actual_value)),
                "uncertainty": uncertainty,
            })
    return tuple(forecasts), diagnostics
