"""Deterministic hourly synthetic actual operating conditions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import ceil
from typing import Iterable

import numpy as np

from aiqon.config.loader import RouteProfile, WorldConfig
from aiqon.data_generation.randomness import derive_seed, rng_for
from aiqon.schemas.events import ActualEnvironment

UTC = timezone.utc
HOUR = timedelta(hours=1)


def floor_hour(value: datetime) -> datetime:
    """Return the containing UTC hour bucket."""
    value = value.astimezone(UTC)
    return value.replace(minute=0, second=0, microsecond=0)


class EnvironmentIndex:
    """Index actual conditions by location and hourly valid time."""

    def __init__(self, records: Iterable[dict]):
        self._rows = {(row["location_id"], row["valid_at"]): row for row in records}

    def at(self, location_id: str, timestamp: datetime) -> dict:
        """Return actual conditions from the hour containing the actual event."""
        key = (location_id, floor_hour(timestamp))
        try:
            return self._rows[key]
        except KeyError as exc:
            raise ValueError(f"No environment actual for {location_id!r} at {key[1].isoformat()}") from exc


def generate_environment(routes: Iterable[RouteProfile], world: WorldConfig, master_seed: int,
                         cohort_size: int = 300) -> list[dict]:
    """Generate route/location actuals from independent seeded streams.

    Congestion, weather, customs workload, and inland traffic use bounded
    autoregressive hourly indicators. Berth capacity follows configured route
    capacity and is not stochastically disrupted.
    """
    routes = tuple(routes)
    start = floor_hour(world.start_at_utc)
    longest_transit = max(route.scheduled_transit_hours for route in routes)
    max_route_count = ceil(cohort_size / len(routes))
    queue_buffer_days = max(
        14,
        max(ceil(max_route_count / (route.berth_capacity * len(route.berth_slot_hours))) + 7
            for route in routes),
    )
    end = (start + timedelta(days=world.departure_window_days)
           + timedelta(hours=longest_transit) + timedelta(days=queue_buffer_days))
    hours = int((end - start).total_seconds() // 3600) + 1
    output: list[dict] = []

    for route in routes:
        locations = (
            (route.origin, "origin"),
            (route.destination_port, "port"),
            (route.final_destination, "inland"),
        )
        for location_id, location_kind in locations:
            environment_seed = derive_seed(master_seed, "environment", route.route_id, location_id)
            rng = rng_for(master_seed, "environment", route.route_id, location_id)
            congestion = float(route.congestion.baseline)
            weather = 0.30
            customs = 0.50
            traffic = 0.45
            for offset in range(hours):
                valid_at = start + offset * HOUR
                hour_angle = 2 * np.pi * valid_at.hour / 24
                week_angle = 2 * np.pi * valid_at.weekday() / 7
                daily = 0.10 * np.sin(hour_angle - np.pi / 3)
                weekly = 0.06 * np.cos(week_angle)
                noise_scale = min(0.12, 0.025 + 0.15 * route.congestion.volatility)
                congestion_target = route.congestion.baseline + daily + weekly
                if location_kind == "origin":
                    congestion_target = 0.72 * route.congestion.baseline + daily * 0.65 + weekly * 0.4
                elif location_kind == "inland":
                    congestion_target = 0.62 * route.congestion.baseline + daily * 0.8 + weekly * 0.25
                congestion = float(np.clip(
                    0.72 * congestion + 0.28 * congestion_target + rng.normal(0, noise_scale), 0.0, 1.0
                ))
                weather_target = 0.28 + 0.08 * np.sin(2 * np.pi * offset / (24 * 20))
                weather = float(np.clip(0.84 * weather + 0.16 * weather_target + rng.normal(0, 0.055), 0.0, 1.0))
                customs_target = 0.28 + 0.62 * congestion if location_kind == "port" else 0.25
                customs = float(np.clip(0.75 * customs + 0.25 * customs_target + rng.normal(0, 0.045), 0.0, 1.0))
                traffic_target = 0.2 + 0.6 * congestion + 0.12 * max(0.0, np.sin(hour_angle))
                if location_kind != "inland":
                    traffic_target = 0.25 + 0.3 * congestion
                traffic = float(np.clip(0.7 * traffic + 0.3 * traffic_target + rng.normal(0, 0.05), 0.0, 1.0))

                actual = ActualEnvironment(
                    location_id=location_id,
                    valid_at=valid_at,
                    congestion_actual=congestion,
                    weather_actual=weather,
                    customs_workload_actual=customs if location_kind == "port" else None,
                    berth_capacity_actual=route.berth_capacity if location_kind == "port" else None,
                    inland_traffic_actual=traffic,
                )
                row = actual.model_dump(mode="python")
                row["route_id"] = route.route_id
                row["location_kind"] = location_kind
                row["environment_seed"] = environment_seed
                output.append(row)
    return output
