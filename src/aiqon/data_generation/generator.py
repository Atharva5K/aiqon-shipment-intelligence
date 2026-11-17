"""Generate and export the Phase 1 synthetic ground-truth dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from aiqon.config.loader import SimulationConfig, load_config, load_routes
from aiqon.data_generation.environment import floor_hour
from aiqon.data_generation.randomness import derive_seed
from aiqon.schemas.events import ActualEvent
from aiqon.schemas.shipment import Shipment
from aiqon.schemas.snapshots import CheckpointSnapshot
from aiqon.simulation.engine import SimulationResult, controlled_propagation_examples, simulate_cohort


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "p50": None, "p90": None, "max": None}
    series = pd.Series(values, dtype="float64")
    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "min": float(series.min()),
        "p50": float(series.quantile(0.50)),
        "p90": float(series.quantile(0.90)),
        "max": float(series.max()),
    }


def build_summary(result: SimulationResult, config: SimulationConfig, routes: tuple) -> dict:
    """Build reproducibility metadata and honest dataset diagnostics."""
    delays = [row["total_delay_hours"] for row in result.journey_metrics]
    significant = sum(value > config.prediction.significant_delay_threshold_hours for value in delays)
    counts = Counter(shipment.route_id for shipment in result.shipments)
    metrics = result.journey_metrics
    forecast_frame = pd.DataFrame(result.forecast_diagnostics)
    forecast_errors_by_lead: dict[str, dict] = {}
    if not forecast_frame.empty:
        bins = [(0, 24, "0-24"), (24, 72, "24-72"), (72, float("inf"), "72+")]
        for low, high, label in bins:
            subset = forecast_frame[(forecast_frame["lead_hours"] >= low) &
                                    (forecast_frame["lead_hours"] < high)]
            forecast_errors_by_lead[label] = {
                "count": int(len(subset)),
                "mae": float(subset["absolute_error"].mean()) if len(subset) else None,
                "mean_uncertainty": float(subset["uncertainty"].mean()) if len(subset) else None,
            }
    controlled_route = next(
        (route for route in routes if route.berth_capacity == 1 and
         0 in route.berth_slot_hours and 12 in route.berth_slot_hours), routes[0]
    )
    controlled = controlled_propagation_examples(controlled_route, config.world)
    return {
        "project": config.project.name,
        "synthetic_data_notice": "All route profiles and generated outcomes are synthetic assumptions, not empirical port statistics.",
        "configuration": {
            "master_seed": config.data.random_seed,
            "shipment_count_configured": config.data.n_shipments,
            "significant_delay_threshold_hours": config.prediction.significant_delay_threshold_hours,
            "route_config_path": config.routes.config_path,
            "world_start_at_utc": config.world.start_at_utc.isoformat(),
            "departure_window_days": config.world.departure_window_days,
            "transit_perturbation_distribution": {
                "distribution": "Uniform",
                "lower_hours": -config.world.transit_perturbation_hours,
                "upper_hours": config.world.transit_perturbation_hours,
            },
            "berth_service_multiplier_distribution": {
                "distribution": "Uniform", "lower": 0.85, "upper": 1.15,
                "base_service_hours": config.world.berth_service_base_hours,
            },
            "random_streams": "SHA-256-derived per-shipment and per-route/location PCG64 streams; no global RNG",
            "environment_time_index": "Hourly UTC buckets; event conditions use the hour containing actual event time.",
            "gate_window_semantics": "Freight ready by the configured daily cutoff dispatches immediately; freight ready after cutoff waits until the next day's cutoff.",
        },
        "counts": {
            "unique_shipments": len(result.shipments),
            "unique_routes": len({shipment.route_id for shipment in result.shipments}),
            "shipments_per_route": {route.route_id: counts.get(route.route_id, 0) for route in routes},
            "actual_events": len(result.events),
            "environment_observations": len(result.environment_rows),
            "forecast_records": len(result.forecast_rows),
            "checkpoint_snapshots": len(result.snapshots),
            "significant_delay_shipments": significant,
            "significant_delay_prevalence": significant / len(result.shipments) if result.shipments else 0.0,
        },
        "total_delay_hours": _distribution(delays),
        "delay_class_counts_hours": {
            "early_or_on_time_le_0": sum(value <= 0 for value in delays),
            "normal_0_to_12": sum(0 < value <= config.prediction.significant_delay_threshold_hours for value in delays),
            "moderate_over_12_to_48": sum(config.prediction.significant_delay_threshold_hours < value <= 48 for value in delays),
            "severe_over_48": sum(value > 48 for value in delays),
        },
        "stage_durations_hours": {
            field: _distribution([float(row[field]) for row in metrics])
            for field in (
                "transit_duration_hours", "berth_service_duration_hours",
                "customs_processing_duration_hours", "inland_travel_duration_hours",
            )
        },
        "waiting_time_hours": {
            field: _distribution([float(row[field]) for row in metrics])
            for field in (
                "berth_queue_wait_hours", "berth_slot_wait_hours",
                "customs_closure_wait_hours", "gate_wait_hours",
            )
        },
        "mechanism_affected_journeys": {
            "berth_slot_wait_affected": sum(bool(row["berth_slot_wait_affected"]) for row in metrics),
            "vessel_bunching_or_berth_queue": sum(bool(row["queue_affected"]) for row in metrics),
            "customs_operating_closure": sum(bool(row["customs_closure_affected"]) for row in metrics),
            "weekend_customs_closure": sum(bool(row["weekend_customs_closure_affected"]) for row in metrics),
            "inland_gate_cutoff_miss": sum(bool(row["gate_cutoff_affected"]) for row in metrics),
        },
        "berth_service_multiplier_distribution": _distribution(
            [float(row["berth_service_multiplier"]) for row in metrics]
        ),
        "forecast_error_by_lead_bucket_hours": forecast_errors_by_lead,
        "forecast_uncertainty_increases_with_lead": all(
            abs(float(row["uncertainty"]) - (0.035 + 0.0025 * float(row["lead_hours"]))) < 1e-12
            for row in result.forecast_diagnostics
        ),
        "controlled_propagation_examples": controlled,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Cannot JSON encode {type(value).__name__}")


def _snapshot_rows(snapshots) -> list[dict]:
    rows: list[dict] = []
    for snapshot in snapshots:
        row = snapshot.model_dump(mode="python")
        for field in ("observed_event_times", "observed_stage_fields", "current_environment",
                      "forecasts", "feature_missingness"):
            row[field] = json.dumps(row[field], default=_jsonable, sort_keys=True)
        rows.append(row)
    return rows


def _write_tables(tables: dict[str, list[dict]], root: Path) -> tuple[str, dict[str, str]]:
    """Write every tabular artifact in one consistent format."""
    try:
        import pyarrow  # noqa: F401
        file_format = "parquet"
    except ImportError:
        file_format = "csv"
    paths: dict[str, str] = {}
    for logical_name, rows in tables.items():
        folder = "raw" if logical_name in {"shipments", "events", "environment_actuals", "forecasts"} else "processed"
        directory = root / "data" / folder
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{logical_name}.{file_format}"
        frame = pd.DataFrame(rows)
        if file_format == "parquet":
            frame.to_parquet(path, index=False, engine="pyarrow")
        else:
            frame.to_csv(path, index=False)
        paths[logical_name] = str(path.relative_to(root))
    return file_format, paths


def generate_dataset(config: SimulationConfig, route_config_path: str | Path | None = None,
                     output_root: str | Path = ".") -> dict:
    """Generate, validate, write, and summarize a complete configured cohort."""
    routes_path = Path(route_config_path or config.routes.config_path)
    routes = tuple(load_routes(routes_path).routes)
    result = simulate_cohort(config, routes)
    if len(result.shipments) != config.data.n_shipments:
        raise AssertionError("Generated shipment count does not match configuration")
    if len(result.snapshots) != 3 * len(result.shipments):
        raise AssertionError("Generated snapshot count must be three per shipment")

    root = Path(output_root).resolve()
    tables = {
        "shipments": [record.model_dump(mode="python") for record in result.shipments],
        "events": result.event_rows,
        "environment_actuals": result.environment_rows,
        "forecasts": result.forecast_rows,
        "prediction_snapshots": _snapshot_rows(result.snapshots),
        "snapshot_targets": result.snapshot_targets,
    }
    file_format, output_files = _write_tables(tables, root)
    summary = build_summary(result, config, routes)
    summary["storage_format"] = file_format
    summary["output_files"] = output_files
    metrics_dir = root / "reports" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary_path = metrics_dir / "dataset_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False, default=_jsonable),
        encoding="utf-8",
    )
    summary["summary_path"] = str(summary_path.relative_to(root))
    return {"result": result, "summary": summary}


@dataclass(frozen=True)
class RegeneratedJourney:
    """A shipment plus its full route-context simulation outputs."""

    shipment: Shipment
    events: tuple[ActualEvent, ...]
    journey_metrics: dict
    snapshots: tuple[CheckpointSnapshot, ...]
    snapshot_targets: tuple[dict, ...]
    forecast_records: tuple[dict, ...]
    environment_records: tuple[dict, ...]


def extract_journey(result: SimulationResult, shipment_id: str) -> RegeneratedJourney:
    """Extract one complete journey from a full-cohort result without resimulation."""
    shipment = next((row for row in result.shipments if row.shipment_id == shipment_id), None)
    if shipment is None:
        raise KeyError(f"Shipment ID not found in simulation result: {shipment_id}")
    events = tuple(row for row in result.events if row.shipment_id == shipment_id)
    snapshots = tuple(row for row in result.snapshots if row.shipment_id == shipment_id)
    targets = tuple(row.copy() for row in result.snapshot_targets if row["shipment_id"] == shipment_id)
    forecasts = tuple(row.copy() for row in result.forecast_rows if row["shipment_id"] == shipment_id)
    metric = next(row for row in result.journey_metrics if row["shipment_id"] == shipment_id)
    relevant_times = [event.actual_at for event in events]
    relevant_times.extend(snapshot.checkpoint_at for snapshot in snapshots)
    relevant_times.extend(row["forecast_valid_at"] for row in forecasts)
    start_at, end_at = floor_hour(min(relevant_times)), floor_hour(max(relevant_times))
    environment_records = tuple(
        row.copy() for row in result.environment_rows
        if row["route_id"] == shipment.route_id and start_at <= row["valid_at"] <= end_at
    )
    return RegeneratedJourney(
        shipment=shipment,
        events=events,
        journey_metrics=metric.copy(),
        snapshots=snapshots,
        snapshot_targets=targets,
        forecast_records=forecasts,
        environment_records=environment_records,
    )


def regenerate_shipment(shipment_id: str, config: SimulationConfig,
                        route_config_path: str | Path | None = None) -> RegeneratedJourney:
    """Regenerate the full cohort, then return one complete journey with route context."""
    routes = tuple(load_routes(route_config_path or config.routes.config_path).routes)
    result = simulate_cohort(config, routes)
    return extract_journey(result, shipment_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic Aiqon shipment dataset")
    parser.add_argument("--config", default="configs/simulation.yaml")
    parser.add_argument("--routes", default=None)
    parser.add_argument("--output-root", default=".")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    routes_path = args.routes or config.routes.config_path
    generated = generate_dataset(config, routes_path, args.output_root)
    summary = generated["summary"]
    print(json.dumps({"counts": summary["counts"], "storage_format": summary["storage_format"],
                      "summary_path": summary["summary_path"]}, indent=2))


if __name__ == "__main__":
    main()
