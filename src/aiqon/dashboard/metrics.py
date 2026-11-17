"""Shipment-level and checkpoint-level dashboard aggregations."""
from __future__ import annotations

import numpy as np
import pandas as pd

from aiqon.dashboard.data_loader import CHECKPOINTS, DashboardData


def actual_outcomes(data: DashboardData) -> pd.DataFrame:
    deliveries = data.actual_events[data.actual_events.event_type == "delivery"]
    rows = []
    for shipment_id, shipment in data.shipments.items():
        event = deliveries[deliveries.shipment_id.astype(str) == shipment_id].iloc[0]
        delivered_at = event.actual_at
        delay = (delivered_at - shipment.scheduled_delivery_at).total_seconds() / 3600.0
        rows.append({"shipment_id": shipment_id, "actual_delivery_at": delivered_at,
                     "actual_delay_hours": float(delay), "actual_delayed": bool(delay > 0),
                     "actual_significant_delay": bool(delay > 12.0)})
    return pd.DataFrame(rows)


def build_dashboard_frame(data: DashboardData, predictions: pd.DataFrame) -> pd.DataFrame:
    shipment_rows = [{"shipment_id": sid, **shipment.model_dump(mode="python")}
                     for sid, shipment in data.shipments.items()]
    shipments = pd.DataFrame(shipment_rows)
    return (predictions.merge(shipments, on="shipment_id", validate="many_to_one")
            .merge(actual_outcomes(data), on="shipment_id", validate="many_to_one"))


def filter_dashboard_frame(frame: pd.DataFrame, *, route="All", scenario="All",
                           delay_status="All historical statuses", checkpoint="All checkpoints") -> pd.DataFrame:
    result = frame.copy(deep=True)
    if route != "All":
        result = result[result.route_id == route]
    if scenario != "All":
        result = result[result.scenario_id == scenario]
    if delay_status == "On time / early":
        result = result[~result.actual_delayed]
    elif delay_status == "Delayed (> 0h)":
        result = result[result.actual_delayed]
    elif delay_status == "Significant delay (> 12h)":
        result = result[result.actual_significant_delay]
    if checkpoint != "All checkpoints":
        result = result[result.checkpoint_type == checkpoint]
    return result.copy(deep=True)


def shipment_level_frame(frame: pd.DataFrame, checkpoint: str) -> pd.DataFrame:
    """Select one prediction per shipment; never count snapshots as shipments."""
    if checkpoint not in CHECKPOINTS:
        raise ValueError(f"Choose one prediction checkpoint from {CHECKPOINTS}.")
    result = frame[frame.checkpoint_type == checkpoint].copy(deep=True)
    if result.shipment_id.duplicated().any():
        raise ValueError("Shipment-level dashboard rows contain duplicate shipment IDs.")
    return result


def checkpoint_performance(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for checkpoint in CHECKPOINTS:
        part = frame[frame.checkpoint_type == checkpoint]
        if part.empty:
            rows.append({"checkpoint_type": checkpoint, "sample_count": 0,
                         "mae_hours": None, "rmse_hours": None})
            continue
        errors = part.predicted_total_delay_hours.to_numpy() - part.actual_delay_hours.to_numpy()
        rows.append({"checkpoint_type": checkpoint, "sample_count": int(part.shipment_id.nunique()),
                     "mae_hours": float(np.mean(np.abs(errors))),
                     "rmse_hours": float(np.mean(errors ** 2) ** .5)})
    return pd.DataFrame(rows)


def overview_kpis(selected: pd.DataFrame) -> dict:
    if selected.empty:
        return {"unique_shipments": 0, "actual_delayed_shipments": 0,
                "actual_significant_delay_shipments": 0, "predicted_mean_delay_hours": None,
                "predicted_median_delay_hours": None}
    return {"unique_shipments": int(selected.shipment_id.nunique()),
            "actual_delayed_shipments": int(selected.drop_duplicates("shipment_id").actual_delayed.sum()),
            "actual_significant_delay_shipments": int(selected.drop_duplicates("shipment_id").actual_significant_delay.sum()),
            "predicted_mean_delay_hours": float(selected.predicted_total_delay_hours.mean()),
            "predicted_median_delay_hours": float(selected.predicted_total_delay_hours.median())}

