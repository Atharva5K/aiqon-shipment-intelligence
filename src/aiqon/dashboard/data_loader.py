"""Read-only loading and validation for Phase 4 dashboard inputs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from aiqon.prediction.training import _parse_snapshot, _read_table
from aiqon.schemas.shipment import Shipment
from aiqon.schemas.snapshots import CheckpointSnapshot

CHECKPOINTS = ("departure", "port_arrival", "customs_cleared")


class DashboardDataError(RuntimeError):
    """An actionable dashboard dataset or schema error."""


@dataclass(frozen=True)
class DashboardData:
    shipments: dict[str, Shipment]
    snapshots: dict[tuple[str, str], CheckpointSnapshot]
    actual_events: pd.DataFrame
    prediction_targets: pd.DataFrame

    @property
    def shipment_count(self) -> int:
        return len(self.shipments)

    @property
    def snapshot_count(self) -> int:
        return len(self.snapshots)


def load_dashboard_data(root: str | Path = ".") -> DashboardData:
    """Load data once; keep targets/events separate from predictor inputs."""
    root = Path(root)
    try:
        shipment_frame = _read_table(root, "shipments")
        event_frame = _read_table(root, "events")
        snapshot_frame = _read_table(root, "prediction_snapshots")
        target_frame = _read_table(root, "snapshot_targets")
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise DashboardDataError(f"Dashboard data is unavailable: {exc}") from exc
    if shipment_frame.empty or event_frame.empty or snapshot_frame.empty or target_frame.empty:
        raise DashboardDataError("Dashboard source tables must not be empty.")
    if shipment_frame["shipment_id"].astype(str).duplicated().any():
        raise DashboardDataError("Shipment table contains duplicate shipment IDs.")
    if target_frame.duplicated(["shipment_id", "checkpoint_type"]).any():
        raise DashboardDataError("Snapshot targets contain duplicate shipment/checkpoint rows.")
    if snapshot_frame.duplicated(["shipment_id", "checkpoint_type"]).any():
        raise DashboardDataError("Prediction snapshots contain duplicate shipment/checkpoint rows.")
    if event_frame["event_id"].astype(str).duplicated().any():
        raise DashboardDataError("Actual event table contains duplicate event IDs.")

    shipments = {}
    try:
        for row in shipment_frame.to_dict("records"):
            shipment = Shipment.model_validate(row)
            shipments[shipment.shipment_id] = shipment
        snapshots = {}
        for _, row in snapshot_frame.iterrows():
            snapshot = _parse_snapshot(row)
            key = (snapshot.shipment_id, snapshot.checkpoint_type)
            if key[0] not in shipments:
                raise DashboardDataError(f"Snapshot references unknown shipment {key[0]!r}.")
            snapshots[key] = snapshot
    except DashboardDataError:
        raise
    except Exception as exc:
        raise DashboardDataError(f"Could not validate dashboard shipment/snapshot data: {exc}") from exc

    expected_ids = set(shipments)
    if len(shipments) != 300:
        raise DashboardDataError(f"Expected 300 unique shipments; found {len(shipments)}.")
    if len(snapshot_frame) != 900 or len(snapshots) != 900:
        raise DashboardDataError(f"Expected 900 unique checkpoint snapshots; found {len(snapshots)}.")
    if set(snapshot_frame["shipment_id"].astype(str)) != expected_ids:
        raise DashboardDataError("Snapshot shipment IDs do not match the shipment master table.")
    for shipment_id in expected_ids:
        checkpoints = {cp for sid, cp in snapshots if sid == shipment_id}
        if checkpoints != set(CHECKPOINTS):
            raise DashboardDataError(f"Shipment {shipment_id} does not have exactly the three required checkpoints.")
    route_ids = {shipment.route_id for shipment in shipments.values()}
    if len(route_ids) != 3:
        raise DashboardDataError(f"Expected exactly three route profiles; found {len(route_ids)}.")
    deliveries = event_frame[event_frame["event_type"] == "delivery"]
    if set(deliveries["shipment_id"].astype(str)) != expected_ids or deliveries["shipment_id"].duplicated().any():
        raise DashboardDataError("Actual event table must contain exactly one delivery event per shipment.")

    for column in ("actual_at", "scheduled_at"):
        if column in event_frame:
            event_frame[column] = pd.to_datetime(event_frame[column], utc=True, errors="coerce")
    # This table is retained for retrospective display/metrics only. The
    # prediction adapter receives only Shipment + CheckpointSnapshot objects.
    target_frame = target_frame.copy(deep=True)
    return DashboardData(shipments, snapshots, event_frame.copy(deep=True), target_frame)

