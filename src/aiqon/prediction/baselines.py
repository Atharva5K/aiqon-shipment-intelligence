"""Deterministic schedule baselines."""
from aiqon.schemas.shipment import Shipment
from aiqon.schemas.snapshots import CheckpointSnapshot


def scheduled_eta_remaining(snapshot: CheckpointSnapshot, shipment: Shipment) -> float:
    return max(0.0, (shipment.scheduled_delivery_at - snapshot.checkpoint_at).total_seconds() / 3600)


def carry_forward_remaining(snapshot: CheckpointSnapshot, shipment: Shipment) -> float:
    milestone = {"departure": shipment.scheduled_departure_at,
                 "port_arrival": shipment.scheduled_port_arrival_at,
                 "customs_cleared": shipment.scheduled_customs_clearance_at}[snapshot.checkpoint_type]
    deviation = (snapshot.checkpoint_at - milestone).total_seconds() / 3600
    return max(0.0, scheduled_eta_remaining(snapshot, shipment) + deviation)
