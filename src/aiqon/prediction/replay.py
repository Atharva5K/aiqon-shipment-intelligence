"""Replay exactly one ordered as-of snapshot at each checkpoint."""
from collections import Counter
from aiqon.prediction.training import _read_table, _parse_snapshot
from aiqon.schemas.shipment import Shipment

EXPECTED_CHECKPOINT_ORDER=("departure","port_arrival","customs_cleared")

def replay_snapshots(service, shipment, snapshots, variant="hybrid"):
    """Validate the complete journey before predicting; pass only the current snapshot per call."""
    rows=list(snapshots)
    counts=Counter(s.checkpoint_type for s in rows)
    duplicates=sorted(cp for cp,n in counts.items() if n>1)
    missing=[cp for cp in EXPECTED_CHECKPOINT_ORDER if counts.get(cp,0)==0]
    unexpected=sorted(set(counts)-set(EXPECTED_CHECKPOINT_ORDER))
    if duplicates or missing or unexpected:
        raise ValueError(f"Replay requires exactly one snapshot per checkpoint; missing={missing}, duplicates={duplicates}, unexpected={unexpected}")
    order=[s.checkpoint_type for s in rows]
    if order != list(EXPECTED_CHECKPOINT_ORDER):
        raise ValueError(f"Replay snapshots must be ordered {EXPECTED_CHECKPOINT_ORDER}; got {tuple(order)}")
    times=[s.checkpoint_at for s in rows]
    if any(later <= earlier for earlier,later in zip(times,times[1:])):
        raise ValueError("Replay checkpoint timestamps must be strictly chronological")
    if any(s.shipment_id != shipment.shipment_id for s in rows):
        raise ValueError("Replay snapshots must all belong to the requested shipment")
    return [service.predict(snapshot,shipment,variant) for snapshot in rows]

def replay_shipment(service, shipment_id, data_root=".", variant="hybrid"):
    ships={r["shipment_id"]:Shipment.model_validate(r) for r in _read_table(data_root,"shipments").to_dict("records")}
    if shipment_id not in ships: raise KeyError(f"Unknown shipment_id: {shipment_id}")
    frame=_read_table(data_root,"prediction_snapshots"); frame=frame[frame.shipment_id==shipment_id]
    snapshots=tuple(_parse_snapshot(r) for _,r in frame.iterrows())
    return replay_snapshots(service,ships[shipment_id],snapshots,variant)
