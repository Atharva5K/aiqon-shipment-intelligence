"""Typed interfaces for later phases; intentionally no service implementations."""

from datetime import datetime
from typing import Protocol, Sequence
from aiqon.schemas.events import ActualEvent, ActualEnvironment
from aiqon.schemas.predictions import PredictionOutput
from aiqon.schemas.shipment import Shipment
from aiqon.schemas.snapshots import CheckpointSnapshot


class GroundTruthSimulator(Protocol):
    def simulate(self, shipment: Shipment, seed: int) -> Sequence[ActualEvent]:
        """Return the sequentially simulated ground-truth events."""


class CheckpointSnapshotBuilder(Protocol):
    def build(self, shipment: Shipment, events: Sequence[ActualEvent],
              environment: Sequence[ActualEnvironment], checkpoint: str,
              checkpoint_at: datetime) -> CheckpointSnapshot:
        """Build an as-of snapshot; implementations must filter future knowledge."""


class PropagationEstimator(Protocol):
    def estimate_remaining_hours(self, snapshot: CheckpointSnapshot) -> float:
        """Estimate remaining duration from allowed checkpoint inputs only."""


class PredictionService(Protocol):
    def predict(self, snapshot: CheckpointSnapshot) -> PredictionOutput:
        """Return a prediction record for one as-of checkpoint snapshot."""
