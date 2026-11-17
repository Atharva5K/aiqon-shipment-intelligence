"""Dashboard orchestration over immutable Phase 3 prediction inputs."""
from __future__ import annotations

import pandas as pd

from aiqon.checkpoints import OBSERVED_EVENT_ALLOWLISTS
from aiqon.dashboard.data_loader import DashboardData
from aiqon.dashboard.metrics import build_dashboard_frame
from aiqon.dashboard.services.prediction_adapter import Phase3PredictionAdapter


class DashboardService:
    def __init__(self, data: DashboardData, predictor: Phase3PredictionAdapter):
        self.data = data
        self.predictor = predictor
        self._prediction_frame = None
        self._dashboard_frame = None

    def prediction_frame(self) -> pd.DataFrame:
        if self._prediction_frame is None:
            rows = []
            for (shipment_id, checkpoint), snapshot in self.data.snapshots.items():
                result = self.predictor.predict(snapshot, self.data.shipments[shipment_id])
                rows.append({"shipment_id": shipment_id, "checkpoint_type": checkpoint,
                    "checkpoint_at": result.prediction_created_at,
                    "predicted_remaining_hours": result.predicted_remaining_hours,
                    "predicted_delivery_at": result.predicted_delivery_at,
                    "predicted_total_delay_hours": result.predicted_total_delay_hours,
                    "probability_delay_over_12h": result.probability_delay_over_12h,
                    "remaining_hours_lower": result.remaining_hours_lower,
                    "remaining_hours_upper": result.remaining_hours_upper,
                    "propagation_remaining_hours": result.propagation_remaining_hours,
                    "ml_residual_hours": result.ml_residual_hours,
                    "model_version": result.model_version,
                    "feature_schema_version": result.feature_schema_version,
                    "interval_status": result.interval_status,
                    "propagation_components": dict(result.propagation_components)})
            self._prediction_frame = pd.DataFrame(rows)
        return self._prediction_frame.copy(deep=True)

    def frame(self) -> pd.DataFrame:
        if self._dashboard_frame is None:
            self._dashboard_frame = build_dashboard_frame(self.data, self.prediction_frame())
        return self._dashboard_frame.copy(deep=True)

    def shipment_history(self, shipment_id: str) -> pd.DataFrame:
        if shipment_id not in self.data.shipments:
            raise KeyError(f"Unknown shipment ID {shipment_id!r}.")
        frame = self.frame()
        return frame[frame.shipment_id == shipment_id].sort_values("checkpoint_at").copy(deep=True)

    def visible_actual_events(self, shipment_id: str, checkpoint: str, *, retrospective=False) -> pd.DataFrame:
        """Return only checkpoint-visible event types unless explicitly retrospective."""
        if shipment_id not in self.data.shipments:
            raise KeyError(f"Unknown shipment ID {shipment_id!r}.")
        snapshot = self.data.snapshots[(shipment_id, checkpoint)]
        events = self.data.actual_events[self.data.actual_events.shipment_id.astype(str) == shipment_id].copy(deep=True)
        events = events.sort_values("actual_at")
        if retrospective:
            return events
        visible_event_types = {name.removesuffix("_at") for name in OBSERVED_EVENT_ALLOWLISTS[checkpoint]}
        return events[events.event_type.isin(visible_event_types) & (events.actual_at <= snapshot.checkpoint_at)].copy(deep=True)

