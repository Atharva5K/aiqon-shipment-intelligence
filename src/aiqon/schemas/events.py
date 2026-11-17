"""Actual event and ground-truth contract."""

from datetime import datetime
from typing import Literal
from pydantic import Field, model_validator
from .common import TimestampModel

EventType = Literal["departure", "port_arrival", "berth_start", "port_complete",
                    "customs_start", "customs_complete", "gate_out", "delivery"]
CheckpointName = Literal["departure", "port_arrival", "customs_cleared"]
EventVisibility = CheckpointName | Literal["ground_truth_only"]
GROUND_TRUTH_ONLY_EVENT_TYPES = frozenset({"gate_out", "delivery"})


class ActualEvent(TimestampModel):
    shipment_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    event_type: EventType
    scheduled_at: datetime | None = None
    actual_at: datetime
    stage: str = Field(min_length=1)
    event_cause: str | None = None
    delay_hours: float | None = None
    observable_after: EventVisibility

    @model_validator(mode="after")
    def visibility_matches_event_type(self):
        required_visibility = {
            "departure": "departure",
            "port_arrival": "port_arrival",
            "berth_start": "customs_cleared",
            "port_complete": "customs_cleared",
            "customs_start": "customs_cleared",
            "customs_complete": "customs_cleared",
            "gate_out": "ground_truth_only",
            "delivery": "ground_truth_only",
        }[self.event_type]
        if self.observable_after != required_visibility:
            raise ValueError(
                f"event_type {self.event_type} must have observable_after={required_visibility}"
            )
        return self


class ActualEnvironment(TimestampModel):
    location_id: str = Field(min_length=1)
    valid_at: datetime
    congestion_actual: float | None = None
    weather_actual: float | None = None
    customs_workload_actual: float | None = None
    berth_capacity_actual: int | None = Field(default=None, ge=0)
    inland_traffic_actual: float | None = None
