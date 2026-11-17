"""Prediction output contract (not a prediction implementation)."""

from datetime import datetime, timedelta
from typing import ClassVar
from pydantic import Field, model_validator
from .common import TimestampModel
from .events import CheckpointName


class PredictionOutput(TimestampModel):
    """Prediction record.

    ``predicted_remaining_hours`` is measured from ``prediction_created_at`` for
    contract validation. ``scheduled_delivery_at`` is absent, so predicted total
    delay cannot be cross-checked against a scheduled ETA here.
    """

    shipment_id: str = Field(min_length=1)
    checkpoint_type: CheckpointName
    prediction_created_at: datetime
    predicted_remaining_hours: float = Field(ge=0)
    predicted_delivery_at: datetime
    predicted_total_delay_hours: float
    probability_delay_over_12h: float = Field(ge=0, le=1)
    remaining_hours_lower: float = Field(ge=0)
    remaining_hours_upper: float = Field(ge=0)
    propagation_remaining_hours: float | None = Field(default=None, ge=0)
    ml_residual_hours: float | None = None
    model_version: str = Field(min_length=1)
    feature_schema_version: str = Field(min_length=1)

    DELIVERY_TIME_TOLERANCE: ClassVar[timedelta] = timedelta(seconds=1)

    @model_validator(mode="after")
    def interval_is_ordered(self):
        if self.remaining_hours_lower > self.remaining_hours_upper:
            raise ValueError("remaining-hours interval lower bound exceeds upper bound")
        if not self.remaining_hours_lower <= self.predicted_remaining_hours <= self.remaining_hours_upper:
            raise ValueError("predicted_remaining_hours must fall within its interval")
        if self.predicted_remaining_hours < 0:
            raise ValueError("predicted_remaining_hours must be nonnegative")
        if self.propagation_remaining_hours is not None and self.propagation_remaining_hours < 0:
            raise ValueError("propagation_remaining_hours must be nonnegative")
        expected_delivery = self.prediction_created_at + timedelta(hours=self.predicted_remaining_hours)
        if abs(self.predicted_delivery_at - expected_delivery) > self.DELIVERY_TIME_TOLERANCE:
            raise ValueError(
                "predicted_delivery_at must equal prediction_created_at plus "
                "predicted_remaining_hours within 1 second"
            )
        return self
