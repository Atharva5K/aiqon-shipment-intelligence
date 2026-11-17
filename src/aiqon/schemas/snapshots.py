"""As-of checkpoint inputs and forecast vintage contract."""

from datetime import datetime
from typing import Literal
from pydantic import Field, field_validator, model_validator
from .common import TimestampModel, require_aware_utc
from .events import CheckpointName


class Forecast(TimestampModel):
    location_id: str = Field(min_length=1)
    forecast_issued_at: datetime
    forecast_valid_at: datetime
    forecast_variable: str = Field(min_length=1)
    forecast_value: float
    uncertainty: float | None = Field(default=None, ge=0)
    source: str | None = None

    @model_validator(mode="after")
    def valid_time_follows_issue_time(self):
        if self.forecast_valid_at <= self.forecast_issued_at:
            raise ValueError("forecast_valid_at must be later than forecast_issued_at")
        return self


class CheckpointSnapshot(TimestampModel):
    """As-of prediction view; each checkpoint timestamp equals its trigger event."""

    shipment_id: str = Field(min_length=1)
    checkpoint_type: CheckpointName
    checkpoint_at: datetime
    scheduled_delivery_at: datetime
    route_id: str = Field(min_length=1)
    observed_event_times: dict[str, datetime] = Field(default_factory=dict)
    observed_stage_fields: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    current_environment: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    forecasts: tuple[Forecast, ...] = ()
    feature_missingness: dict[str, bool] = Field(default_factory=dict)

    @field_validator("observed_event_times", mode="before")
    @classmethod
    def observed_event_times_are_utc(cls, value):
        if not isinstance(value, dict):
            raise ValueError("observed_event_times must be a mapping of event names to timestamps")
        normalized = {}
        for event_name, event_at in value.items():
            if not isinstance(event_at, datetime):
                raise ValueError(f"observed event timestamp {event_name} must be a datetime")
            normalized[event_name] = require_aware_utc(event_at)
        return normalized

    @model_validator(mode="after")
    def nested_fields_are_allowlisted(self):
        # Local import avoids a module cycle: checkpoint contracts use schemas.
        from aiqon.checkpoints import validate_snapshot
        validate_snapshot(self)
        return self

