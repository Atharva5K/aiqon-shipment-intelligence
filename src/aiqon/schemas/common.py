"""Shared schema types and UTC timestamp validation."""

from datetime import datetime, timezone
from pydantic import BaseModel, ConfigDict, field_validator


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def require_aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


class TimestampModel(ContractModel):
    @field_validator("*", mode="after")
    @classmethod
    def normalize_datetime_fields(cls, value: object) -> object:
        if isinstance(value, datetime):
            return require_aware_utc(value)
        return value
