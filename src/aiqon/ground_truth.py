"""Small ground-truth arithmetic utilities; no simulation logic."""

from datetime import datetime
from .schemas.common import require_aware_utc


def _hours_between(start: datetime, end: datetime) -> float:
    start_utc, end_utc = require_aware_utc(start), require_aware_utc(end)
    return (end_utc - start_utc).total_seconds() / 3600


def final_delay_hours(actual_delivery_at: datetime, scheduled_delivery_at: datetime) -> float:
    """Return actual minus scheduled delivery duration in hours."""
    return _hours_between(scheduled_delivery_at, actual_delivery_at)


def remaining_duration_hours(actual_delivery_at: datetime, checkpoint_at: datetime) -> float:
    """Return actual delivery minus checkpoint time in hours."""
    return _hours_between(checkpoint_at, actual_delivery_at)


def is_significant_delay(actual_delivery_at: datetime, scheduled_delivery_at: datetime,
                         threshold_hours: float = 12.0) -> bool:
    """Return whether final delay is strictly greater than threshold_hours."""
    if threshold_hours <= 0:
        raise ValueError("threshold_hours must be positive")
    return final_delay_hours(actual_delivery_at, scheduled_delivery_at) > threshold_hours
