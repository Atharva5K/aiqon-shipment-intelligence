"""Calibration-residual-derived significant-delay probabilities."""
from __future__ import annotations

import numpy as np


def normalized_signed_residuals(actual, predicted, propagation, minimum_scale_hours: float):
    if minimum_scale_hours <= 0:
        raise ValueError("minimum_scale_hours must be positive")
    scale = np.maximum(np.asarray(propagation, dtype=float), minimum_scale_hours)
    return (np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)) / scale


def significant_delay_probabilities(predicted, propagation, checkpoint_at,
                                    scheduled_delivery_at, signed_calibration_residuals,
                                    threshold_hours: float, minimum_scale_hours: float):
    """Apply calibration normalized residuals at the new example's propagation scale."""
    predicted = np.asarray(predicted, dtype=float)
    propagation = np.asarray(propagation, dtype=float)
    residuals = np.asarray(signed_calibration_residuals, dtype=float)
    scale = np.maximum(propagation, minimum_scale_hours)
    required = np.asarray([
        (scheduled - checkpoint).total_seconds() / 3600.0 + threshold_hours
        for checkpoint, scheduled in zip(checkpoint_at, scheduled_delivery_at)
    ], dtype=float)
    corrected = predicted[:, None] + residuals[None, :] * scale[:, None]
    return np.mean(corrected > required[:, None], axis=1)

