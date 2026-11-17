"""Normalized split conformal calibration by checkpoint."""
from __future__ import annotations

import math
import numpy as np


def normalized_scores(actual, predicted, propagation, minimum_scale_hours: float):
    if minimum_scale_hours <= 0:
        raise ValueError("minimum_scale_hours must be positive")
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    propagation = np.asarray(propagation, dtype=float)
    scale = np.maximum(propagation, minimum_scale_hours)
    return np.abs(actual - predicted) / scale, scale


def finite_sample_quantile(scores, coverage: float) -> float:
    """Return the split-conformal order statistic or infinity when rank exceeds n."""
    scores = np.asarray(scores, dtype=float)
    if not 0 < coverage < 1:
        raise ValueError("coverage must be strictly between zero and one")
    if len(scores) == 0:
        raise ValueError("At least one calibration score is required")
    rank = math.ceil((len(scores) + 1) * coverage)
    if rank > len(scores):
        return float("inf")
    return float(np.partition(scores, rank - 1)[rank - 1])


def prediction_interval(predicted, propagation, quantile: float, minimum_scale_hours: float):
    predicted = np.asarray(predicted, dtype=float)
    propagation = np.asarray(propagation, dtype=float)
    scale = np.maximum(propagation, minimum_scale_hours)
    lower = np.maximum(0.0, predicted - quantile * scale)
    upper = np.maximum(lower, predicted + quantile * scale)
    return lower, upper

