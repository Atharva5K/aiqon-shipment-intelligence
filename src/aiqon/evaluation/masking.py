"""Schema-preserving inference and training feature masking."""
from __future__ import annotations

import numpy as np
import pandas as pd

_DEPENDENCIES = {
    "actual_transit_hours": ("transit_deviation_congestion_interaction",),
    "forecast_port_congestion": ("arrival_congestion_interaction", "transit_deviation_congestion_interaction"),
}

_SNAPSHOT_SOURCES = {
    "actual_transit_hours": ("stage", "transit_duration_hours"),
    "actual_customs_hours": ("stage", "customs_processing_duration_hours"),
    "observed_port_congestion": ("environment", "port_congestion"),
    "observed_customs_workload": ("environment", "customs_workload"),
    "observed_inland_traffic": ("environment", "inland_traffic"),
    "forecast_port_congestion": ("forecast", "port_congestion"),
    "forecast_customs_workload": ("forecast", "customs_workload"),
    "forecast_inland_traffic": ("forecast", "inland_traffic"),
}


def mask_snapshot_inputs(snapshot, columns):
    """Return an as-of snapshot copy with named source inputs removed.

    The feature missingness map is updated using the source keys consumed by
    ``build_features``. Other observations and forecast variables are retained.
    """
    stage = dict(snapshot.observed_stage_fields)
    environment = dict(snapshot.current_environment)
    forecasts = list(snapshot.forecasts)
    missingness = dict(snapshot.feature_missingness)
    for column in columns:
        source = _SNAPSHOT_SOURCES.get(column)
        if source is None:
            raise ValueError(f"No snapshot source is defined for feature {column!r}")
        kind, name = source
        if kind == "stage":
            stage.pop(name, None)
        elif kind == "environment":
            environment.pop(name, None)
        else:
            forecasts = [forecast for forecast in forecasts if forecast.forecast_variable != name]
        missingness[name] = True
    return snapshot.model_copy(update={
        "observed_stage_fields": stage,
        "current_environment": environment,
        "forecasts": tuple(forecasts),
        "feature_missingness": missingness,
    })


def apply_feature_mask(frame: pd.DataFrame, columns, mask=None) -> pd.DataFrame:
    """Mask eligible values while retaining exact feature columns and flags."""
    result = frame.copy(deep=True)
    if mask is None:
        mask = np.ones(len(result), dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if len(mask) != len(result):
            raise ValueError("mask row selector must match feature frame length")
    for column in columns:
        if column not in result.columns:
            raise ValueError(f"Unknown feature selected for missingness experiment: {column}")
        eligible = mask & result[column].notna().to_numpy()
        result.loc[eligible, column] = np.nan
        missing_column = f"{column}__missing"
        if missing_column in result.columns:
            result.loc[eligible, missing_column] = 1
        for derived in _DEPENDENCIES.get(column, ()):
            if derived in result.columns:
                result.loc[eligible, derived] = np.nan
                derived_missing = f"{derived}__missing"
                if derived_missing in result.columns:
                    result.loc[eligible, derived_missing] = 1
    return result


def random_mask_features(frame: pd.DataFrame, columns, probability: float, seed: int) -> pd.DataFrame:
    if not 0 <= probability < 1:
        raise ValueError("masking probability must be in [0, 1)")
    rng = np.random.default_rng(seed)
    result = frame.copy(deep=True)
    for column in columns:
        present = result[column].notna().to_numpy()
        selected = present & (rng.random(len(result)) < probability)
        result = apply_feature_mask(result, (column,), selected)
    return result

