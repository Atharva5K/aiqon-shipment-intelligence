"""As-of shipment prediction engine (Phase 2)."""

from .features import build_features, FEATURE_COLUMNS

def __getattr__(name):
    if name == "PredictionService":
        from .service import PredictionService
        return PredictionService
    raise AttributeError(name)

__all__ = ["build_features", "FEATURE_COLUMNS", "PredictionService"]
