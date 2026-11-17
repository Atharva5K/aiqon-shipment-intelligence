"""Strict adapter over the separately trained Phase 3 artifacts."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from aiqon.evaluation.conformal import prediction_interval
from aiqon.evaluation.probability import significant_delay_probabilities
from aiqon.prediction.features import FEATURE_COLUMNS, build_features
from aiqon.prediction.propagation import estimate_remaining_duration
from aiqon.prediction.service import validate_artifact_runtime_versions
from aiqon.schemas.predictions import PredictionOutput
from aiqon.schemas.shipment import Shipment
from aiqon.schemas.snapshots import CheckpointSnapshot
from aiqon.dashboard.data_loader import CHECKPOINTS


class DashboardArtifactError(RuntimeError):
    """An essential Phase 3 artifact is missing, incompatible, or invalid."""


@dataclass(frozen=True)
class Phase3Prediction:
    shipment_id: str
    checkpoint_type: str
    prediction_created_at: datetime
    predicted_remaining_hours: float
    predicted_delivery_at: datetime
    predicted_total_delay_hours: float
    probability_delay_over_12h: float
    remaining_hours_lower: float | None
    remaining_hours_upper: float | None
    propagation_remaining_hours: float
    ml_residual_hours: float
    model_version: str
    feature_schema_version: str
    propagation_components: dict[str, float]
    clamped: bool
    interval_status: str
    probability_status: str = "empirical calibration-based approximation"
    validated_output: PredictionOutput | None = None


class Phase3PredictionAdapter:
    """Apply the Phase 3 hybrid model, residual risk, and conformal interval."""

    def __init__(self, bundle: dict, calibration: dict, metadata: dict):
        self._validate_artifacts(bundle, calibration, metadata)
        self.bundle = bundle
        self.calibration = calibration
        self.metadata = metadata
        self.routes = {route.route_id: route for route in bundle["routes"]}
        self.evaluation = bundle["evaluation_configuration"]

    @classmethod
    def load(cls, bundle_path, calibration_path=None, metadata_path=None):
        bundle_path = Path(bundle_path)
        calibration_path = Path(calibration_path or bundle_path.with_name("phase3_calibration.joblib"))
        metadata_path = Path(metadata_path or bundle_path.with_name("phase3_metadata.json"))
        for label, path in (("Phase 3 model bundle", bundle_path),
                            ("Phase 3 calibration", calibration_path),
                            ("Phase 3 metadata", metadata_path)):
            if not path.is_file():
                raise DashboardArtifactError(f"{label} is missing: {path}")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            validate_artifact_runtime_versions(metadata)
            bundle = joblib.load(bundle_path)
            calibration = joblib.load(calibration_path)
        except Exception as exc:
            if isinstance(exc, DashboardArtifactError):
                raise
            raise DashboardArtifactError(
                f"Could not load compatible Phase 3 model/calibration artifacts: {exc}") from exc
        return cls(bundle, calibration, metadata)

    @staticmethod
    def _validate_artifacts(bundle, calibration, metadata):
        if not isinstance(bundle, dict) or not isinstance(calibration, dict) or not isinstance(metadata, dict):
            raise DashboardArtifactError("Phase 3 artifacts must contain serialized mappings.")
        required_bundle = {"model_version", "feature_schema_version", "feature_columns", "direct",
                           "residual", "random_masking_residual", "routes", "world",
                           "evaluation_configuration", "experiment_fingerprint_sha256"}
        missing = required_bundle - set(bundle)
        if missing:
            raise DashboardArtifactError(f"Phase 3 bundle is missing required fields: {sorted(missing)}")
        if tuple(bundle["feature_columns"]) != tuple(FEATURE_COLUMNS):
            raise DashboardArtifactError("Phase 3 feature columns do not match the runtime schema and order.")
        if bundle["feature_schema_version"] != "phase2.features.v2":
            raise DashboardArtifactError(f"Unsupported Phase 3 feature schema: {bundle['feature_schema_version']!r}")
        for component in ("direct", "residual", "random_masking_residual"):
            if not callable(getattr(bundle[component], "predict", None)):
                raise DashboardArtifactError(f"Phase 3 component {component!r} has no predict method.")
        for field in ("python_version", "sklearn_version", "joblib_version"):
            if field not in metadata:
                raise DashboardArtifactError(f"Phase 3 metadata is missing runtime field {field!r}.")
        try:
            validate_artifact_runtime_versions(metadata)
        except Exception as exc:
            raise DashboardArtifactError(f"Phase 3 runtime compatibility check failed: {exc}") from exc
        fingerprint = bundle["experiment_fingerprint_sha256"]
        if not fingerprint or fingerprint != metadata.get("experiment_fingerprint_sha256"):
            raise DashboardArtifactError("Phase 3 bundle and metadata experiment fingerprints do not match.")
        if fingerprint != calibration.get("experiment_fingerprint_sha256"):
            raise DashboardArtifactError("Phase 3 calibration belongs to a different experiment fingerprint.")
        required_calibration = {"conformal_by_checkpoint", "normalized_signed_calibration_residuals_by_checkpoint",
                                "minimum_scale_hours", "significant_delay_threshold_hours"}
        missing_calibration = required_calibration - set(calibration)
        if missing_calibration:
            raise DashboardArtifactError(f"Phase 3 calibration is missing fields: {sorted(missing_calibration)}")
        minimum_scale = float(calibration["minimum_scale_hours"])
        threshold = float(calibration["significant_delay_threshold_hours"])
        if not math.isfinite(minimum_scale) or minimum_scale <= 0 or not math.isfinite(threshold) or threshold <= 0:
            raise DashboardArtifactError("Phase 3 calibration scale and significant-delay threshold must be positive and finite.")
        if not 0 < float(calibration.get("coverage", 0)) < 1:
            raise DashboardArtifactError("Phase 3 conformal coverage is missing or invalid.")
        for checkpoint in CHECKPOINTS:
            qrow = calibration["conformal_by_checkpoint"].get(checkpoint)
            residuals = np.asarray(calibration["normalized_signed_calibration_residuals_by_checkpoint"].get(checkpoint, []), dtype=float)
            if not isinstance(qrow, dict) or not residuals.ndim == 1 or not len(residuals) or not np.isfinite(residuals).all():
                raise DashboardArtifactError(f"Phase 3 calibration for {checkpoint} is missing or invalid.")
            q = float(qrow.get("quantile", float("nan")))
            if math.isnan(q) or q < 0:
                raise DashboardArtifactError(f"Phase 3 conformal quantile for {checkpoint} is invalid.")
        if not bundle["routes"] or not bundle["world"]:
            raise DashboardArtifactError("Phase 3 bundle is missing route or world configuration.")

    @property
    def experiment_fingerprint(self) -> str:
        return self.bundle["experiment_fingerprint_sha256"]

    def predict(self, snapshot: CheckpointSnapshot, shipment: Shipment, *, route=None, world=None) -> Phase3Prediction:
        if snapshot.checkpoint_type not in CHECKPOINTS:
            raise ValueError(f"Unsupported prediction checkpoint: {snapshot.checkpoint_type!r}")
        if snapshot.shipment_id != shipment.shipment_id or snapshot.route_id != shipment.route_id:
            raise ValueError("Snapshot and shipment identity do not match.")
        effective_route = route or self.routes.get(shipment.route_id)
        if effective_route is None:
            raise ValueError(f"Route {shipment.route_id!r} is not present in the Phase 3 bundle.")
        effective_world = world or self.bundle["world"]
        features = build_features(snapshot, shipment, effective_route, effective_world)
        X = pd.DataFrame([{name: features[name] for name in FEATURE_COLUMNS}], columns=FEATURE_COLUMNS)
        raw_propagation, components = estimate_remaining_duration(snapshot, effective_route, effective_world)
        propagation = max(0.0, float(raw_propagation))
        residual = float(self.bundle["residual"].predict(X)[0])
        unclamped = propagation + residual
        remaining = max(0.0, float(unclamped))
        eta = snapshot.checkpoint_at + pd.Timedelta(hours=remaining).to_pytimedelta()
        delay = (eta - shipment.scheduled_delivery_at).total_seconds() / 3600.0
        checkpoint = snapshot.checkpoint_type
        residuals = np.asarray(
            self.calibration["normalized_signed_calibration_residuals_by_checkpoint"][checkpoint], dtype=float)
        probability = float(significant_delay_probabilities(
            [remaining], [propagation], [snapshot.checkpoint_at], [shipment.scheduled_delivery_at],
            residuals, self.calibration["significant_delay_threshold_hours"],
            self.calibration["minimum_scale_hours"])[0])
        q = float(self.calibration["conformal_by_checkpoint"][checkpoint]["quantile"])
        lower = upper = None
        validated = None
        interval_status = "unavailable: conformal quantile is infinite" if not math.isfinite(q) else "available"
        if math.isfinite(q):
            low, high = prediction_interval([remaining], [propagation], q,
                self.calibration["minimum_scale_hours"])
            lower, upper = float(low[0]), float(high[0])
            model_version = str(self.bundle["model_version"])
            validated = PredictionOutput(
                shipment_id=shipment.shipment_id, checkpoint_type=checkpoint,
                prediction_created_at=snapshot.checkpoint_at,
                predicted_remaining_hours=remaining, predicted_delivery_at=eta,
                predicted_total_delay_hours=delay, probability_delay_over_12h=probability,
                remaining_hours_lower=lower, remaining_hours_upper=upper,
                propagation_remaining_hours=propagation, ml_residual_hours=residual,
                model_version=model_version, feature_schema_version=self.bundle["feature_schema_version"])
        return Phase3Prediction(
            shipment_id=shipment.shipment_id, checkpoint_type=checkpoint,
            prediction_created_at=snapshot.checkpoint_at, predicted_remaining_hours=remaining,
            predicted_delivery_at=eta, predicted_total_delay_hours=delay,
            probability_delay_over_12h=probability, remaining_hours_lower=lower,
            remaining_hours_upper=upper, propagation_remaining_hours=propagation,
            ml_residual_hours=residual, model_version=str(self.bundle["model_version"]),
            feature_schema_version=str(self.bundle["feature_schema_version"]),
            propagation_components={k: float(v) for k, v in components.items()},
            clamped=bool(unclamped < 0.0), interval_status=interval_status,
            validated_output=validated)

