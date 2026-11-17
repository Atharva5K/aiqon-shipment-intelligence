"""Single-snapshot and historical replay prediction service."""
from __future__ import annotations
import json
import platform
from pathlib import Path
from datetime import timedelta
import joblib
import sklearn
import numpy as np
import pandas as pd
from aiqon.schemas.predictions import PredictionOutput
from aiqon.schemas.shipment import Shipment
from aiqon.schemas.snapshots import CheckpointSnapshot
from .features import build_features, FEATURE_COLUMNS
from .baselines import scheduled_eta_remaining, carry_forward_remaining
from .propagation import estimate_remaining_duration
from .training import BUNDLE_VERSION, FEATURE_SCHEMA_VERSION

PINNED_SKLEARN_VERSION="1.6.1"
PINNED_JOBLIB_VERSION="1.4.2"

def validate_artifact_runtime_versions(stored, runtime=None):
    """Require Python major/minor compatibility and exact pinned sklearn/joblib versions.

    Python patch releases are accepted within the same major/minor series.
    Pickled sklearn pipelines require the declared exact sklearn and joblib
    versions; callers on other versions should install the project environment.
    """
    runtime=runtime or {"python_version":platform.python_version(),
                        "sklearn_version":sklearn.__version__,"joblib_version":joblib.__version__}
    for key in ("python_version","sklearn_version","joblib_version"):
        if not stored.get(key):
            raise ValueError(f"Artifact metadata is missing {key}; use the declared project environment and retrain the bundle.")
    def major_minor(version,key):
        try: return tuple(int(part) for part in version.split(".")[:2])
        except (AttributeError,ValueError) as exc:
            raise ValueError(f"Invalid {key} in artifact/runtime metadata: {version!r}") from exc
    if major_minor(stored["python_version"],"python_version") != major_minor(runtime["python_version"],"python_version"):
        raise ValueError(f"Python artifact version mismatch: stored={stored['python_version']}, current={runtime['python_version']}; use Python {'.'.join(stored['python_version'].split('.')[:2])}.x for this bundle.")
    for key,pinned in (("sklearn_version",PINNED_SKLEARN_VERSION),("joblib_version",PINNED_JOBLIB_VERSION)):
        if runtime[key] != pinned:
            raise ValueError(f"{key} compatibility mismatch: stored={stored[key]}, current={runtime[key]}, declared project version={pinned}; install the pinned project environment and retry.")
        if stored[key] != runtime[key]:
            raise ValueError(f"{key} mismatch: stored={stored[key]}, current={runtime[key]}; install the pinned project environment ({key}=={pinned}) or retrain the model bundle.")

class PredictionService:
    def __init__(self, bundle, history_path=None):
        self._validate_bundle(bundle)
        self.bundle=bundle; self.history_path=Path(history_path) if history_path else None
        self.routes={r.route_id:r for r in bundle["routes"]}

    @staticmethod
    def _validate_bundle(bundle):
        required={"bundle_version","direct","residual","classifier","routes","world","model_version",
                  "feature_schema_version","feature_columns","python_version","sklearn_version","joblib_version"}
        missing=required-set(bundle) if isinstance(bundle,dict) else required
        if missing:
            raise ValueError(f"Invalid prediction bundle: missing required components {sorted(missing)}")
        if bundle["bundle_version"] != BUNDLE_VERSION:
            raise ValueError(f"Unsupported model bundle version {bundle['bundle_version']!r}; expected {BUNDLE_VERSION}. Retrain the model bundle.")
        if bundle["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
            raise ValueError(f"Feature schema mismatch: bundle={bundle['feature_schema_version']!r}, runtime={FEATURE_SCHEMA_VERSION!r}. Retrain the model bundle.")
        if tuple(bundle["feature_columns"]) != tuple(FEATURE_COLUMNS):
            raise ValueError("Feature name/order mismatch between model bundle and runtime. Retrain the model bundle.")
        validate_artifact_runtime_versions(bundle)
        for name, method in (("direct","predict"),("residual","predict"),("classifier","predict_proba")):
            model=bundle[name]
            if not callable(getattr(model,method,None)):
                raise ValueError(f"Invalid model bundle: {name} component must implement {method}().")
            if tuple(getattr(model,"feature_names_in_",())) != tuple(FEATURE_COLUMNS):
                raise ValueError(f"Invalid model bundle: {name} was trained with a different feature order. Retrain the model bundle.")

    @classmethod
    def load(cls,path="artifacts/models/phase2_bundle.joblib",history_path=None):
        bundle_path=Path(path)
        metadata_path=bundle_path.with_name("phase2_metadata.json")
        if not metadata_path.exists():
            raise ValueError(f"Required model metadata sidecar is missing: {metadata_path}; refusing to deserialize an artifact without compatibility information.")
        try:
            metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read model metadata at {metadata_path}: {exc}") from exc
        validate_artifact_runtime_versions(metadata)
        if metadata.get("bundle_version") != BUNDLE_VERSION or metadata.get("feature_schema_version") != FEATURE_SCHEMA_VERSION or tuple(metadata.get("feature_columns",())) != tuple(FEATURE_COLUMNS):
            raise ValueError("Model metadata version or feature schema mismatch; retrain the model bundle with this project version.")
        try:
            loaded=joblib.load(bundle_path)
        except Exception as exc:
            raise ValueError(f"Could not deserialize model bundle {bundle_path}; verify the declared Python/scikit-learn/joblib versions and retrain if needed: {exc}") from exc
        return cls(loaded,history_path)

    def predict(self,snapshot: CheckpointSnapshot,shipment: Shipment,variant="hybrid") -> PredictionOutput:
        return self.predict_with_diagnostics(snapshot,shipment,variant)[0]

    def predict_with_diagnostics(self,snapshot,shipment,variant="hybrid"):
        if variant not in {"scheduled_eta","carry_forward","propagation_only","ml_only","hybrid"}:
            raise ValueError(f"unknown prediction variant: {variant}")
        route=self.routes[shipment.route_id]
        row=build_features(snapshot,shipment,route,self.bundle["world"]); X=pd.DataFrame([{k:row[k] for k in FEATURE_COLUMNS}],columns=FEATURE_COLUMNS)
        raw_prop,components=estimate_remaining_duration(snapshot,route,self.bundle["world"])
        prop=max(0.,raw_prop); ml_raw=float(self.bundle["direct"].predict(X)[0]); residual=float(self.bundle["residual"].predict(X)[0])
        if variant=="scheduled_eta": raw=scheduled_eta_remaining(snapshot,shipment)
        elif variant=="carry_forward": raw=carry_forward_remaining(snapshot,shipment)
        elif variant=="propagation_only": raw=prop
        elif variant=="ml_only": raw=ml_raw
        else: raw=prop+residual
        remaining=max(0.,float(raw))
        prob=float(self.bundle["classifier"].predict_proba(X)[0,list(self.bundle["classifier"].named_steps["model"].classes_).index(1)])
        eta=snapshot.checkpoint_at+timedelta(hours=remaining)
        delay=(eta-shipment.scheduled_delivery_at).total_seconds()/3600
        output=PredictionOutput(shipment_id=shipment.shipment_id,checkpoint_type=snapshot.checkpoint_type,
            prediction_created_at=snapshot.checkpoint_at,predicted_remaining_hours=remaining,predicted_delivery_at=eta,
            predicted_total_delay_hours=delay,probability_delay_over_12h=prob,
            remaining_hours_lower=remaining,remaining_hours_upper=remaining,
            propagation_remaining_hours=prop,ml_residual_hours=residual,
            model_version=f"{self.bundle['model_version']}:{variant}",feature_schema_version=FEATURE_SCHEMA_VERSION)
        diag={"variant":variant,"raw_prediction_hours":float(raw),"clamped":bool(raw<0),"raw_ml_hours":ml_raw,
              "raw_residual_hours":residual,"raw_propagation_hours":raw_prop,"propagation_components":components,
              "probability_status":"provisional, uncalibrated","interval_status":"point placeholders; Phase 3 conformal calibration not implemented"}
        if self.history_path:
            self.history_path.parent.mkdir(parents=True,exist_ok=True)
            with self.history_path.open("a",encoding="utf-8") as f:
                f.write(json.dumps(output.model_dump(mode="json"),sort_keys=True)+"\n")
        return output,diag

def predict_main():
    import argparse
    from aiqon.prediction.training import _read_table
    from aiqon.prediction.training import _parse_snapshot
    from aiqon.schemas.shipment import Shipment
    p=argparse.ArgumentParser(); p.add_argument("shipment_id"); p.add_argument("--checkpoint",default="departure"); p.add_argument("--variant",default="hybrid"); p.add_argument("--data-root",default="."); p.add_argument("--model",default="artifacts/models/phase2_bundle.joblib"); a=p.parse_args()
    service=PredictionService.load(a.model,Path(a.data_root)/"artifacts/prediction_history.jsonl")
    ships={r["shipment_id"]:Shipment.model_validate(r) for r in _read_table(a.data_root,"shipments").to_dict("records")}
    rows=_read_table(a.data_root,"prediction_snapshots"); row=rows[(rows.shipment_id==a.shipment_id)&(rows.checkpoint_type==a.checkpoint)].iloc[0]
    out,diag=service.predict_with_diagnostics(_parse_snapshot(row),ships[a.shipment_id],a.variant)
    print(json.dumps({"prediction":out.model_dump(mode="json"),"diagnostics":diag},indent=2))

if __name__=="__main__": predict_main()
