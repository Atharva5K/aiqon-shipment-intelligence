"""Train, validate and persist Phase 2 candidate models."""
from __future__ import annotations
import argparse, json
import hashlib
import platform
from datetime import datetime, timezone
from pathlib import Path
import joblib
import sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    log_loss, mean_absolute_error, mean_squared_error, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from aiqon.config.loader import load_config, load_routes
from aiqon.schemas.shipment import Shipment
from aiqon.schemas.snapshots import CheckpointSnapshot
from .features import FEATURE_COLUMNS, NUMERIC, CATEGORICAL, MISSINGNESS_COLUMNS, build_features, validate_feature_names
from .baselines import scheduled_eta_remaining, carry_forward_remaining
from .propagation import estimate_remaining_duration

BUNDLE_VERSION = 3
FEATURE_SCHEMA_VERSION = "phase2.features.v2"
CLASSIFICATION_THRESHOLD = 0.5

def _classification_metrics(y_true, probabilities):
    """Preliminary metrics only; threshold is fixed a priori and probabilities uncalibrated."""
    y_true=np.asarray(y_true,dtype=int); probabilities=np.asarray(probabilities,dtype=float)
    predicted=(probabilities>=CLASSIFICATION_THRESHOLD).astype(int)
    two_classes=len(np.unique(y_true))==2
    return {
        "status":"preliminary, uncalibrated",
        "probability_status":"uncalibrated",
        "threshold":CLASSIFICATION_THRESHOLD,
        "roc_auc":float(roc_auc_score(y_true,probabilities)) if two_classes else None,
        "roc_auc_unavailable_reason":None if two_classes else "validation labels contain only one class",
        "pr_auc_average_precision":float(average_precision_score(y_true,probabilities)) if two_classes else None,
        "pr_auc_unavailable_reason":None if two_classes else "validation labels contain only one class",
        "brier_score":float(brier_score_loss(y_true,probabilities)),
        "log_loss":float(log_loss(y_true,probabilities,labels=[0,1])),
        "confusion_matrix_labels_0_then_1":confusion_matrix(y_true,predicted,labels=[0,1]).tolist(),
        "precision":float(precision_score(y_true,predicted,zero_division=0)),
        "recall":float(recall_score(y_true,predicted,zero_division=0)),
        "f1":float(f1_score(y_true,predicted,zero_division=0)),
        "positive_class_count":int(y_true.sum()),
        "negative_class_count":int(len(y_true)-y_true.sum()),
        "validation_shipment_count":int(len(y_true)),
    }

def _read_table(root, logical):
    for folder in ("raw", "processed"):
        for suffix, reader in (("parquet", pd.read_parquet), ("csv", pd.read_csv)):
            path = Path(root) / "data" / folder / f"{logical}.{suffix}"
            if path.exists(): return reader(path)
    raise FileNotFoundError(f"Missing dataset table {logical} under {root}")

def _parse_snapshot(row):
    import json
    from datetime import datetime
    values = row.to_dict()
    for name in ("observed_event_times", "observed_stage_fields", "current_environment", "forecasts", "feature_missingness"):
        if isinstance(values[name], str): values[name] = json.loads(values[name])
    values["observed_event_times"] = {k: datetime.fromisoformat(v) if isinstance(v, str) else v
                                      for k,v in values["observed_event_times"].items()}
    return CheckpointSnapshot.model_validate(values)

def load_training_frame(root, config, routes, shipment_ids=None):
    """Build as-of rows for selected IDs; targets stay separate from features.

    Phase 3 calls this only after splitting the unique shipment IDs, so no
    checkpoint feature rows are expanded before the partition assignment.
    """
    shipments = {s.shipment_id:s for s in (Shipment.model_validate(r) for r in _read_table(root,"shipments").to_dict("records"))}
    route_map = {r.route_id:r for r in routes}
    target = _read_table(root,"snapshot_targets").set_index(["shipment_id","checkpoint_type"])
    rows=[]; meta=[]
    selected_ids = None if shipment_ids is None else set(shipment_ids)
    snapshot_frame = _read_table(root,"prediction_snapshots")
    if selected_ids is not None:
        snapshot_frame = snapshot_frame[snapshot_frame["shipment_id"].isin(selected_ids)]
    for _, sr in snapshot_frame.iterrows():
        snap=_parse_snapshot(sr); shipment=shipments[snap.shipment_id]; route=route_map[snap.route_id]
        row=build_features(snap,shipment,route,config.world)
        key=(snap.shipment_id,snap.checkpoint_type)
        if key not in target.index: raise ValueError(f"Missing target for {key}")
        y=target.loc[key]
        # Keep target data in a parallel structure, never concatenate it to feature columns.
        prop, parts=estimate_remaining_duration(snap,route,config.world)
        rows.append(row); meta.append({"shipment_id":snap.shipment_id,"checkpoint_type":snap.checkpoint_type,
            "remaining_hours_actual":float(y.remaining_hours_actual),"actual_delay_hours":float(y.actual_delay_hours),
            "significant_delay_label":int(bool(y.significant_delay_label)),"propagation_raw":prop,
            "propagation_remaining":max(0.,prop),"scheduled_remaining":scheduled_eta_remaining(snap,shipment),
            "carry_remaining":carry_forward_remaining(snap,shipment),"snapshot":snap,"shipment":shipment,
            "route":route,"world":config.world})
    X=pd.DataFrame(rows,columns=FEATURE_COLUMNS); validate_feature_names(X.columns)
    return X,meta

def _pipeline(estimator):
    numeric_columns = list(NUMERIC) + list(MISSINGNESS_COLUMNS)
    prep=ColumnTransformer([
        ("num",Pipeline([("impute",SimpleImputer(strategy="median",add_indicator=False,keep_empty_features=True))]),numeric_columns),
        ("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent",keep_empty_features=True)),
                          ("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),list(CATEGORICAL)),
    ],remainder="drop",verbose_feature_names_out=False)
    return Pipeline([("preprocess",prep),("model",estimator)])

def train_models(root=".", config_path="configs/simulation.yaml", routes_path=None,
                 artifact_dir="artifacts/models", seed=None):
    config=load_config(config_path); routes=tuple(load_routes(routes_path or config.routes.config_path).routes)
    seed=config.data.random_seed if seed is None else seed
    X,meta=load_training_frame(root,config,routes)
    groups=np.array([m["shipment_id"] for m in meta]); y=np.array([m["remaining_hours_actual"] for m in meta])
    labels=np.array([m["significant_delay_label"] for m in meta]); props=np.array([m["propagation_remaining"] for m in meta])
    split=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed)
    train_i,val_i=next(split.split(X,y,groups)); train_ids=set(groups[train_i]); val_ids=set(groups[val_i])
    if train_ids & val_ids: raise AssertionError("shipment overlap in split")
    direct=_pipeline(GradientBoostingRegressor(n_estimators=150,learning_rate=.05,max_depth=2,min_samples_leaf=8,random_state=seed))
    residual=_pipeline(GradientBoostingRegressor(n_estimators=150,learning_rate=.05,max_depth=2,min_samples_leaf=8,random_state=seed))
    classifier=_pipeline(GradientBoostingClassifier(n_estimators=120,learning_rate=.05,max_depth=2,min_samples_leaf=8,random_state=seed))
    direct.fit(X.iloc[train_i],y[train_i]); residual.fit(X.iloc[train_i],y[train_i]-props[train_i]); classifier.fit(X.iloc[train_i],labels[train_i])
    direct_raw=direct.predict(X.iloc[val_i]); residual_raw=residual.predict(X.iloc[val_i]); hybrid_raw=props[val_i]+residual_raw
    classifier_prob=classifier.predict_proba(X.iloc[val_i])[:,list(classifier.named_steps["model"].classes_).index(1)]
    pred_variants={"scheduled_eta":np.array([meta[i]["scheduled_remaining"] for i in val_i]),
        "carry_forward":np.array([meta[i]["carry_remaining"] for i in val_i]),
        "propagation_only":props[val_i],"ml_only":np.maximum(0,direct_raw),"hybrid":np.maximum(0,hybrid_raw)}
    report={"status":"preliminary validation; not a final generalization estimate","seed":int(seed),
        "training_shipments":len(train_ids),"validation_shipments":len(val_ids),
        "training_snapshots":len(train_i),"validation_snapshots":len(val_i),"feature_schema_version":FEATURE_SCHEMA_VERSION,
        "feature_names":list(FEATURE_COLUMNS),"threshold_hours":config.prediction.significant_delay_threshold_hours,
        "class_balance_by_checkpoint":{},"metrics_by_variant":{},"negative_ml_raw_count":int((direct_raw<0).sum()),
        "negative_residual_raw_count":int((residual_raw<0).sum()),"provisional_probability":True,
        "classification_probability_status":"uncalibrated","classification_threshold":CLASSIFICATION_THRESHOLD,
        "classification_metrics_by_checkpoint":{}}
    for cp in ("departure","port_arrival","customs_cleared"):
        idx=np.array([j for j,i in enumerate(val_i) if meta[i]["checkpoint_type"]==cp]); cp_labels=labels[val_i][idx]
        report["class_balance_by_checkpoint"][cp]={"n":int(len(idx)),"significant":int(cp_labels.sum()),"not_significant":int(len(idx)-cp_labels.sum())}
        report["classification_metrics_by_checkpoint"][cp]=_classification_metrics(cp_labels,classifier_prob[idx])
    for name,p in pred_variants.items():
        report["metrics_by_variant"][name]={}
        for cp in ("departure","port_arrival","customs_cleared"):
            idx=np.array([j for j,i in enumerate(val_i) if meta[i]["checkpoint_type"]==cp]); truth=y[val_i][idx]; pred=p[idx]
            report["metrics_by_variant"][name][cp]={"n":int(len(idx)),"mae_hours":float(mean_absolute_error(truth,pred)),"rmse_hours":float(mean_squared_error(truth,pred)**.5)}
    train_id_list=sorted(train_ids); validation_id_list=sorted(val_ids)
    digest=hashlib.sha256()
    for logical in ("shipments","prediction_snapshots","snapshot_targets"):
        path=next((Path(root)/folder/f"{logical}.{suffix}" for folder in ("data/raw","data/processed")
                   for suffix in ("parquet","csv") if (Path(root)/folder/f"{logical}.{suffix}").exists()),None)
        if path is None: raise FileNotFoundError(f"Missing training input table {logical}")
        digest.update(logical.encode()); digest.update(path.read_bytes())
    dataset_fingerprint=digest.hexdigest()
    model_config={"regressor":"GradientBoostingRegressor","n_estimators":150,"learning_rate":.05,"max_depth":2,"min_samples_leaf":8,
                  "classifier":"GradientBoostingClassifier","classifier_n_estimators":120,
                  "classifier_learning_rate":.05,"classifier_max_depth":2,"classifier_min_samples_leaf":8}
    bundle={"bundle_version":BUNDLE_VERSION,"direct":direct,"residual":residual,"classifier":classifier,"routes":routes,"world":config.world,
        "prediction_config":config.prediction,"seed":seed,"model_version":f"phase2-gbr-seed{seed}",
        "feature_schema_version":FEATURE_SCHEMA_VERSION,"feature_columns":FEATURE_COLUMNS,
        "python_version":platform.python_version(),"sklearn_version":sklearn.__version__,"joblib_version":joblib.__version__,
        "training_config":model_config,"random_seed":int(seed),"dataset_fingerprint_sha256":dataset_fingerprint,
        "training_shipment_ids":train_id_list,"validation_shipment_ids":validation_id_list,
        "trained_at_utc":datetime.now(timezone.utc).isoformat(),"training_shipments":len(train_ids)}
    out=Path(root)/artifact_dir; out.mkdir(parents=True,exist_ok=True)
    joblib.dump(bundle,out/"phase2_bundle.joblib")
    try:
        report["artifact"]=(out/"phase2_bundle.joblib").resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        # Never leak a machine-specific temp directory into a portable report.
        report["artifact"]="artifacts/models/phase2_bundle.joblib"
    report["training_timestamp_utc"]=bundle["trained_at_utc"]
    report["best_variant_by_checkpoint"] = {cp:min(report["metrics_by_variant"],key=lambda name:report["metrics_by_variant"][name][cp]["mae_hours"]) for cp in ("departure","port_arrival","customs_cleared")}
    report["dataset_fingerprint_sha256"]=dataset_fingerprint
    report["python_version"]=bundle["python_version"]; report["sklearn_version"]=bundle["sklearn_version"]; report["joblib_version"]=bundle["joblib_version"]
    (Path(root)/"reports/metrics").mkdir(parents=True,exist_ok=True)
    (Path(root)/"reports/metrics/phase2_preliminary.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    metadata={k:v for k,v in bundle.items() if k not in {"direct","residual","classifier","routes","world","prediction_config"}}
    (out/"phase2_metadata.json").write_text(json.dumps(metadata,indent=2,allow_nan=False),encoding="utf-8")
    return bundle,report

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-root",default="."); p.add_argument("--config",default="configs/simulation.yaml"); p.add_argument("--routes"); p.add_argument("--artifact-dir",default="artifacts/models"); a=p.parse_args()
    _,report=train_models(a.data_root,a.config,a.routes,a.artifact_dir); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
