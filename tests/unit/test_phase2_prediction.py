import json
from pathlib import Path
import os, subprocess, sys
import pytest

from aiqon.config.loader import load_config, load_routes
from aiqon.prediction.features import FEATURE_COLUMNS, FORBIDDEN, validate_feature_names
from aiqon.prediction.training import train_models, _read_table, _parse_snapshot
from aiqon.prediction.service import PredictionService
from aiqon.prediction.replay import replay_shipment
from aiqon.schemas.shipment import Shipment


def test_feature_schema_has_no_target_columns():
    assert not set(FEATURE_COLUMNS) & FORBIDDEN
    with pytest.raises(ValueError):
        validate_feature_names([*FEATURE_COLUMNS, "actual_delay_hours"])


def test_asof_loader_uses_only_visible_snapshot_fields():
    config=load_config("configs/simulation.yaml")
    routes=load_routes(config.routes.config_path).routes
    rows=_read_table(".","prediction_snapshots")
    first=_parse_snapshot(rows.iloc[0])
    assert set(first.observed_event_times) == {"departure_at"}
    assert "actual_delivery_at" not in FEATURE_COLUMNS
    assert "actual_delay_hours" not in FEATURE_COLUMNS


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out=tmp_path_factory.mktemp("models")
    bundle,report=train_models(".",artifact_dir=str(out.relative_to(Path.cwd())) if out.is_relative_to(Path.cwd()) else str(out))
    # `train_models` resolves artifacts under data root; absolute path is supported by Path joining.
    return bundle,report,out


def test_train_variants_validate_grouped_split_and_save_load(trained):
    bundle,report,out=trained
    assert report["training_shipments"] + report["validation_shipments"] == 300
    assert set(report["metrics_by_variant"]) == {"scheduled_eta","carry_forward","propagation_only","ml_only","hybrid"}
    assert set(report["metrics_by_variant"]["hybrid"]) == {"departure","port_arrival","customs_cleared"}
    assert report["status"].startswith("preliminary")
    assert report["classification_probability_status"]=="uncalibrated"
    assert set(report["classification_metrics_by_checkpoint"])=={"departure","port_arrival","customs_cleared"}
    for metrics in report["classification_metrics_by_checkpoint"].values():
        assert metrics["validation_shipment_count"]==60
        assert metrics["positive_class_count"]+metrics["negative_class_count"]==60
        assert metrics["threshold"]==0.5
        assert len(metrics["confusion_matrix_labels_0_then_1"])==2
    # Fresh joblib load plus single and full replay arithmetic and history persistence.
    from aiqon.prediction.training import _read_table
    shipment=Shipment.model_validate(_read_table(".","shipments").iloc[0].to_dict())
    snap_frame=_read_table(".","prediction_snapshots")
    record=snap_frame[(snap_frame.shipment_id==shipment.shipment_id)&(snap_frame.checkpoint_type=="departure")].iloc[0]
    snapshot=_parse_snapshot(record)
    import joblib
    fresh=PredictionService(joblib.load(Path(out)/"phase2_bundle.joblib"))
    history=Path(out)/"history.jsonl"
    fresh.history_path=history
    pred=fresh.predict(snapshot,shipment)
    assert abs((pred.predicted_delivery_at-pred.prediction_created_at).total_seconds()/3600-pred.predicted_remaining_hours)<1e-9
    replay=replay_shipment(fresh,shipment.shipment_id)
    assert [r.checkpoint_type for r in replay]==["departure","port_arrival","customs_cleared"]
    assert len(history.read_text(encoding="utf-8").splitlines())==4
    assert 0 <= pred.probability_delay_over_12h <= 1
    assert fresh.predict(snapshot,shipment)==pred
    for variant in ("scheduled_eta","carry_forward","propagation_only","ml_only","hybrid"):
        assert fresh.predict(snapshot,shipment,variant).checkpoint_type=="departure"


def test_classifier_target_is_overall_delay_label():
    config=load_config("configs/simulation.yaml")
    targets=_read_table(".","snapshot_targets")
    assert set(targets.significant_delay_label.unique()) <= {True,False,0,1}
    assert targets.loc[targets.actual_delay_hours > config.prediction.significant_delay_threshold_hours,
                       "significant_delay_label"].astype(bool).all()


def test_single_class_classifier_metrics_are_explicitly_unavailable():
    from aiqon.prediction.training import _classification_metrics
    result=_classification_metrics([1,1,1],[.8,.7,.9])
    assert result["roc_auc"] is None and result["pr_auc_average_precision"] is None
    assert "only one class" in result["roc_auc_unavailable_reason"]
    assert result["brier_score"]>=0 and result["log_loss"]>=0


def test_negative_operational_duration_is_clamped_with_raw_diagnostic(trained):
    bundle,_,_=trained
    service=PredictionService(bundle)
    ships={r["shipment_id"]:Shipment.model_validate(r) for r in _read_table(".","shipments").to_dict("records")}
    shipment=ships["SHP-0001"]
    rows=_read_table(".","prediction_snapshots")
    snap=_parse_snapshot(rows[(rows.shipment_id==shipment.shipment_id)&(rows.checkpoint_type=="departure")].iloc[0])
    class Negative:
        def predict(self, X): return [-500.]
    service.bundle["residual"]=Negative()
    output,diagnostics=service.predict_with_diagnostics(snap,shipment,"hybrid")
    assert output.predicted_remaining_hours==0
    assert diagnostics["clamped"] is True
    assert diagnostics["raw_prediction_hours"] < 0


def test_fresh_process_reload_is_prediction_deterministic(trained):
    _,report,out=trained
    code = """import json,os; from aiqon.prediction.service import PredictionService; from aiqon.prediction.training import _read_table,_parse_snapshot; from aiqon.schemas.shipment import Shipment; ships={r['shipment_id']:Shipment.model_validate(r) for r in _read_table('.','shipments').to_dict('records')}; s=ships['SHP-0001']; f=_read_table('.','prediction_snapshots'); snap=_parse_snapshot(f[(f.shipment_id==s.shipment_id)&(f.checkpoint_type=='departure')].iloc[0]); print(json.dumps(PredictionService.load(os.environ['AIQON_MODEL']).predict(snap,s).model_dump(mode='json'),sort_keys=True))"""
    env=os.environ.copy(); env["PYTHONPATH"]=str(Path("src").resolve()); env["AIQON_MODEL"]=str(Path(out)/"phase2_bundle.joblib")
    first=subprocess.run([sys.executable,"-c",code],check=True,capture_output=True,text=True,env=env).stdout.strip()
    second=subprocess.run([sys.executable,"-c",code],check=True,capture_output=True,text=True,env=env).stdout.strip()
    assert json.loads(first)==json.loads(second)
