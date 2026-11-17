"""Full deterministic Phase 3 workflow regression using the checked-in cohort."""
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
from aiqon.config.loader import load_config, load_routes
from aiqon.evaluation.phase3 import _outputs, run_phase3
from aiqon.evaluation.conformal import finite_sample_quantile, normalized_scores
from aiqon.evaluation.probability import normalized_signed_residuals
from aiqon.evaluation.splits import stratified_shipment_split, labels_by_shipment
from aiqon.prediction.service import PredictionService
from aiqon.prediction.training import _parse_snapshot, _read_table, load_training_frame
from aiqon.schemas.shipment import Shipment


def test_full_phase3_workflow_twice_is_reproducible_and_isolated(tmp_path):
    project = Path(__file__).resolve().parents[2]
    root = tmp_path / "cohort"
    for folder, files in (("data/raw", ("shipments", "events", "environment_actuals", "forecasts")),
                          ("data/processed", ("prediction_snapshots", "snapshot_targets"))):
        destination = root / folder
        destination.mkdir(parents=True)
        for name in files:
            shutil.copy2(project / folder / f"{name}.parquet", destination / f"{name}.parquet")

    config_path = project / "configs/simulation.yaml"
    routes_path = project / "configs/routes.yaml"
    config = load_config(config_path)
    routes = tuple(load_routes(routes_path).routes)
    phases2 = [project / "artifacts/models" / name for name in ("phase2_bundle.joblib", "phase2_metadata.json")]
    phase2_before = {path.name: path.read_bytes() for path in phases2}
    phase2_service = PredictionService.load(phases2[0])
    phase2_shipment = Shipment.model_validate(_read_table(root, "shipments").iloc[0].to_dict())
    phase2_snapshots = _read_table(root, "prediction_snapshots")
    phase2_snapshot_row = phase2_snapshots[
        (phase2_snapshots.shipment_id == phase2_shipment.shipment_id)
        & (phase2_snapshots.checkpoint_type == "departure")].iloc[0]
    phase2_snapshot = _parse_snapshot(phase2_snapshot_row)
    phase2_prediction_before = phase2_service.predict(phase2_snapshot, phase2_shipment).model_dump(mode="json")

    first = run_phase3(root, config_path, routes_path, "artifacts/models")
    first_manifest = json.loads((root / "reports/metrics/phase3_split_manifest.json").read_text())
    first_calibration = joblib.load(root / "artifacts/models/phase3_calibration.joblib")
    first_bundle = joblib.load(root / "artifacts/models/phase3_bundle.joblib")
    labels = labels_by_shipment(_read_table(root, "shipments"), _read_table(root, "snapshot_targets"))
    partitions = stratified_shipment_split(labels, config.evaluation.phase3.split_seed)
    X_test, meta_test = load_training_frame(root, config, routes, partitions["test"])
    first_outputs = _outputs(first_bundle, X_test, meta_test)
    X_cal, meta_cal = load_training_frame(root, config, routes, partitions["calibration"])
    cal_outputs = _outputs(first_bundle, X_cal, meta_cal)

    second = run_phase3(root, config_path, routes_path, "artifacts/models")
    second_manifest = json.loads((root / "reports/metrics/phase3_split_manifest.json").read_text())
    second_calibration = joblib.load(root / "artifacts/models/phase3_calibration.joblib")
    second_bundle = joblib.load(root / "artifacts/models/phase3_bundle.joblib")
    second_outputs = _outputs(second_bundle, X_test, meta_test)

    assert first["dataset"]["shipments"] == 300 and first["dataset"]["routes"] == 3
    assert first["dataset"]["snapshots"] == 900
    assert first_manifest == second_manifest
    assert first_manifest["partition_counts"] == {
        "fit": {"shipments": 210, "checkpoint_snapshots": 630, "significant_delay_shipments": 184,
                "not_significant_shipments": 26},
        "calibration": {"shipments": 45, "checkpoint_snapshots": 135, "significant_delay_shipments": 40,
                        "not_significant_shipments": 5},
        "test": {"shipments": 45, "checkpoint_snapshots": 135, "significant_delay_shipments": 40,
                 "not_significant_shipments": 5},
    }
    assert not (set(partitions["fit"]) & set(partitions["calibration"]) |
                set(partitions["fit"]) & set(partitions["test"]) |
                set(partitions["calibration"]) & set(partitions["test"]))
    assert all(ids == partitions["test"] for ids in first["evaluation_shipment_ids_by_variant"].values())
    assert set(first["regression_metrics_by_variant"]) == {
        "scheduled_eta", "carry_forward", "propagation_only", "ml_only", "hybrid"}
    assert all(set(row) == {"departure", "port_arrival", "customs_cleared"}
               for row in first["regression_metrics_by_variant"].values())
    assert first["bootstrap"] == second["bootstrap"]
    assert first["conformal"] == second["conformal"]
    assert first["significant_delay_probability"] == second["significant_delay_probability"]
    assert first_calibration["conformal_by_checkpoint"] == second_calibration["conformal_by_checkpoint"]
    for variant in first_outputs:
        np.testing.assert_array_equal(first_outputs[variant], second_outputs[variant])
    assert first_calibration["normalized_signed_calibration_residuals_by_checkpoint"].keys() == {
        "departure", "port_arrival", "customs_cleared"}
    for checkpoint in ("departure", "port_arrival", "customs_cleared"):
        cal_indices = [i for i, row in enumerate(meta_cal) if row["checkpoint_type"] == checkpoint]
        test_indices = [i for i, row in enumerate(meta_test) if row["checkpoint_type"] == checkpoint]
        actual_cal = [meta_cal[i]["remaining_hours_actual"] for i in cal_indices]
        prop_cal = [meta_cal[i]["propagation_remaining"] for i in cal_indices]
        predicted_cal = cal_outputs["hybrid"][cal_indices]
        residuals = normalized_signed_residuals(actual_cal, predicted_cal, prop_cal,
            config.evaluation.phase3.minimum_scale_hours)
        np.testing.assert_array_equal(
            first_calibration["normalized_signed_calibration_residuals_by_checkpoint"][checkpoint], residuals)
        scores, _ = normalized_scores(actual_cal, predicted_cal, prop_cal,
            config.evaluation.phase3.minimum_scale_hours)
        expected_q = finite_sample_quantile(scores, config.prediction.conformal_coverage)
        assert first_calibration["conformal_by_checkpoint"][checkpoint]["quantile"] == expected_q
        assert len(test_indices) == 45
    assert first["significant_delay_probability"]["formula"]["test_residuals_used"] is False
    assert first_bundle["fit_shipment_ids"] == partitions["fit"]
    assert first_bundle["calibration_shipment_ids"] == partitions["calibration"]
    assert first_bundle["test_shipment_ids"] == partitions["test"]
    assert first_bundle["experiment_fingerprint_sha256"] == first["dataset"]["experiment_fingerprint_sha256"]
    assert (root / "artifacts/models/phase3_metadata.json").exists()
    assert (root / "reports/metrics/phase3_prediction_driver_attribution.csv").exists()
    assert (root / "reports/metrics/phase3_comparison.csv").exists()
    assert all((root / path).exists() for path in first["artifacts"]["figures"])
    assert {path.name: path.read_bytes() for path in phases2} == phase2_before
    phase2_service_after = PredictionService.load(phases2[0])
    assert phase2_service_after.predict(phase2_snapshot, phase2_shipment).model_dump(mode="json") == phase2_prediction_before
