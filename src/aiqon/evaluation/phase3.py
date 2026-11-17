"""Complete leakage-safe Phase 3 evaluation and robustness workflow."""
from __future__ import annotations

import argparse
import json
import math
import platform
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor

from aiqon.config.loader import load_config, load_routes
from aiqon.data_generation.generator import _jsonable
from aiqon.prediction.baselines import carry_forward_remaining, scheduled_eta_remaining
from aiqon.prediction.features import FEATURE_COLUMNS, build_features
from aiqon.prediction.propagation import estimate_remaining_duration
from aiqon.prediction.training import _pipeline, _read_table, load_training_frame
from aiqon.simulation.engine import controlled_propagation_examples, interval_intersects_weekend_closure, simulate_cohort
from .bootstrap import shipment_bootstrap
from .conformal import finite_sample_quantile, normalized_scores, prediction_interval
from .masking import apply_feature_mask, mask_snapshot_inputs
from .metrics import probability_metrics, regression_metrics
from .probability import normalized_signed_residuals, significant_delay_probabilities
from .splits import (dataset_fingerprint, experiment_fingerprints, labels_by_shipment,
                     split_manifest, stratified_shipment_split)

UTC = timezone.utc
CHECKPOINTS = ("departure", "port_arrival", "customs_cleared")
VARIANTS = ("scheduled_eta", "carry_forward", "propagation_only", "ml_only", "hybrid")
RANDOM_MASK_VARIANT = "hybrid_random_masking_trained"


def _json_write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    def jsonable(item):
        if isinstance(item, np.generic):
            return item.item()
        return _jsonable(item)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False,
                               default=jsonable), encoding="utf-8")


def _outputs(models, X, meta):
    prop = np.asarray([row["propagation_remaining"] for row in meta], dtype=float)
    direct = np.maximum(0.0, models["direct"].predict(X))
    residual = models["residual"].predict(X)
    hybrid = np.maximum(0.0, prop + residual)
    return {"ml_only": direct, "hybrid": hybrid,
            "propagation_only": prop,
            "scheduled_eta": np.asarray([row["scheduled_remaining"] for row in meta], dtype=float),
            "carry_forward": np.asarray([row["carry_remaining"] for row in meta], dtype=float)}


def _model_settings(model_config):
    return {"n_estimators": int(model_config.n_estimators),
            "learning_rate": float(model_config.learning_rate),
            "max_depth": int(model_config.max_depth),
            "min_samples_leaf": int(model_config.min_samples_leaf),
            "random_state": int(model_config.random_seed),
            "random_seed": int(model_config.random_seed)}


def _fit_models(X_fit, meta_fit, model_config):
    y = np.asarray([row["remaining_hours_actual"] for row in meta_fit], dtype=float)
    prop = np.asarray([row["propagation_remaining"] for row in meta_fit], dtype=float)
    settings = _model_settings(model_config)
    estimator_settings = {k: v for k, v in settings.items() if k != "random_seed"}
    direct = _pipeline(GradientBoostingRegressor(**estimator_settings))
    residual = _pipeline(GradientBoostingRegressor(**estimator_settings))
    direct.fit(X_fit, y)
    residual.fit(X_fit, y - prop)
    return {"direct": direct, "residual": residual}, settings


def _fit_random_mask_model(X_fit, residual_target, model_settings):
    settings = {k: v for k, v in model_settings.items() if k != "random_seed"}
    estimator = _pipeline(GradientBoostingRegressor(**settings))
    estimator.fit(X_fit, residual_target)
    return estimator


def _long_prediction_frame(X, meta, outputs):
    rows = []
    for index, row in enumerate(meta):
        item = {"shipment_id": row["shipment_id"], "checkpoint_type": row["checkpoint_type"],
                "actual": row["remaining_hours_actual"], "label": row["significant_delay_label"],
                "propagation": row["propagation_remaining"],
                "checkpoint_at": row["snapshot"].checkpoint_at,
                "scheduled_delivery_at": row["shipment"].scheduled_delivery_at}
        item.update({name: float(values[index]) for name, values in outputs.items()})
        rows.append(item)
    return pd.DataFrame(rows)


def _metrics_by_checkpoint(frame):
    output = {variant: {} for variant in VARIANTS}
    comparisons = []
    for cp in CHECKPOINTS:
        part = frame[frame.checkpoint_type == cp]
        for variant in VARIANTS:
            metrics = regression_metrics(part.actual, part[variant], part.carry_forward)
            output[variant][cp] = metrics
            comparisons.append({"variant": variant, "checkpoint_type": cp, **metrics})
    return output, comparisons


def _conformal_calibration(cal_frame, test_frame, coverage, min_scale, prediction_key="hybrid"):
    calibrators, report, test_intervals = {}, {}, {}
    for cp in CHECKPOINTS:
        cal = cal_frame[cal_frame.checkpoint_type == cp]
        test = test_frame[test_frame.checkpoint_type == cp]
        scores, _ = normalized_scores(cal.actual, cal[prediction_key], cal.propagation, min_scale)
        q = finite_sample_quantile(scores, coverage)
        lower, upper = prediction_interval(test[prediction_key], test.propagation, q, min_scale)
        finite = math.isfinite(q)
        covered = (test.actual.to_numpy() >= lower) & (test.actual.to_numpy() <= upper)
        widths = upper - lower
        report[cp] = {
            "calibration_sample_count": int(len(cal)), "test_sample_count": int(len(test)),
            "nominal_coverage": float(coverage), "finite_sample_rank": int(math.ceil((len(cal) + 1) * coverage)),
            "quantile": float(q) if finite else None, "quantile_is_infinite": not finite,
            "empirical_coverage": float(covered.mean()),
            "mean_interval_width_hours": float(np.mean(widths)) if np.isfinite(widths).all() else None,
            "median_interval_width_hours": float(np.median(widths)) if np.isfinite(widths).all() else None,
            "coverage_by_scenario_family": {"synthetic_baseline": {"n": int(len(test)),
                "empirical_coverage": float(covered.mean())}},
        }
        calibrators[cp] = {"quantile": q, "calibration_sample_count": len(cal),
                           "coverage": coverage, "minimum_scale_hours": min_scale}
        test_intervals[cp] = {"lower": lower, "upper": upper}
    return calibrators, report, test_intervals


def _probability_evaluation(cal_frame, test_frame, threshold, min_scale, delay_threshold):
    residuals, predictions, report = {}, {}, {}
    for cp in CHECKPOINTS:
        cal = cal_frame[cal_frame.checkpoint_type == cp]
        test = test_frame[test_frame.checkpoint_type == cp]
        r = normalized_signed_residuals(cal.actual, cal.hybrid, cal.propagation, min_scale)
        p = significant_delay_probabilities(
            test.hybrid, test.propagation, test.checkpoint_at, test.scheduled_delivery_at,
            r, delay_threshold, min_scale,
        )
        residuals[cp] = r
        predictions[cp] = p
        report[cp] = probability_metrics(test.label, p, threshold)
        report[cp]["calibration_sample_count"] = int(len(cal))
        report[cp]["calibration_signed_residuals_mean"] = float(np.mean(r))
        report[cp]["calibration_signed_residuals_median"] = float(np.median(r))
        report[cp]["significant_delay_threshold_hours"] = float(delay_threshold)
    return residuals, predictions, report


def _prediction_driver_attribution(test_X, test_meta, references):
    """Propagation-only forecast sensitivity. This function has no oracle input."""
    rows = []
    for index, meta in enumerate(test_meta):
        if set(meta) - {"snapshot", "shipment", "route", "world"}:
            raise ValueError("Prediction driver attribution accepts as-of inputs only")
        snapshot, shipment = meta["snapshot"], meta["shipment"]
        route = meta["route"]
        trace = {}
        original, _ = estimate_remaining_duration(snapshot, route, meta["world"], forecast_trace=trace)
        for variable, reference in references.items():
            selected = trace.get(variable)
            base = {"shipment_id": shipment.shipment_id, "checkpoint": snapshot.checkpoint_type,
                    "component": variable, "forecast_used": False,
                    "forecast_issue_time": None, "forecast_valid_time": None, "forecast_location": None,
                    "original_forecast_value": None, "neutral_reference_value": reference,
                    "original_propagation_remaining": max(0.0, original),
                    "neutralized_propagation_remaining": None, "effect_hours": None,
                    "unavailable_reason": None}
            if selected is None:
                base["unavailable_reason"] = "unsupported_not_consumed_by_propagation"
                rows.append(base)
                continue
            forecast = selected["forecast"]
            if forecast is None:
                base["unavailable_reason"] = f"no_eligible_forecast; propagation_used_{selected['source']}"
                base["forecast_location"] = selected["location_id"]
                base["forecast_requested_at"] = selected["requested_at"]
                rows.append(base)
                continue
            if forecast.forecast_issued_at > snapshot.checkpoint_at:
                raise ValueError("selected forecast was issued after the prediction checkpoint")
            changed = _neutralize_selected_forecast(snapshot, forecast, reference)
            neutralized, _ = estimate_remaining_duration(changed, route, meta["world"])
            base.update({"forecast_used": True, "forecast_issue_time": forecast.forecast_issued_at,
                         "forecast_valid_time": forecast.forecast_valid_at,
                         "forecast_location": forecast.location_id,
                         "original_forecast_value": forecast.forecast_value,
                         "neutralized_propagation_remaining": max(0.0, neutralized),
                         "effect_hours": max(0.0, original) - max(0.0, neutralized),
                         "forecast_requested_at": selected["requested_at"]})
            rows.append(base)
    return rows


def _neutralize_selected_forecast(snapshot, selected_forecast, reference_value):
    """Copy the as-of snapshot and change only the exact selected vintage's value."""
    selected_index = next((i for i, item in enumerate(snapshot.forecasts)
                           if item is selected_forecast), None)
    if selected_index is None:
        raise ValueError("selected forecast vintage does not belong to the input snapshot")
    forecasts = list(snapshot.forecasts)
    forecasts[selected_index] = selected_forecast.model_copy(update={"forecast_value": reference_value})
    return snapshot.model_copy(update={"forecasts": tuple(forecasts)})


def _masked_partition(meta, columns, *, row_columns=None):
    """Rebuild both ML features and propagation from copied, masked snapshots."""
    rows, masked_meta = [], []
    for index, item in enumerate(meta):
        selected = tuple(row_columns[index]) if row_columns is not None else tuple(columns)
        snapshot = mask_snapshot_inputs(item["snapshot"], selected) if selected else item["snapshot"]
        features = build_features(snapshot, item["shipment"], item["route"], item["world"])
        feature_frame = pd.DataFrame([features], columns=FEATURE_COLUMNS)
        if selected:
            feature_frame = apply_feature_mask(feature_frame, selected)
        rows.append(feature_frame.iloc[0].to_dict())
        updated = dict(item)
        updated["snapshot"] = snapshot
        updated["propagation_remaining"] = max(0.0, estimate_remaining_duration(
            snapshot, item["route"], item["world"])[0])
        masked_meta.append(updated)
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS), masked_meta


def _random_mask_columns(frame, columns, probability, seed):
    """Choose deterministic row-level source masks from values actually present."""
    rng = np.random.default_rng(seed)
    selected_rows = []
    for _, row in frame.iterrows():
        selected_rows.append(tuple(column for column in columns
                                   if pd.notna(row[column]) and rng.random() < probability))
    return selected_rows


def _retrospective_attribution(root, config, routes, fingerprint, seed,
                              timestamp_tolerance_seconds=1e-6):
    event_frame = _read_table(root, "events")
    event_frame["actual_at"] = pd.to_datetime(event_frame["actual_at"], utc=True)
    shipments_frame = _read_table(root, "shipments")
    shipment_map = {str(r["shipment_id"]): r for r in shipments_frame.to_dict("records")}
    route_map = {route.route_id: route for route in routes}
    events_by_id = {sid: group.sort_values("actual_at") for sid, group in event_frame.groupby("shipment_id")}
    candidates = defaultdict(list)
    for sid, events in events_by_id.items():
        ev = {str(row.event_type): row for row in events.itertuples(index=False)}
        route = route_map[shipment_map[str(sid)]["route_id"]]
        if float(getattr(ev.get("berth_start"), "slot_wait_hours", 0) or 0) > 0:
            candidates["berth_slot"].append(str(sid))
        if float(getattr(ev.get("berth_start"), "queue_wait_hours", 0) or 0) > 0:
            candidates["vessel_bunching"].append(str(sid))
        port_complete, customs_start = ev["port_complete"].actual_at, ev["customs_start"].actual_at
        if interval_intersects_weekend_closure(port_complete, customs_start, route):
            candidates["weekend_customs"].append(str(sid))
        if float(getattr(ev.get("gate_out"), "gate_wait_hours", 0) or 0) > 0:
            candidates["gate_cutoff"].append(str(sid))
    id_to_mechanism = {}
    for mechanism in ("berth_slot", "vessel_bunching", "weekend_customs", "gate_cutoff"):
        if candidates[mechanism]:
            sid = candidates[mechanism][0]
            id_to_mechanism.setdefault(sid, []).append(mechanism)
    if config.data.random_seed != seed:
        raise ValueError("Counterfactual seed must match the original dataset seed")
    stored_delivery = {
        sid: events[events.event_type == "delivery"].iloc[0].actual_at
        for sid, events in events_by_id.items() if sid in id_to_mechanism
    }
    original_rerun = simulate_cohort(config, routes)
    rerun_delivery = {sid: event.actual_at for sid in id_to_mechanism
                      for event in original_rerun.events
                      if event.shipment_id == sid and event.event_type == "delivery"}
    deltas = {}
    for sid, stored_at in stored_delivery.items():
        if sid not in rerun_delivery:
            raise ValueError(f"Baseline simulator rerun omitted representative shipment {sid}")
        delta = abs((rerun_delivery[sid] - stored_at).total_seconds())
        deltas[sid] = delta
        if delta > timestamp_tolerance_seconds:
            raise ValueError(
                f"Counterfactual baseline mismatch for {sid}: stored={stored_at.isoformat()}, "
                f"rerun={rerun_delivery[sid].isoformat()}, delta_seconds={delta}, "
                f"tolerance_seconds={timestamp_tolerance_seconds}"
            )
    rows = []
    for sid, mechanisms in id_to_mechanism.items():
        events = events_by_id[sid]
        delivery = events[events.event_type == "delivery"].iloc[0].actual_at
        for mechanism in mechanisms:
            counterfactual = simulate_cohort(config, routes, neutralize_mechanism=mechanism)
            counterfactual_delivery = next(e.actual_at for e in counterfactual.events
                                           if e.shipment_id == sid and e.event_type == "delivery")
            effect = (delivery - counterfactual_delivery).total_seconds() / 3600
            rows.append({"label": "Retrospective cause attribution", "shipment_id": sid,
                         "mechanism": mechanism, "realized": True,
                         "neutralization_semantics": {
                             "berth_slot": "remove slot windows; preserve full-cohort berth service and queue competition",
                             "vessel_bunching": "set route-global berth capacity to the full route cohort; preserve berth slots and service draws",
                             "weekend_customs": "make weekend days open; preserve daily customs hours and processing duration",
                             "gate_cutoff": "dispatch at customs completion; preserve processing and inland travel",
                         }[mechanism],
                         "matched_random_seed": int(config.data.random_seed),
                         "baseline_rerun_delivery_at": rerun_delivery[sid],
                         "baseline_equivalence_delta_seconds": deltas[sid],
                         "baseline_timestamp_tolerance_seconds": timestamp_tolerance_seconds,
                         "dataset_fingerprint_sha256": fingerprint,
                         "original_actual_delivery_at": delivery,
                         "counterfactual_actual_delivery_at": counterfactual_delivery,
                         "retrospective_cause_effect_hours": effect})
    return {"label": "Retrospective cause attribution", "method": "verify original full-cohort rerun against stored outcomes, then rerun one realized mechanism at a time",
            "representative_shipments": int(len(id_to_mechanism)),
            "baseline_equivalence": {"status": "passed", "checked_shipments": int(len(deltas)),
                "timestamp_tolerance_seconds": float(timestamp_tolerance_seconds),
                "maximum_absolute_delta_seconds": max(deltas.values(), default=0.0),
                "matched_seed": int(config.data.random_seed)},
            "results": rows,
            "limitations": ["One-at-a-time effects can interact and are not additive.",
                            "Effects describe this synthetic simulator, not real-world causal effects."]}


def _amplification_experiments(routes, world):
    route = next((r for r in routes if r.berth_capacity == 1 and 0 in r.berth_slot_hours and 12 in r.berth_slot_hours), routes[0])
    examples = controlled_propagation_examples(route, world)
    for name, row in examples.items():
        row["experimental_design"] = "paired route berth schedule with common shipment attributes, service duration and downstream rules"
        row["route_id"] = route.route_id
        row["injected_delay_hours_positive"] = row["injected_transit_delay_hours"] > 0
        if not row["injected_delay_hours_positive"]:
            raise ValueError("Amplification experiment requires a positive injected upstream delay")
    return {"seed": None, "reproducible_inputs": examples, "results": examples,
            "interpretation": {"below_one": "delay absorption/recovery", "near_one": "simple carry-through",
                               "above_one": "delay amplification"},
            "limitation": "Controlled synthetic sensitivity experiment, not a real-world causal estimate."}


def _stress_test(root, config, routes, models, seed_base, stress_cfg):
    """Run separate, small simulator cohorts; no outcomes are used for fitting."""
    results, scenarios = [], []
    settings = [
        ("severe_transit_plus_peak_congestion", stress_cfg.stress_transit_delay_hours,
         {"congestion": stress_cfg.stress_peak_congestion}, None, None),
        ("transit_delay_plus_weekend_customs", stress_cfg.stress_weekend_injected_transit_delay_hours,
         {"scheduled_transit_hours": stress_cfg.stress_weekend_scheduled_transit_hours},
         stress_cfg.stress_weekend_start_at_utc, 1),
        ("vessel_bunching_plus_capacity_reduction", 0.0,
         {"berth_capacity": stress_cfg.stress_reduced_berth_capacity}, None, None),
        ("customs_delay_plus_missed_gate_cutoff", 0.0,
         {"customs_processing_multiplier": stress_cfg.stress_customs_processing_multiplier,
          "gate_cutoff_hour_utc": stress_cfg.stress_gate_cutoff_hour_utc}, None, None),
    ]
    for offset, (name, injection, changes, start_at, window_days) in enumerate(settings):
        scenario_routes = []
        for route in routes:
            values = {}
            if "congestion" in changes:
                values["congestion"] = route.congestion.model_copy(update={"baseline": changes["congestion"]})
            if "scheduled_transit_hours" in changes:
                values["scheduled_transit_hours"] = changes["scheduled_transit_hours"]
            if "berth_capacity" in changes:
                values["berth_capacity"] = min(route.berth_capacity, changes["berth_capacity"])
            if "customs_processing_multiplier" in changes:
                values["customs"] = route.customs.model_copy(update={"base_processing_hours":
                    route.customs.base_processing_hours * changes["customs_processing_multiplier"]})
            if "gate_cutoff_hour_utc" in changes:
                values["inland"] = route.inland.model_copy(update={"gate_cutoff_hour_utc": changes["gate_cutoff_hour_utc"]})
            scenario_routes.append(route.model_copy(update=values) if values else route)
        world_changes = {}
        if start_at is not None:
            world_changes["start_at_utc"] = start_at
        data = config.data.model_copy(update={"n_shipments": stress_cfg.stress_shipment_count,
                                               "random_seed": seed_base + offset})
        world = config.world.model_copy(update={**world_changes,
            "departure_window_days": window_days or config.world.departure_window_days,
            "transit_perturbation_hours": (stress_cfg.stress_weekend_transit_perturbation_hours
                if name == "transit_delay_plus_weekend_customs" else config.world.transit_perturbation_hours)})
        scenario_config = config.model_copy(update={"data": data, "world": world})
        simulation = simulate_cohort(scenario_config, scenario_routes, injected_transit_delay_hours=injection)
        route_map = {r.route_id: r for r in scenario_routes}
        ship_map = {s.shipment_id: s for s in simulation.shipments}
        target_map = {(x["shipment_id"], x["checkpoint_type"]): x for x in simulation.snapshot_targets}
        metrics_map = {x["shipment_id"]: x for x in simulation.journey_metrics}
        scenario_rows = []
        for snapshot in simulation.snapshots:
            shipment = ship_map[snapshot.shipment_id]
            route = route_map[shipment.route_id]
            features = build_features(snapshot, shipment, route, scenario_config.world)
            X = pd.DataFrame([features], columns=FEATURE_COLUMNS)
            meta = [{"propagation_remaining": max(0.0, estimate_remaining_duration(snapshot, route, scenario_config.world)[0])}]
            pred = _outputs(models, X, [{**meta[0], "snapshot": snapshot, "shipment": shipment,
                                         "checkpoint_type": snapshot.checkpoint_type,
                                         "remaining_hours_actual": 0.0, "significant_delay_label": 0,
                                         "scheduled_remaining": scheduled_eta_remaining(snapshot, shipment),
                                         "carry_remaining": carry_forward_remaining(snapshot, shipment)}])["hybrid"][0]
            truth = target_map[(shipment.shipment_id, snapshot.checkpoint_type)]["remaining_hours_actual"]
            row = {"scenario_family": name, "scenario_id": scenario_config.data.random_seed,
                   "shipment_id": shipment.shipment_id, "checkpoint_type": snapshot.checkpoint_type,
                   "predicted_remaining_hours": float(pred), "actual_remaining_hours": float(truth),
                   "absolute_error_hours": float(abs(pred - truth)),
                   "actual_delay_hours": float(metrics_map[shipment.shipment_id]["total_delay_hours"]),
                   "scenario_configuration": {"random_seed": scenario_config.data.random_seed,
                       "injected_transit_delay_hours": injection, "route_parameter_overrides": changes,
                       "shipment_count": len(simulation.shipments)}}
            results.append(row); scenario_rows.append(row)
        scenarios.append({"scenario_family": name, "scenario_configuration": scenario_rows[0]["scenario_configuration"],
                          "shipment_count": len(simulation.shipments),
                          "checkpoint_count": len(scenario_rows),
                          "weekend_customs_affected_journeys": int(sum(x["weekend_customs_closure_affected"] for x in simulation.journey_metrics)),
                          "mean_absolute_error_hours_by_checkpoint": {
                              cp: float(np.mean([r["absolute_error_hours"] for r in scenario_rows if r["checkpoint_type"] == cp]))
                              for cp in CHECKPOINTS}})
    return {"scenario_families": scenarios, "prediction_rows": results,
            "unrepresentable_scenarios": [{"scenario_family": "weather_disruption_plus_reduced_recovery_capacity",
                "reason": "The Phase 1 simulator records weather but has no actual weather-dependent transit or recovery mechanism. Adding one would change Phase 1 ground-truth behavior."}]}


def _make_plots(root, comparisons, test_frame, conformal_report, probability_report,
                probabilities, shap_values, shap_names, amplification, missingness_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    folder = Path(root) / "reports" / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    # Baseline comparison.
    fig, ax = plt.subplots(figsize=(10, 5))
    for variant in VARIANTS:
        rows = [x for x in comparisons if x["variant"] == variant]
        ax.plot(CHECKPOINTS, [x["mae_hours"] for x in rows], marker="o", label=variant)
    ax.set_ylabel("MAE (hours)"); ax.set_title("Phase 3 regression error by checkpoint")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(folder / "phase3_baseline_comparison.png", dpi=140); plt.close(fig)
    # Prediction error distribution.
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in ("carry_forward", "propagation_only", "ml_only", "hybrid"):
        ax.hist(test_frame[variant] - test_frame.actual, bins=25, alpha=.4, label=variant)
    ax.set_xlabel("Predicted minus actual remaining hours"); ax.set_ylabel("Snapshot count")
    ax.set_title("Held-out prediction error distribution"); ax.legend(); fig.tight_layout()
    fig.savefig(folder / "phase3_prediction_error_distribution.png", dpi=140); plt.close(fig)
    # Conformal coverage.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(CHECKPOINTS, [conformal_report[cp]["empirical_coverage"] for cp in CHECKPOINTS])
    ax.axhline(next(iter(conformal_report.values()))["nominal_coverage"], color="red", linestyle="--", label="nominal")
    ax.set_ylim(0, 1); ax.set_ylabel("Coverage"); ax.set_title("Conformal coverage by checkpoint"); ax.legend()
    fig.tight_layout(); fig.savefig(folder / "phase3_conformal_coverage.png", dpi=140); plt.close(fig)
    # Probability calibration curve.
    fig, ax = plt.subplots(figsize=(6, 5)); ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    for cp in CHECKPOINTS:
        part = test_frame[test_frame.checkpoint_type == cp]
        p = probabilities[cp]
        order = np.argsort(p); chunks = np.array_split(order, min(8, max(1, len(order))))
        mean_p = [float(np.mean(p[idx])) for idx in chunks if len(idx)]
        mean_y = [float(part.label.to_numpy()[idx].mean()) for idx in chunks if len(idx)]
        ax.plot(mean_p, mean_y, marker="o", label=cp)
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title("Derived significant-delay probability calibration"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(folder / "phase3_probability_calibration.png", dpi=140); plt.close(fig)
    # SHAP residual summary, transformed with the fitted preprocessing.
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, features=None, feature_names=shap_names, plot_type="bar", show=False, max_display=20)
    plt.title("SHAP: ML residual feature contribution")
    plt.gca().set_xlabel("Mean absolute SHAP value (hours)")
    plt.tight_layout(); plt.savefig(folder / "phase3_shap_global_summary.png", dpi=140, bbox_inches="tight"); plt.close()
    # Amplification comparison.
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(amplification["results"])
    values = [amplification["results"][n]["amplification_factor"] for n in names]
    ax.bar([n.replace("_", "\n") for n in names], values); ax.axhline(1, color="red", linestyle="--")
    ax.set_ylabel("Amplification factor"); ax.set_title("Quiet and busy port amplification")
    fig.tight_layout(); fig.savefig(folder / "phase3_amplification_quiet_busy.png", dpi=140); plt.close(fig)
    # Missing data robustness.
    rows = [r for r in missingness_rows if r["checkpoint_type"] == "port_arrival"
            and r.get("mae_hours") is not None]
    port_complete = next(r["mae_hours"] for r in rows if r["scenario"] == "original_complete")
    original_masked = [r["mae_hours"] for r in rows if r["scenario"].startswith("original_model_masked:")]
    masked_trained_complete = next(r["mae_hours"] for r in rows if r["scenario"] == "random_masking_model_complete")
    masked_trained_masked = [r["mae_hours"] for r in rows if r["scenario"].startswith("random_masking_model_masked:")]
    labels = ["Original model\ncomplete", "Original model\nmasked (mean)",
              "Random-mask trained\ncomplete", "Random-mask trained\nmasked (mean)"]
    values = [port_complete, float(np.mean(original_masked)), masked_trained_complete,
              float(np.mean(masked_trained_masked))]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, values)
    ax.set_ylabel("MAE (hours)"); ax.set_title("Missing data robustness: port arrival\nmasked bars average configured applicable scenarios")
    fig.tight_layout()
    fig.savefig(folder / "phase3_missing_data_robustness.png", dpi=140); plt.close(fig)
    return [str(path.relative_to(root)) for path in sorted(folder.glob("phase3_*.png"))]


def run_phase3(data_root=".", config_path="configs/simulation.yaml", routes_path=None,
               artifact_dir="artifacts/models") -> dict:
    root = Path(data_root).resolve()
    config = load_config(config_path)
    effective_routes_path = routes_path or config.routes.config_path
    routes = tuple(load_routes(effective_routes_path).routes)
    if len(routes) != 3:
        raise ValueError("Phase 3 expects the existing three route profiles")
    p3 = config.evaluation.phase3
    fingerprint = dataset_fingerprint(root)
    model_configuration = _model_settings(p3.model)
    provenance = experiment_fingerprints(root, config_path, effective_routes_path,
                                         model_configuration, config.data.random_seed)
    # Read only shipment IDs and shipment-level labels before constructing features.
    shipment_table = _read_table(root, "shipments")
    target_table = _read_table(root, "snapshot_targets")
    labels = labels_by_shipment(shipment_table, target_table)
    partitions = stratified_shipment_split(labels, p3.split_seed,
        (p3.fit_fraction, p3.calibration_fraction, p3.test_fraction))
    split_cfg = {"fit_fraction": p3.fit_fraction, "calibration_fraction": p3.calibration_fraction,
                 "test_fraction": p3.test_fraction, "stratification": "significant-delay label at shipment level",
                 "split_algorithm": "two-stage sklearn train_test_split with exact integer partition sizes"}
    manifest = split_manifest(partitions, labels, fingerprint, p3.split_seed, split_cfg, provenance)
    split_path = root / "reports" / "metrics" / "phase3_split_manifest.json"
    _json_write(split_path, manifest)

    # Each feature partition is materialized only after its shipment IDs are fixed.
    X_fit, meta_fit = load_training_frame(root, config, routes, partitions["fit"])
    X_cal, meta_cal = load_training_frame(root, config, routes, partitions["calibration"])
    X_test, meta_test = load_training_frame(root, config, routes, partitions["test"])
    if (len(X_fit), len(X_cal), len(X_test)) != (630, 135, 135):
        raise ValueError("Each Phase 3 partition must contain exactly three checkpoint rows per shipment")
    seed = int(config.data.random_seed)
    models, model_settings = _fit_models(X_fit, meta_fit, p3.model)
    cal_predictions = _outputs(models, X_cal, meta_cal)
    test_predictions = _outputs(models, X_test, meta_test)
    cal_frame = _long_prediction_frame(X_cal, meta_cal, cal_predictions)
    test_frame = _long_prediction_frame(X_test, meta_test, test_predictions)
    regression_report, comparison_rows = _metrics_by_checkpoint(test_frame)
    aggregate_report = {variant: regression_metrics(test_frame.actual, test_frame[variant],
        test_frame.carry_forward) for variant in VARIANTS}
    best_report = {cp: min(VARIANTS, key=lambda v: regression_report[v][cp]["mae_hours"])
                   for cp in CHECKPOINTS}

    coverage = config.prediction.conformal_coverage
    conformal, conformal_report, intervals = _conformal_calibration(
        cal_frame, test_frame, coverage, p3.minimum_scale_hours)
    signed_residuals, probabilities, probability_report = _probability_evaluation(
        cal_frame, test_frame, p3.probability_threshold, p3.minimum_scale_hours,
        config.prediction.significant_delay_threshold_hours)
    for cp in CHECKPOINTS:
        probability_report[cp]["conformal_interval_is_not_probability_distribution"] = True

    bootstrap_report, _ = shipment_bootstrap(test_frame, VARIANTS, CHECKPOINTS,
        config.evaluation.bootstrap_samples, p3.split_seed)

    # Training-time random masking uses copied snapshots and recomputed baseline values.
    random_columns = ("actual_transit_hours", "observed_port_congestion", "actual_customs_hours",
                      "observed_customs_workload", "observed_inland_traffic", "forecast_port_congestion",
                      "forecast_customs_workload", "forecast_inland_traffic")
    random_fit_masks = _random_mask_columns(X_fit, random_columns, p3.random_mask_probability, p3.split_seed)
    X_fit_random, meta_fit_random = _masked_partition(meta_fit, (), row_columns=random_fit_masks)
    fit_residual_target = np.asarray([m["remaining_hours_actual"] - m["propagation_remaining"]
                                      for m in meta_fit_random])
    random_model = _fit_random_mask_model(X_fit_random, fit_residual_target, model_settings)
    random_cal_raw = np.maximum(0.0, np.asarray([m["propagation_remaining"] for m in meta_cal]) + random_model.predict(X_cal))
    random_test_raw = np.maximum(0.0, np.asarray([m["propagation_remaining"] for m in meta_test]) + random_model.predict(X_test))
    random_cal_frame = _long_prediction_frame(X_cal, meta_cal, {"hybrid_random": random_cal_raw})
    random_test_frame = _long_prediction_frame(X_test, meta_test, {"hybrid_random": random_test_raw})
    random_conformal, random_conformal_report, _ = _conformal_calibration(
        random_cal_frame.rename(columns={"hybrid_random": "hybrid"}),
        random_test_frame.rename(columns={"hybrid_random": "hybrid"}), coverage, p3.minimum_scale_hours)
    missingness_rows = []
    for cp in CHECKPOINTS:
        idx = [i for i, m in enumerate(meta_test) if m["checkpoint_type"] == cp]
        actual = np.asarray([meta_test[i]["remaining_hours_actual"] for i in idx])
        for name, prediction, calibrator, baseline_mae in (
                ("original_complete", test_predictions["hybrid"], conformal,
                 regression_report["hybrid"][cp]["mae_hours"]),
                ("random_masking_model_complete", random_test_raw, random_conformal,
                 float(np.mean(np.abs(random_test_raw[idx] - actual))))):
            q = calibrator[cp]["quantile"]
            lower, upper = prediction_interval(prediction[idx],
                [meta_test[i]["propagation_remaining"] for i in idx], q, p3.minimum_scale_hours)
            widths = upper - lower
            mae = float(np.mean(np.abs(prediction[idx] - actual)))
            missingness_rows.append({"scenario": name, "checkpoint_type": cp,
                "applicable_sample_count": len(idx), "mae_hours": mae,
                "rmse_hours": float(np.mean((prediction[idx] - actual) ** 2) ** .5),
                "conformal_coverage": float(np.mean((actual >= lower) & (actual <= upper))),
                "mean_interval_width_hours": float(np.mean(widths)) if np.isfinite(widths).all() else None,
                "degradation_vs_complete_mae_hours": mae - baseline_mae,
                "status": "reference" if name == "original_complete" else "evaluated"})

    for name, cols in p3.missingness_scenarios.items():
        X_cal_masked, meta_cal_masked = _masked_partition(meta_cal, cols)
        X_test_masked, meta_test_masked = _masked_partition(meta_test, cols)
        original_cal = _outputs(models, X_cal_masked, meta_cal_masked)["hybrid"]
        original_test = _outputs(models, X_test_masked, meta_test_masked)["hybrid"]
        random_cal_masked = np.maximum(0.0, np.asarray([m["propagation_remaining"] for m in meta_cal_masked])
                                        + random_model.predict(X_cal_masked))
        random_test_masked = np.maximum(0.0, np.asarray([m["propagation_remaining"] for m in meta_test_masked])
                                         + random_model.predict(X_test_masked))
        cal_original_frame = _long_prediction_frame(X_cal_masked, meta_cal_masked, {"hybrid": original_cal})
        test_original_frame = _long_prediction_frame(X_test_masked, meta_test_masked, {"hybrid": original_test})
        original_masked_conformal, _, _ = _conformal_calibration(
            cal_original_frame, test_original_frame, coverage, p3.minimum_scale_hours)
        cal_random_frame = _long_prediction_frame(X_cal_masked, meta_cal_masked, {"hybrid": random_cal_masked})
        test_random_frame = _long_prediction_frame(X_test_masked, meta_test_masked, {"hybrid": random_test_masked})
        random_masked_conformal, _, _ = _conformal_calibration(
            cal_random_frame, test_random_frame, coverage, p3.minimum_scale_hours)
        for cp in CHECKPOINTS:
            all_idx = [i for i, m in enumerate(meta_test) if m["checkpoint_type"] == cp]
            applicable = [i for i in all_idx if any(pd.notna(X_test.iloc[i][column]) for column in cols)]
            if not applicable:
                for model_name in ("original_model_masked", "random_masking_model_masked"):
                    missingness_rows.append({"scenario": f"{model_name}:{name}", "checkpoint_type": cp,
                        "applicable_sample_count": 0, "mae_hours": None, "rmse_hours": None,
                        "conformal_coverage": None, "mean_interval_width_hours": None,
                        "degradation_vs_complete_mae_hours": None, "status": "not applicable"})
                continue
            actual = np.asarray([meta_test[i]["remaining_hours_actual"] for i in applicable])
            for model_name, prediction, masked_meta, calibrator, complete_name in (
                    ("original_model_masked", original_test, meta_test_masked,
                     original_masked_conformal, "original_complete"),
                    ("random_masking_model_masked", random_test_masked, meta_test_masked,
                     random_masked_conformal, "random_masking_model_complete")):
                q = calibrator[cp]["quantile"]
                lower, upper = prediction_interval(prediction[applicable],
                    [masked_meta[i]["propagation_remaining"] for i in applicable], q,
                    p3.minimum_scale_hours)
                widths = upper - lower
                errors = prediction[applicable] - actual
                complete_mae = next(row["mae_hours"] for row in missingness_rows
                    if row["scenario"] == complete_name and row["checkpoint_type"] == cp)
                missingness_rows.append({"scenario": f"{model_name}:{name}", "checkpoint_type": cp,
                    "applicable_sample_count": len(applicable),
                    "mae_hours": float(np.mean(np.abs(errors))),
                    "rmse_hours": float(np.mean(errors ** 2) ** .5),
                    "conformal_coverage": float(np.mean((actual >= lower) & (actual <= upper))),
                    "mean_interval_width_hours": float(np.mean(widths)) if np.isfinite(widths).all() else None,
                    "degradation_vs_complete_mae_hours": float(np.mean(np.abs(errors)) - complete_mae),
                    "status": "evaluated"})
    random_masked_rows = sum(bool(selected) for selected in random_fit_masks)
    random_mask_effect = {}
    for cp in CHECKPOINTS:
        original_masked = [r["mae_hours"] for r in missingness_rows
            if r["checkpoint_type"] == cp and r["scenario"].startswith("original_model_masked:") and r["mae_hours"] is not None]
        randomized_masked = [r["mae_hours"] for r in missingness_rows
            if r["checkpoint_type"] == cp and r["scenario"].startswith("random_masking_model_masked:") and r["mae_hours"] is not None]
        original_complete = regression_report["hybrid"][cp]["mae_hours"]
        randomized_complete_values = [r["mae_hours"] for r in missingness_rows
            if r["checkpoint_type"] == cp and r["scenario"] == "random_masking_model_complete"]
        randomized_complete = float(np.mean(randomized_complete_values)) if randomized_complete_values else None
        random_mask_effect[cp] = {
            "original_complete_mae_hours": original_complete,
            "random_masking_model_complete_mae_hours": randomized_complete,
            "original_model_masked_scenario_mean_mae_hours": float(np.mean(original_masked)) if original_masked else None,
            "random_masking_model_masked_scenario_mean_mae_hours": float(np.mean(randomized_masked)) if randomized_masked else None,
            "masked_scenario_mae_improvement_hours": (float(np.mean(original_masked) - np.mean(randomized_masked))
                if original_masked and randomized_masked else None),
        }
    missingness_report = {"random_mask_probability": p3.random_mask_probability,
        "random_mask_seed": p3.split_seed, "random_masked_fit_rows": random_masked_rows,
        "inference_masking_scope": "complete hybrid: remove corresponding as-of snapshot inputs, rebuild feature rows, and recompute propagation before prediction",
        "training_random_masking_scope": "complete hybrid: deterministically mask as-of fit snapshot inputs, rebuild features and propagation, and fit residuals against the masked propagation baseline",
        "masked_conformal_calibration": "each masked model/scenario uses checkpoint-specific residuals recomputed on the correspondingly masked calibration partition",
        "compared_cases": ["original model on complete test features", "original model on masked test features",
                           "random-masking-trained model on complete test features", "random-masking-trained model on masked test features"],
        "training_time_masking_effect": random_mask_effect,
        "results": missingness_rows}

    # Forecast attribution uses as-of snapshots and the independent propagation estimator only.
    driver_meta = [{key: item[key] for key in ("snapshot", "shipment", "route", "world")} for item in meta_test]
    driver_rows = _prediction_driver_attribution(X_test, driver_meta, p3.attribution_reference_values)
    driver_summary = {}
    for variable in p3.attribution_reference_values:
        matching = [r for r in driver_rows if r["component"] == variable]
        values = [r["effect_hours"] for r in matching if r["forecast_used"]]
        driver_summary[variable] = {"reference_value": p3.attribution_reference_values[variable],
            "sample_count_with_forecast_used": len(values),
            "sample_count_not_used": len(matching) - len(values),
            "mean_effect_hours": float(np.mean(values)) if values else None,
            "median_effect_hours": float(np.median(values)) if values else None,
            "unavailable_reasons": dict(pd.Series([r["unavailable_reason"] for r in matching
                if r["unavailable_reason"] is not None]).value_counts())}
    driver_report = {"label": "Prediction driver attribution", "method": "neutralize only the exact forecast vintage selected inside propagation-only estimation",
                     "neutral_reference_values": p3.attribution_reference_values,
                     "selection_rules": "checkpoint issue-time eligibility, target valid-hour window, variable, and expected location use the propagation estimator's selector",
                     "summary": driver_summary, "rows": driver_rows,
                     "future_actuals_accessed": False}

    attribution_report = _retrospective_attribution(root, config, routes, fingerprint, seed,
        p3.counterfactual_timestamp_tolerance_seconds)
    amplification_report = _amplification_experiments(routes, config.world)
    stress_report = _stress_test(root, config, routes, models, p3.stress_seed_base, p3)

    # SHAP operates on the fitted preprocessing and only the Phase 3 residual model.
    import shap
    preprocess = models["residual"].named_steps["preprocess"]
    transformed = preprocess.transform(X_test)
    shap_names = list(preprocess.get_feature_names_out())
    explainer = shap.TreeExplainer(models["residual"].named_steps["model"], feature_names=shap_names)
    shap_raw = explainer.shap_values(transformed)
    if hasattr(shap_raw, "values"):
        shap_raw = shap_raw.values
    shap_values = np.asarray(shap_raw, dtype=float)
    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values[:, :, 0]
    if shap_values.shape != transformed.shape:
        raise RuntimeError(f"SHAP residual shape {shap_values.shape} does not match transformed feature shape {transformed.shape}")
    feature_groups = {}
    family_prefixes = {"shipment_and_schedule": ("volume", "shipment_type", "scheduled_", "checkpoint_type"),
        "route": ("route_", "route_id"), "observed_state": ("actual_", "observed_", "elapsed_", "accumulated_"),
        "forecasts": ("forecast_",), "propagation_timing": ("arrival_", "transit_deviation_", "expected_", "estimated_", "customs_gate_")}
    for name in shap_names:
        original = next((col for col in FEATURE_COLUMNS if name == col or name.startswith(col + "_")), name)
        family = next((group for group, prefixes in family_prefixes.items() if original.startswith(prefixes)), "missingness_or_other")
        feature_groups[name] = {"original_feature": original, "feature_family": family}
    global_rows = [{"feature": name, **feature_groups[name],
                    "mean_absolute_shap": float(np.mean(np.abs(shap_values[:, i]))),
                    "mean_positive_shap": float(np.mean(np.maximum(shap_values[:, i], 0))),
                    "mean_negative_shap": float(np.mean(np.minimum(shap_values[:, i], 0)))}
                   for i, name in enumerate(shap_names)]
    global_rows.sort(key=lambda row: row["mean_absolute_shap"], reverse=True)
    local_rows = []
    for cp in CHECKPOINTS:
        indexes = [i for i, m in enumerate(meta_test) if m["checkpoint_type"] == cp][:p3.local_explanation_count]
        for i in indexes:
            strongest = np.argsort(np.abs(shap_values[i]))[::-1][:10]
            for j in strongest:
                local_rows.append({"shipment_id": meta_test[i]["shipment_id"], "checkpoint_type": cp,
                    "feature": shap_names[j], **feature_groups[shap_names[j]],
                    "shap_residual_contribution_hours": float(shap_values[i, j]),
                    "direction": "positive" if shap_values[i, j] > 0 else "negative" if shap_values[i, j] < 0 else "zero",
                    "predicted_residual_hours": float(models["residual"].predict(X_test.iloc[[i]])[0])})
    shap_report = {"label": "ML residual feature contribution", "explained_model": "Phase 3 GradientBoostingRegressor residual pipeline",
        "transformed_feature_count": len(shap_names), "feature_name_mapping": feature_groups,
        "global_top_features": global_rows[:20], "mean_positive_residual_contribution_hours": float(np.mean(np.maximum(shap_values, 0))),
        "mean_negative_residual_contribution_hours": float(np.mean(np.minimum(shap_values, 0))),
        "local_explanations": local_rows}

    # Persist separate Phase 3 model and calibration artifacts; Phase 2 files are untouched.
    artifact_path = root / artifact_dir
    artifact_path.mkdir(parents=True, exist_ok=True)
    model_bundle = {"model_version": "phase3-gbr-residual-v1", "feature_schema_version": "phase2.features.v2",
        "feature_columns": FEATURE_COLUMNS, "direct": models["direct"], "residual": models["residual"],
        "random_masking_residual": random_model, "routes": routes, "world": config.world,
        "training_configuration": {"direct": model_settings, "residual": model_settings,
                                   "random_masking_residual": model_settings},
        "evaluation_configuration": {"split": split_cfg, "split_seed": p3.split_seed,
            "counterfactual_timestamp_tolerance_seconds": p3.counterfactual_timestamp_tolerance_seconds,
            "bootstrap_samples": config.evaluation.bootstrap_samples,
            "conformal_coverage": coverage, "minimum_scale_hours": p3.minimum_scale_hours,
            "probability_threshold": p3.probability_threshold,
            "significant_delay_threshold_hours": config.prediction.significant_delay_threshold_hours,
            "random_mask_probability": p3.random_mask_probability,
            "random_masking_scope": "as-of snapshots masked with matching feature rebuild and propagation recomputation; fit residual target is measured against masked propagation",
            "missingness_scenarios": p3.missingness_scenarios,
            "attribution_reference_values": p3.attribution_reference_values,
            "stress_seed_base": p3.stress_seed_base},
        "random_seed": seed,
        **provenance, "fit_shipment_ids": partitions["fit"],
        "calibration_shipment_ids": partitions["calibration"], "test_shipment_ids": partitions["test"],
        "python_version": platform.python_version(), "sklearn_version": sklearn.__version__,
        "joblib_version": joblib.__version__, "shap_version": shap.__version__,
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "matplotlib_version": __import__("matplotlib").__version__,
        "trained_at_utc": datetime.now(UTC).isoformat()}
    calibration_artifact = {"experiment_fingerprint_sha256": provenance["experiment_fingerprint_sha256"],
        "coverage": coverage, "minimum_scale_hours": p3.minimum_scale_hours,
        "conformal_by_checkpoint": conformal, "random_masking_conformal_by_checkpoint": random_conformal,
        "normalized_signed_calibration_residuals_by_checkpoint": signed_residuals,
        "probability_threshold": p3.probability_threshold,
        "significant_delay_threshold_hours": config.prediction.significant_delay_threshold_hours}
    joblib.dump(model_bundle, artifact_path / "phase3_bundle.joblib")
    joblib.dump(calibration_artifact, artifact_path / "phase3_calibration.joblib")
    _json_write(artifact_path / "phase3_metadata.json", {k: v for k, v in model_bundle.items()
        if k not in {"direct", "residual", "random_masking_residual", "routes", "world"}})

    table_path = root / "reports" / "metrics" / "phase3_comparison.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparison_rows).to_csv(table_path, index=False)
    pd.DataFrame(driver_rows).to_csv(root / "reports" / "metrics" / "phase3_prediction_driver_attribution.csv", index=False)
    pd.DataFrame(attribution_report["results"]).to_csv(root / "reports" / "metrics" / "phase3_retrospective_attribution.csv", index=False)
    pd.DataFrame(stress_report["prediction_rows"]).to_json(root / "reports" / "metrics" / "phase3_stress_predictions.json", orient="records", indent=2, date_format="iso")
    pd.DataFrame(local_rows).to_csv(root / "reports" / "metrics" / "phase3_shap_local.csv", index=False)
    pd.DataFrame(global_rows).to_csv(root / "reports" / "metrics" / "phase3_shap_global.csv", index=False)
    missing_path = root / "reports" / "metrics" / "phase3_missing_data_robustness.csv"
    pd.DataFrame(missingness_rows).to_csv(missing_path, index=False)
    figure_paths = _make_plots(root, comparison_rows, test_frame, conformal_report,
        probability_report, probabilities, shap_values, shap_names, amplification_report, missingness_rows)

    summary = {
        "phase": "Phase 3", "status": "final held-out evaluation; test partition used only for final reporting",
        "dataset": {"shipments": len(labels), "routes": len(routes), "checkpoints": list(CHECKPOINTS),
                    "snapshots": len(labels) * 3, "seed": config.data.random_seed,
                    "significant_delay_threshold_hours": config.prediction.significant_delay_threshold_hours,
                    "fingerprint_sha256": fingerprint,
                    "experiment_fingerprint_sha256": provenance["experiment_fingerprint_sha256"],
                    "raw_dataset_fingerprint_sha256": provenance["raw_dataset_fingerprint_sha256"],
                    "processed_dataset_fingerprint_sha256": provenance["processed_dataset_fingerprint_sha256"],
                    "simulation_config_fingerprint_sha256": provenance["simulation_config_fingerprint_sha256"],
                    "route_config_fingerprint_sha256": provenance["route_config_fingerprint_sha256"]},
        "experiment_provenance": provenance,
        "split_manifest": str(split_path.relative_to(root)), "split": manifest,
        "bootstrap": {**bootstrap_report, "configured_replicates": config.evaluation.bootstrap_samples,
                      "resolution": "Retained existing 1000-replicate evaluation default; task guidance said approximately 500."},
        "regression_metrics_by_variant": regression_report,
        "model_configuration": model_settings,
        "evaluation_shipment_ids_by_variant": {variant: partitions["test"] for variant in VARIANTS},
        "aggregate_snapshot_weighted_regression_metrics": aggregate_report,
        "best_per_checkpoint_descriptive_only": best_report,
        "test_data_model_selection_or_tuning": False,
        "conformal": {"method": "normalized split conformal by checkpoint", "model": "hybrid",
                      "metrics_by_checkpoint": conformal_report},
        "significant_delay_probability": {"method": "empirical normalized signed calibration residuals, checkpoint-specific",
                                          "formula": {
                                              "calibration_residual_hours": "r_i = actual_remaining_i - predicted_remaining_i",
                                              "calibration_scale_hours": "s_i = max(propagation_remaining_i, minimum_scale_hours)",
                                              "normalized_residual": "z_i = r_i / s_i",
                                              "new_prediction_scale_hours": "s_new = max(propagation_remaining_new, minimum_scale_hours)",
                                              "probability": "mean(predicted_remaining_new + z_i * s_new > scheduled_delivery_at + threshold_hours - checkpoint_at)",
                                              "normalization_rationale": "Remaining-time residual magnitude can scale with journey duration; normalization expresses calibration errors relative to the propagation duration and rescales those relative errors to the new prediction. This assumes approximate proportional error scaling and is not guaranteed by the data-generating process.",
                                              "calibration_only": True,
                                              "test_residuals_used": False,
                                              "separate_from_conformal_intervals": True,
                                          },
                                          "metrics_by_checkpoint": probability_report,
                                          "direct_classifier": {"retained": False,
                                              "reason": "Small 45-shipment calibration/test sets and class imbalance do not justify an uncalibrated classifier."}},
        "retrospective_cause_attribution": attribution_report,
        "prediction_driver_attribution": driver_report,
        "shap_residual_explanation": shap_report,
        "amplification_experiments": amplification_report,
        "missing_data_robustness": missingness_report,
        "stress_tests": stress_report,
        "artifacts": {"model_bundle": str((artifact_path / "phase3_bundle.joblib").relative_to(root)),
                      "calibration": str((artifact_path / "phase3_calibration.joblib").relative_to(root)),
                      "metadata": str((artifact_path / "phase3_metadata.json").relative_to(root)),
                      "comparison_table": str(table_path.relative_to(root)),
                      "figures": figure_paths},
        "limitations": [
            "Synthetic simulator metrics do not establish real-world accuracy or causal effects.",
            "The fixed test set contains 45 shipments and bootstrap intervals remain coarse.",
            "The weather plus reduced recovery stress family is not represented by existing Phase 1 mechanisms.",
            "The existing Phase 2 prediction service is backward compatible but does not consume Phase 3 calibration artifacts; Phase 3 bundles are separate.",
            "The derived probability is a small-sample empirical calibration approximation, not a classifier or conformal probability distribution.",
        ]}
    metrics_path = root / "reports" / "metrics" / "phase3_evaluation.json"
    _json_write(metrics_path, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run the complete Phase 3 evaluation")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--config", default="configs/simulation.yaml")
    parser.add_argument("--routes")
    parser.add_argument("--artifact-dir", default="artifacts/models")
    args = parser.parse_args()
    summary = run_phase3(args.data_root, args.config, args.routes, args.artifact_dir)
    print(json.dumps({"metrics": "reports/metrics/phase3_evaluation.json",
                      "dataset": summary["dataset"], "split_counts": summary["split"]["partition_counts"],
                      "best_descriptive": summary["best_per_checkpoint_descriptive_only"],
                      "figures": summary["artifacts"]["figures"]}, indent=2))


if __name__ == "__main__":
    main()
