# Data dictionary

This dictionary describes the checked-in Parquet datasets and saved Phase 2/3 artifacts. Counts are from the current seed-42 dataset and may change if a user intentionally regenerates it. All route profiles and records are synthetic. Timestamps are timezone-aware UTC. Parquet nested snapshot values are stored as JSON strings and parsed into their typed Pydantic structures before inference.

## Dataset inventory

| Artifact | Location | Grain | Current rows |
|---|---|---|---:|
| Shipment master | `data/raw/shipments.parquet` | One row per shipment | 300 |
| Actual event log | `data/raw/events.parquet` | Eight lifecycle event rows per shipment | 2,400 |
| Actual environment | `data/raw/environment_actuals.parquet` | Route/location/hour | 23,013 |
| Forecast vintages | `data/raw/forecasts.parquet` | Shipment/checkpoint/forecast variable | 3,000 |
| As-of snapshots | `data/processed/prediction_snapshots.parquet` | Shipment/checkpoint | 900 |
| Evaluation targets | `data/processed/snapshot_targets.parquet` | Shipment/checkpoint | 900 |

There are three synthetic route IDs with 100 shipments each: `nhava_sheva_inland_a`, `mundra_inland_b` and `synthetic_corridor_c`. There are exactly three checkpoints: `departure`, `port_arrival` and `customs_cleared`.

## Shipment master

**Source:** `data/raw/shipments.parquet` (300 rows, 13 columns). String columns are read as `object`, volume as `float64`, seed as `int64`, timestamps as `datetime64[ns, UTC]`.

| Field | Type | Meaning / units | Status and feature use |
|---|---|---|---|
| `shipment_id` | string | Stable synthetic shipment key | Identifier; used to group partitions, not a numeric feature |
| `route_id` | string | One of the three configured route profiles | Configuration/feature category |
| `origin` | string | Synthetic origin label | Manifest metadata; feature eligibility is determined by the feature builder |
| `destination_port` | string | Synthetic destination-port label | Manifest metadata; not raw ground truth |
| `final_destination` | string | Synthetic inland destination label | Manifest metadata; feature eligibility is determined by the feature builder |
| `shipment_type` | string | Synthetic cargo category | Shipment attribute; encoded by the feature builder |
| `volume` | float | Synthetic volume indicator; schema has no physical unit | Shipment attribute; encoded by the feature builder |
| `scheduled_departure_at` | UTC timestamp | Planned origin departure | Known schedule; used for schedule-derived features |
| `scheduled_port_arrival_at` | UTC timestamp | Planned port arrival | Known schedule; feature eligibility is checkpoint-specific |
| `scheduled_customs_clearance_at` | UTC timestamp | Planned customs completion | Known schedule; feature eligibility is checkpoint-specific |
| `scheduled_delivery_at` | UTC timestamp | Planned final delivery | Known schedule; used to calculate predicted delay, not actual outcome |
| `simulation_seed` | integer | Seed lineage for deterministic journey generation | Reproducibility metadata; not a predictor feature |
| `scenario_id` | string | Synthetic scenario family label | Provenance metadata; not a predictor feature |

## Actual event log

**Source:** `data/raw/events.parquet` (2,400 rows, 28 columns; eight event types, 300 rows each). Actual event and mechanism fields are ground truth or simulator diagnostics. The complete table is **not** a predictor input; only checkpoint-allowed event timestamps or stage values copied into an as-of snapshot can be used.

| Field(s) | Stored type | Meaning / units | Availability / model policy |
|---|---|---|---|
| `shipment_id`, `event_id` | string | Shipment reference and unique event key | Identifiers |
| `event_type`, `stage` | string | Lifecycle event and operational stage | Event metadata |
| `scheduled_at`, `actual_at` | UTC timestamp | Planned and realized event times | Actual time is ground truth; only events observable by the selected checkpoint may be copied into a snapshot |
| `event_cause` | nullable string | Simulator-coded reason/condition for an event | Retrospective ground-truth explanation only |
| `delay_hours` | nullable float, hours | Actual event deviation from its relevant schedule when defined | Derived actual; not a feature |
| `observable_after` | string | Earliest prediction checkpoint for event visibility | `departure`, `port_arrival`, `customs_cleared`, or `ground_truth_only` according to event type |
| `berth_id` | nullable numeric (semantically berth index) | Assigned berth | Actual simulator outcome; not a future feature |
| `next_eligible_slot_at` | UTC timestamp | Next berth slot used by the simulator | Ground-truth diagnostic |
| `slot_wait_hours`, `queue_wait_hours` | nullable float, hours | Realized berth slot and queue waits | Ground-truth diagnostic; not predictor input |
| `queue_length_at_arrival`, `available_berths_at_arrival` | nullable numeric, count | Queue/capacity measured by simulator at arrival | The observable queue/capacity values are separately copied into the port-arrival snapshot; this full event table remains excluded |
| `transit_perturbation_hours` | nullable float, hours | Seeded transit perturbation | Hidden simulator draw; not a predictor feature |
| `berth_service_duration_hours` | nullable float, hours | Realized berth service duration | Future realized duration; never used as a future feature |
| `berth_service_multiplier` | nullable float, dimensionless | Seeded service multiplier, Uniform[0.85, 1.15] | Hidden simulator draw; not a predictor feature |
| `customs_eligibility_at` | UTC timestamp | Time at which actual processing may start under the calendar | Ground-truth timestamp |
| `customs_closure_wait_hours`, `customs_processing_duration_hours` | nullable float, hours | Realized closure wait and processing duration | Ground-truth outcome; duration is observed only after customs completes and then represented in the snapshot |
| `customs_workload_actual` | nullable float, normalized [0,1] | Actual customs workload used by the simulator | Actual; future values are forbidden; an as-of observation may be included at customs clearance |
| `customs_processing_multiplier` | nullable float, dimensionless | Deterministic multiplier from actual workload | Simulator diagnostic; not a feature |
| `gate_wait_hours` | nullable float, hours | Actual wait for inland dispatch window | Ground-truth outcome |
| `next_dispatch_at` | UTC timestamp | Actual next truck/gate dispatch time | Future actual; hidden before it occurs |
| `inland_travel_duration_hours` | nullable float, hours | Realized inland travel duration | Ground-truth outcome |
| `inland_traffic_actual` | nullable float, normalized [0,1] | Actual inland traffic | Actual; future values are forbidden; an as-of observation/eligible forecast can be used at customs clearance |
| `inland_travel_multiplier` | nullable float, dimensionless | Deterministic travel multiplier from actual traffic | Simulator diagnostic; not a feature |

Event visibility mapping:

| Event type | `observable_after` | Notes |
|---|---|---|
| `departure` | `departure` | Triggers first prediction checkpoint |
| `port_arrival` | `port_arrival` | Arrival is observed before berth completion |
| `berth_start`, `port_complete`, `customs_start`, `customs_complete` | `customs_cleared` | Available by the third checkpoint; earlier snapshots cannot expose them |
| `gate_out`, `delivery` | `ground_truth_only` | Evaluation/retrospective only; never predictor-visible |

## Environment actuals

**Source:** `data/raw/environment_actuals.parquet` (23,013 rows, 10 columns; 7,671 rows per route). Grain is one synthetic route/location at one hourly UTC index. These are realized operating conditions used by the simulator and evaluation. Only the current checkpoint observation can be surfaced into a snapshot; future actual rows are not eligible model features.

| Field | Type | Meaning / units | Status / feature use |
|---|---|---|---|
| `location_id` | string | Synthetic origin, port or inland location | Location key |
| `valid_at` | UTC timestamp | Hour bucket for the condition | Actual environment time index |
| `congestion_actual` | float, normalized [0,1] | Realized congestion indicator | Actual; only a current as-of observation can be a feature |
| `weather_actual` | float, normalized [0,1] | Realized weather-severity indicator | Actual; only a current as-of observation can be a feature |
| `customs_workload_actual` | float, normalized [0,1] | Realized customs workload | Actual; future values forbidden |
| `berth_capacity_actual` | nullable numeric, berth count | Realized available capacity | Actual; current port-arrival observation may be copied to a snapshot |
| `inland_traffic_actual` | float, normalized [0,1] | Realized inland traffic | Actual; future values forbidden |
| `route_id` | string | Synthetic route reference | Configuration key |
| `location_kind` | string | Environment location family | Environment metadata |
| `environment_seed` | integer | Deterministic environment stream seed | Reproducibility metadata; not a feature |

## Forecast records

**Source:** `data/raw/forecasts.parquet` (3,000 rows, 9 columns). Grain is shipment × checkpoint × forecast variable. Variables occur as follows: inland traffic 900; customs workload, port congestion and port weather 600 each; transit congestion 300. Forecast indicators and uncertainty are normalized/dimensionless; forecast issuance/validity are UTC timestamps.

| Field | Type | Meaning | Status / eligibility |
|---|---|---|---|
| `location_id` | string | Synthetic forecast location | Metadata |
| `forecast_issued_at` | UTC timestamp | Time forecast vintage became available | Must be at or before the snapshot checkpoint |
| `forecast_valid_at` | UTC timestamp | Time the forecast describes | Separate from issue time; Phase 1 values are hourly point forecasts |
| `forecast_variable` | string | `transit_congestion`, `port_congestion`, `port_weather`, `customs_workload` or `inland_traffic` | Must also be in the checkpoint's explicit variable allowlist |
| `forecast_value` | float, normalized [0,1] | Forecast value for the named indicator | Forecast feature only when vintage and variable are eligible |
| `uncertainty` | nullable float, normalized/dimensionless | Forecast uncertainty diagnostic | Forecast feature when included by the feature builder |
| `source` | string | Synthetic route-profile forecast source | Provenance metadata |
| `shipment_id` | string | Forecast's journey context | Identifier |
| `checkpoint_type` | string | Forecast issue/checkpoint context | Checkpoint metadata |

## Checkpoint prediction snapshots

**Source:** `data/processed/prediction_snapshots.parquet` (900 rows, 10 columns; one row for each shipment/checkpoint). Top-level physical types are strings/objects for identifiers and JSON-encoded nested structures, plus UTC timestamps for `checkpoint_at` and `scheduled_delivery_at`.

| Field | Type / structure | Meaning and feature policy |
|---|---|---|
| `shipment_id` | string | Journey key; identifier, not a numeric feature |
| `checkpoint_type` | string enum | `departure`, `port_arrival` or `customs_cleared` |
| `checkpoint_at` | UTC timestamp | Prediction-time reference and trigger-event time |
| `scheduled_delivery_at` | UTC timestamp | Known schedule used for delay calculation |
| `route_id` | string | Synthetic route configuration key |
| `observed_event_times` | JSON object of UTC timestamps | Allowlist: departure has `departure_at`; port arrival adds `port_arrival_at`; customs cleared adds `customs_complete_at`. Trigger time equals `checkpoint_at`. No future actual event timestamps. |
| `observed_stage_fields` | JSON object of typed scalar fields | Departure: cargo category, volume, origin condition. Port arrival adds observed transit duration/disruption and queue/capacity state. Customs cleared adds actual customs processing duration and truck availability. Values must be observed by that checkpoint. |
| `current_environment` | JSON object of typed scalar fields | Checkpoint-allowlisted current observations only. Queue state is route-global at port arrival; customs-cleared does not fabricate a queue value. `truck_availability=1.0` encodes the no-shortage assumption. |
| `forecasts` | JSON list of forecast objects | Eligible issue times and checkpoint-allowed variables; object fields are the forecast record fields excluding journey/checkpoint identifiers. |
| `feature_missingness` | JSON object of booleans | Missingness flags for eligible fields. It is `{}` in all 900 checked-in snapshots; model feature construction also uses fixed per-feature missingness flags. |

Snapshot schema fields are predictor inputs only through the checkpoint feature builder. Actual future environment and target fields are absent. Forecasts are selected by the snapshot builder using issue time, not hindsight.

## Evaluation targets

**Source:** `data/processed/snapshot_targets.parquet` (900 rows, 5 columns; exactly one target row per snapshot). These values are deliberately stored separately from snapshot features and are loaded only for model training, calibration/evaluation or retrospective reporting.

| Field | Type | Meaning / units | Model feature? |
|---|---|---|---|
| `shipment_id` | string | Journey key | No; partition/group key |
| `checkpoint_type` | string | Corresponding checkpoint | No target metadata |
| `remaining_hours_actual` | float, hours | `actual_delivery_at - checkpoint_at` | **Target only** for remaining-duration prediction |
| `actual_delay_hours` | float, hours | `actual_delivery_at - scheduled_delivery_at` | **Target/label only** |
| `significant_delay_label` | boolean | Actual total delay strictly greater than the configured 12-hour assumption | **Target/label only** |

## Predictions, model metadata and evaluation outputs

| Artifact | Location | Contents / status |
|---|---|---|
| Phase 2 model bundle | `artifacts/models/phase2_bundle.joblib` | Direct and residual Gradient Boosting models, classifier, preprocessing, route/world settings and prediction configuration. Phase 2 intervals are point placeholders and its classifier probabilities are uncalibrated. |
| Phase 2 metadata | `artifacts/models/phase2_metadata.json` | Feature schema/order, model settings, seed, dataset fingerprint, split IDs and runtime versions (Python 3.13.1, scikit-learn 1.6.1, joblib 1.4.2). |
| Phase 2 preliminary metrics | `reports/metrics/phase2_preliminary.json` | Preliminary grouped 80/20 validation; not final generalization evidence. |
| Optional prediction invocation history | `artifacts/prediction_history.jsonl` | One JSON prediction record per CLI invocation; contains checkpoint, predicted remaining hours/ETA/delay/probability, interval fields, propagation/residual components and model/schema versions. This is not the canonical evaluation dataset. |
| Phase 3 model bundle | `artifacts/models/phase3_bundle.joblib` | Separate evaluation-trained direct/residual/masked residual components and route/world configuration. |
| Phase 3 calibration | `artifacts/models/phase3_calibration.joblib` | Per-checkpoint normalized split conformal quantiles and normalized signed calibration residuals; uses the calibration partition. |
| Phase 3 metadata | `artifacts/models/phase3_metadata.json` | Experiment/data/config fingerprints, model configuration, partition provenance and runtime versions. |
| Phase 3 evaluation JSON | `reports/metrics/phase3_evaluation.json` | Held-out regression, risk, conformal, bootstrap, attribution, masking and stress results, including explicit limitations. |
| Phase 3 comparison CSV | `reports/metrics/phase3_comparison.csv` | One row per baseline/model and checkpoint: n, MAE/RMSE/median/P90 absolute error, secondary R² and carry-forward improvement. |
| Split manifest | `reports/metrics/phase3_split_manifest.json` | Shipment-level fit/calibration/test membership (210/45/45). |
| Attribution/robustness files | `reports/metrics/phase3_*attribution*.csv`, `phase3_shap_*.csv`, `phase3_missing_data_robustness.csv`, `phase3_stress_predictions.json` | Prediction-driver sensitivity, retrospective simulator attribution, residual SHAP, masked-input robustness and separate stress cohorts. These explanations are not real-world causal estimates. |
| Evaluation figures | `reports/figures/phase3_*.png` | Baseline error, coverage, probability, robustness, amplification and SHAP visualizations. |

The Phase 3 test partition has 45 shipments per checkpoint. The probability estimator is based on 45 calibration shipments per checkpoint; the current test split contains 40 significant-delay and 5 non-significant shipments at each checkpoint. Small calibration/test samples limit confidence in risk and coverage estimates.
