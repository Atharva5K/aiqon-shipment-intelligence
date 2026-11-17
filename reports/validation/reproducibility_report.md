# Reproducibility Smoke Test

**Result: PASS** — two independently generated 10-journey cohorts matched exactly.

## Environment

- Python: 3.13.1
- NumPy: 2.1.1
- pandas: 2.2.3
- PyArrow: 26.0.1
- scikit-learn: 1.6.1
- joblib: 1.4.2
- Pydantic: 2.14.3
- PyYAML: 6.0.3

## Configuration and procedure

The smoke test copied `configs/simulation.yaml` and `configs/routes.yaml` to a temporary workspace. In the copied simulation config only `data.n_shipments` was set to 10; the original random seed, 42, and all route values were retained. The production configuration and dataset were not used as output targets.

`generate_dataset` ran twice with those same copied configurations and separate temporary output roots. The test compared all six Parquet outputs using `pandas.testing.assert_frame_equal` with exact value, dtype, and row-order checks. It also compared schemas and generated summary counts, validated shipment and checkpoint snapshot rows with their Pydantic schemas, and checked the three expected checkpoint counts.

| Output table | Rows in each run |
|---|---:|
| `shipments` | 10 |
| `events` | 80 |
| `environment_actuals` | 13,725 |
| `forecasts` | 100 |
| `prediction_snapshots` | 30 |
| `snapshot_targets` | 30 |

There were 10 snapshots at each of `departure`, `port_arrival`, and `customs_cleared`. All three configured route IDs appeared. Both runs produced matching values, order, dtypes, schemas, and summary counts.

## Original files

The SHA-256 values of both configuration files, all six checked-in dataset tables, the Phase 2 and Phase 3 model bundles, Phase 3 calibration, the Phase 3 evaluation JSON/CSV, and the dataset summary matched their pre-validation values. The smoke test wrote only beneath its temporary workspace, which was removed when the run completed.

## Limitation

This smoke test uses 10 journeys to keep validation quick. Each run still uses the simulator's complete-cohort mechanism, including route-global berth competition; it does not estimate full-cohort performance. The checked-in dataset remains the separate 300-journey cohort.
