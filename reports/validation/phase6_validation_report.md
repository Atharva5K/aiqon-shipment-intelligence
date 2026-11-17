# Phase 6 Final Validation Record

**Status: PASS** — final validation completed on 2026-11-17. No Phase 0–5 business logic, datasets, model artifacts, evaluation results, or random seeds were changed.

## Environment and dependencies

- Fresh environment: `.venv_phase6_validation`, created with Python 3.13.1.
- Installation: `py -3.13 -m venv .venv_phase6_validation`; `python -m pip install --upgrade pip`; `python -m pip install -r requirements.txt`.
- `pip` upgrade completed to 26.3. Dependency installation from the sandbox initially could not resolve the package index; the same requested installation completed after network access was approved.
- `requirements.txt` delegates to `pyproject.toml` and installs the `dev` and `assistant` extras for the full test environment. Core-only installation is `python -m pip install .`; Groq and `python-dotenv` are optional assistant dependencies. The dashboard was separately exercised after both packages had been removed.
- Artifact-sensitive versions match saved metadata: NumPy 2.1.1, pandas 2.2.3, scikit-learn 1.6.1, joblib 1.4.2, Matplotlib 3.10.1, and SHAP 0.47.2. Python is constrained to `>=3.13,<3.14`; scikit-learn and joblib are pinned to saved-artifact versions.
- `python -m pip check`: **PASS**, `No broken requirements found` (both before and after removing the optional assistant packages for the core-only dashboard check).

## Testing and compilation

- Command: `.venv_phase6_validation\Scripts\python.exe -m pytest -q` from the repository root, with live Groq opt-in unset.
- Collected: **187** tests; **185 passed, 0 failed, 2 skipped**.
- Warnings: **14**, all Pyparsing deprecation warnings emitted through Matplotlib during the Phase 3 integration test.
- The two skipped tests are the explicit opt-in live Groq text and tool-call smoke tests. No provider request or API quota was used.
- `python -m compileall -q src tests app.py`: **PASS**.
- Documented generator, Phase 2 training, and Phase 3 evaluation CLI help commands all returned successfully. Training/evaluation were not run against the checked-in output directories.

## Reproducibility

- Command: `.venv_phase6_validation\Scripts\python.exe .phase6-repro-smoke.py`.
- Result: **PASS**. With copied configs, 10 journeys, seed 42, and the three original route profiles, two runs matched exactly across all six Parquet tables, including values, row order, dtypes, and schemas.
- Generated 30 snapshots: 10 at each required checkpoint. Pydantic shipment and snapshot validation passed.
- Detailed counts and comparison method: [reproducibility report](reproducibility_report.md).

## Dashboard and model compatibility

- Documented command: `streamlit run app.py`.
- Actual headless startup used the fresh environment at port 8765. Health and home routes both returned **HTTP 200**.
- Streamlit AppTest loaded Overview, Shipment Intelligence, Explainability, What-if Simulation, and AI Assistant after Groq and `python-dotenv` were removed. A Phase 3 shipment prediction rendered; the assistant displayed missing-key guidance and its chat input was disabled. No external API credential was set.
- Phase 2 `PredictionService.load` and Phase 3 `Phase3PredictionAdapter.load` both loaded the saved bundles. A read-only `SHP-0001` departure prediction returned 145.850337 hours remaining and ETA `2025-01-16T00:55:01.213484+00:00`.
- Phase 3 missing-artifact handling and dashboard interactions are also covered by the passing project tests.

## Dataset, artifacts, and repository hygiene

- The checked-in dataset remains **300 unique shipments, 3 routes, 2,400 actual events, 23,013 environment observations, 3,000 forecasts, 900 snapshots, and 900 snapshot targets**.
- Pre/post SHA-256 comparison matched for 14 files: both simulation/route configs, all six source dataset tables, Phase 2 and Phase 3 model bundles, Phase 3 calibration, Phase 3 evaluation JSON/CSV, and the dataset summary. No training or evaluation command wrote into checked-in artifact/output paths.
- `.gitignore` excludes Python caches, virtual environments (including `.venv_*`), build outputs, `.env` and `.env.*`, while retaining `.env.example`.
- No `.env` file or `GROQ_API_KEY` was present. Source/config/docs scan found no Groq `gsk_` credential pattern; `.env.example` contains only the placeholder.
- The repository has no Git metadata, so a Git diff was unavailable. Before/after hashes were used for preserved data, configuration, models, and evaluation outputs.
- Temporary validation scripts, logs, temporary datasets, virtual environment, caches, and bytecode are removed from the final project and ZIP. The final ZIP is verified to contain the project source, tests, datasets, model artifacts, configuration, documentation, and reports, and to exclude credentials and temporary artifacts.

## Definition of Done

- [x] Fresh Python 3.13 installation succeeds and matches saved model versions.
- [x] Dependency declarations are consistent; optional assistant dependencies are not required for core dashboard operation.
- [x] `pip check` succeeds.
- [x] Complete unit and integration test suite passes; live tests are explicitly skipped unless opted in.
- [x] Two-run 10-journey data reproducibility smoke test passes.
- [x] Original 300-journey dataset and 900 snapshots remain unchanged.
- [x] Saved Phase 2 and Phase 3 artifacts load in the supported environment.
- [x] Dashboard core starts and renders all pages without Groq or `python-dotenv`.
- [x] README setup and runtime commands were inspected; generator, training, and evaluation command help checks pass.
- [x] No credentials or unnecessary temporary artifacts are included in the final package.
- [x] Technical report and five-minute demo materials remain present; the PDF is 3 pages.
- [x] Phase 0–5 regression suite passes and the protected data/model/evaluation hashes match.

## Remaining limitations

The dataset and route configurations are synthetic, and the reproducibility smoke cohort has 10 journeys. The 14 test warnings originate from the pinned Matplotlib version's use of newer Pyparsing APIs. No live Groq request was made. Git-based diff/status verification was unavailable because this repository copy has no `.git` directory.

