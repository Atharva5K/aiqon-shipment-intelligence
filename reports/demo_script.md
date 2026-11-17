# Five-minute demonstration script

**Duration:** 5:00 maximum. Use the local synthetic dataset and checked-in Phase 3 bundle. No Groq key or network request is needed.

## Before the demonstration

From the repository root, with the documented Python 3.13 environment installed and the checked-in data/artifacts present:

```powershell
streamlit run app.py
```

Open the local Streamlit URL. Use shipment **SHP-0001**. The app's pages are selected from the left sidebar. Keep the Groq assistant disabled; it is optional.

## Run of show

### 0:00–0:40 — Problem and architecture

**Navigate:** Start on **Overview**.

**Say:** “The assignment asks how delays propagate across a shipment journey and how an ETA changes as information arrives. This is a synthetic experiment: the simulator creates complete truth, while the prediction system sees checkpoint-scoped information. The 300 journeys use three synthetic route profiles.”

**Show:** Overview shipment/risk summaries. Point out that the system is not a live port feed.

### 0:40–1:30 — Synthetic journey and nonlinear propagation

**Navigate:** **Shipment Intelligence**. Select `SHP-0001` if another shipment is selected.

**Say:** “A modest transit shift can cross a berth slot or meet a queue, and customs and gate calendars can carry that delay forward. The simulator models berth slot waiting, berth queue competition, weekend customs closure, and inland dispatch cutoffs.”

**Show:** Route and scheduled/actual timeline, explicitly calling the actual timeline retrospective ground truth. Use `reports/figures/phase3_amplification_quiet_busy.png` only if discussing the controlled experiment: a 2-hour injected transit delay produced a 24-hour delivery shift in the busy case (factor 12), while a 1-hour quiet-case delay was absorbed (factor 0). These are designed simulator examples, not operational causal estimates.

### 1:30–2:30 — Three checkpoint predictions

**Navigate:** Stay on **Shipment Intelligence** and review its checkpoint prediction history for `SHP-0001`.

**Say:** “Each row is a new prediction using information available at that point; later predictions are not evidence that the earlier prediction knew the future.”

**Verified Phase 3 outputs for SHP-0001:**

| Checkpoint | Predicted delivery UTC | Remaining hours | Significant-delay probability |
|---|---|---:|---:|
| Departure | 2025-01-16 00:55 | 145.850 | 0.756 |
| Port arrival | 2025-01-17 04:42 | 71.753 | 0.978 |
| Customs cleared | 2025-01-17 04:51 | 27.556 | 1.000 |

Intervals are shown separately from probabilities. Explain that the risk is an empirical calibration-based approximation and interval width is not a probability distribution.

### 2:30–3:30 — Baseline comparison and explainability

**Navigate:** **Overview** for evaluation summaries if shown, then **Explainability**. Select shipment `SHP-0001` and checkpoint `departure` in the operational-context selectors.

**Say:** “On the 45-shipment test set per checkpoint, hybrid had lower MAE than carry-forward at all checkpoints, and the best MAE at port arrival. ML-only was better at departure and customs clearance. We report all five approaches rather than assuming the hybrid always wins.”

**Show:** The selected checkpoint's observed context and eligible forecasts as separate categories, then the distinct retrospective cause, prediction-driver and SHAP sections. Do not describe SHAP as causation.

### 3:30–4:30 — What-if simulation

**Navigate:** **What-if Simulation**. Keep shipment `SHP-0001` and select `departure`. Set the **Port congestion** assumption to `0.95`, then choose **Apply scenario**.

**Say:** “This is a copied, in-memory assumption change using the existing prediction adapter. It does not rerun the truth simulator or modify saved data.”

**Expected verified output:** The service labels the result **SIMULATED SCENARIO**. Against the stored departure assumptions, the verified scenario increased predicted remaining duration/ETA by approximately **22.387 hours** for this shipment. Use **Reset scenario controls** to restore defaults.

### 4:30–5:00 — Optional assistant or conclusion

If a reviewer has separately supplied a valid Groq key and requests an assistant demonstration, navigate to **AI Assistant** and submit one evidence-backed question; a live model request requires internet and may consume quota. Otherwise, show that the assistant is disabled without credentials and say: “The dashboard and prediction core work without an LLM. These results characterize a controlled synthetic simulator and do not prove real-world performance.”

## Offline fallback

The entire five-minute flow uses checked-in synthetic data and saved model artifacts. If Groq is unavailable, omit live chat; the assistant's missing-key guidance is expected. If Streamlit cannot start, use the checked-in comparison chart and the prediction values above as verified artifact outputs, and do not claim an interactive demonstration occurred.
