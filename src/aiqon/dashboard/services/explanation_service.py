"""Load Phase 3 explanations while keeping attribution concepts separate."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ExplanationBundle:
    retrospective_cause_attribution: pd.DataFrame
    prediction_driver_attribution: pd.DataFrame
    shap_global: pd.DataFrame
    shap_local: pd.DataFrame
    shap_figure: Path | None
    unavailable: dict[str, str]


def _load_csv(path: Path, category: str, unavailable: dict[str, str]) -> pd.DataFrame:
    if not path.is_file():
        unavailable[category] = f"Optional explanation artifact is missing: {path}"
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path).copy(deep=True)
        required_by_category = {
            "retrospective": {"shipment_id", "mechanism"},
            "prediction_driver": {"shipment_id", "checkpoint", "component"},
            "shap_global": {"feature", "mean_absolute_shap"},
            "shap_local": {"shipment_id", "checkpoint_type", "feature"},
        }
        missing = required_by_category[category] - set(frame.columns)
        if missing:
            unavailable[category] = f"{path.name} is missing expected fields: {sorted(missing)}"
            return pd.DataFrame()
        return frame
    except Exception as exc:
        unavailable[category] = f"Could not read {path.name}: {exc}"
        return pd.DataFrame()


def load_explanations(metrics_dir: str | Path, figures_dir: str | Path) -> ExplanationBundle:
    metrics_dir, figures_dir = Path(metrics_dir), Path(figures_dir)
    unavailable: dict[str, str] = {}
    retrospective = _load_csv(metrics_dir / "phase3_retrospective_attribution.csv",
                               "retrospective", unavailable)
    driver = _load_csv(metrics_dir / "phase3_prediction_driver_attribution.csv",
                       "prediction_driver", unavailable)
    global_shap = _load_csv(metrics_dir / "phase3_shap_global.csv", "shap_global", unavailable)
    local_shap = _load_csv(metrics_dir / "phase3_shap_local.csv", "shap_local", unavailable)
    figure = figures_dir / "phase3_shap_global_summary.png"
    if not figure.is_file():
        unavailable["shap_figure"] = f"Optional SHAP summary figure is missing: {figure}"
        figure = None
    return ExplanationBundle(retrospective.copy(deep=True), driver.copy(deep=True),
                             global_shap.copy(deep=True), local_shap.copy(deep=True),
                             figure, dict(unavailable))

