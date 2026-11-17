"""Regression and significant-delay probability metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)


def regression_metrics(actual, predicted, carry_forward=None) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    mae = float(mean_absolute_error(actual, predicted))
    carry_mae = (float(mean_absolute_error(actual, np.asarray(carry_forward, dtype=float)))
                 if carry_forward is not None else None)
    return {
        "n": int(len(actual)), "mae_hours": mae,
        "rmse_hours": float(mean_squared_error(actual, predicted) ** 0.5),
        "median_absolute_error_hours": float(np.median(np.abs(error))),
        "p90_absolute_error_hours": float(np.quantile(np.abs(error), 0.9)),
        "r2_secondary": float(r2_score(actual, predicted)) if len(actual) > 1 and np.var(actual) > 0 else None,
        "carry_forward_mae_hours": carry_mae,
        "improvement_over_carry_forward_hours": (carry_mae - mae if carry_mae is not None else None),
        "improvement_over_carry_forward_fraction": ((carry_mae - mae) / carry_mae
            if carry_mae is not None and carry_mae > 0 else None),
    }


def probability_metrics(labels, probabilities, threshold: float) -> dict:
    y = np.asarray(labels, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    predicted = (p >= threshold).astype(int)
    classes = np.unique(y)
    two_classes = len(classes) == 2
    prevalence = float(y.mean()) if len(y) else None
    return {
        "n": int(len(y)), "positive_count": int(y.sum()),
        "negative_count": int(len(y) - y.sum()), "prevalence": prevalence,
        "probability_method": "checkpoint-specific empirical normalized signed calibration residuals",
        "decision_threshold": float(threshold), "brier_score": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if two_classes else None,
        "roc_auc_unavailable_reason": None if two_classes else "test labels contain only one class",
        "average_precision": float(average_precision_score(y, p)) if two_classes else None,
        "average_precision_unavailable_reason": None if two_classes else "test labels contain only one class",
        "pr_auc_reported_due_to_low_prevalence": bool(prevalence is not None and prevalence < 0.10),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "confusion_matrix_labels_0_then_1": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }

