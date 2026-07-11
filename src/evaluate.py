"""
evaluate.py — Reusable evaluation metrics suite.

Used by:
  - notebooks (imported for consistent metric computation)
  - src/train.py (self-check after training)
  - tests/test_evaluate.py

Single public function: evaluate_model(model, X, y, threshold, label)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.50,
    label: str = "",
    verbose: bool = True,
) -> dict[str, float]:
    """
    Compute the full evaluation metric suite for a fitted model on one split.

    Parameters
    ----------
    model     : Fitted sklearn-compatible estimator with predict_proba().
    X         : Feature DataFrame.
    y         : True binary labels.
    threshold : Decision boundary for converting probabilities to 0/1.
                Use 0.50 as default; override with ticker-specific value
                from optimized_hyperparameters.json for deployment metrics.
    label     : Optional label printed in verbose output (e.g. "Val", "Test").
    verbose   : If True, prints a formatted metric summary.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc, threshold.
    """
    if len(y) == 0:
        return {
            "accuracy": float("nan"), "precision": float("nan"),
            "recall":   float("nan"), "f1":        float("nan"),
            "roc_auc":  float("nan"), "threshold": threshold,
        }

    y_proba = model.predict_proba(X)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    metrics = {
        "accuracy":  round(accuracy_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y, y_proba), 4),
        "threshold": threshold,
    }

    if verbose:
        header = f"  [{label}]" if label else " "
        print(f"{header}")
        print(f"    ROC-AUC   : {metrics['roc_auc']:.4f}  ← primary metric")
        print(f"    F1        : {metrics['f1']:.4f}")
        print(f"    Precision : {metrics['precision']:.4f}")
        print(f"    Recall    : {metrics['recall']:.4f}")
        print(f"    Accuracy  : {metrics['accuracy']:.4f}")
        print(f"    Threshold : {metrics['threshold']:.4f}")

        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        print(f"    Confusion matrix:")
        print(f"      Predicted DOWN | Predicted UP")
        print(f"      True DOWN  {tn:>5}  |  {fp:>5}")
        print(f"      True UP    {fn:>5}  |  {tp:>5}")

    return metrics
