"""
train.py — Train production models and save .joblib artifacts.

Reads models/optimized_hyperparameters.json (produced by notebook 04),
trains each ticker's winning architecture on the combined train+val window,
and saves a fitted model to models/{ticker}_best_model.joblib.

Usage
-----
From the project root:
    python -m src.train                        # train all tickers
    python -m src.train --tickers COMI_CA      # train one ticker
    python -m src.train --tickers COMI_CA ORWE_CA  # train a subset

The test set (> 2024-12-31) is never loaded here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier

from src.config import (
    FEATURE_COLUMNS,
    MODELS_DIR,
    OPTIMIZED_HP_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TICKERS,
    TRAIN_END,
    VAL_END,
    model_path,
)
from src.data_loader import chronological_split, load_processed_data


# ── Model builders ────────────────────────────────────────────────────────────

def _scale_pos_weight(y: pd.Series) -> float:
    """
    XGBoost's class-balance parameter.
    Equivalent to sklearn's class_weight='balanced' but XGBoost-native.
    XGBoost silently ignores class_weight — this is the correct fix.
    """
    n_down = int((y == 0).sum())
    n_up   = int((y == 1).sum())
    return round(n_down / n_up, 4) if n_up > 0 else 1.0


def build_model(
    model_name: str,
    params: dict[str, Any],
    y_train: pd.Series,
) -> Any:
    """
    Instantiate the correct estimator with optimised hyperparameters.

    Parameters
    ----------
    model_name : "XGBoost", "LightGBM", or "RandomForest"
    params     : hyperparameter dict from optimized_hyperparameters.json
    y_train    : training labels (needed to compute scale_pos_weight for XGBoost)
    """
    shared = dict(random_state=RANDOM_STATE, n_jobs=-1)

    if model_name == "XGBoost":
        return XGBClassifier(
            **params,
            scale_pos_weight  = _scale_pos_weight(y_train),
            use_label_encoder = False,
            eval_metric       = "logloss",
            verbosity         = 0,
            **shared,
        )

    if model_name == "LightGBM":
        return LGBMClassifier(
            **params,
            class_weight = "balanced",
            verbose      = -1,
            **shared,
        )

    if model_name == "RandomForest":
        return RandomForestClassifier(
            **params,
            class_weight = "balanced",
            **shared,
        )

    raise ValueError(
        f"Unknown model_name: '{model_name}'. "
        "Expected one of: XGBoost, LightGBM, RandomForest."
    )


# ── Training logic ────────────────────────────────────────────────────────────

def train_ticker(ticker: str, optimized_hp: dict[str, Any]) -> dict[str, Any]:
    """
    Train the production model for one ticker.

    Strategy:
    - Train on the combined train + val window (everything up to 2024-12-31).
    - This gives the model ~1,390 rows vs the original 1,150 — a meaningful
      15% increase in signal before the model is deployed to serve 2025+ data.
    - The test set (> 2024-12-31) is never loaded.

    Returns a summary dict for the progress report.
    """
    if ticker not in optimized_hp:
        raise KeyError(
            f"{ticker} not found in optimized_hyperparameters.json. "
            "Run notebook 04 first."
        )

    hp_entry   = optimized_hp[ticker]
    model_name = hp_entry["model"]
    params     = {k: v for k, v in hp_entry["params"].items()}

    print(f"\n  {'─'*55}")
    print(f"  {ticker}  →  {model_name}")
    print(f"  {'─'*55}")

    # ── Load and split ────────────────────────────────────────────────────────
    df = load_processed_data(ticker)
    X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(df)

    # Combine train + val for production training
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])

    print(f"  Train+Val rows : {len(X_trainval):,}")
    print(f"  Test rows      : {len(X_test):,}  (not loaded — sealed)")
    print(f"  Target balance : {y_trainval.mean()*100:.1f}% UP")
    print(f"  Hyperparameters:")
    for k, v in params.items():
        print(f"    {k:<22} = {v}")

    # ── Build and fit ─────────────────────────────────────────────────────────
    model = build_model(model_name, params, y_trainval)
    model.fit(X_trainval, y_trainval)

    # ── Quick self-check on train+val (not a performance estimate) ────────────
    train_proba  = model.predict_proba(X_trainval)[:, 1]
    train_roc    = roc_auc_score(y_trainval, train_proba)

    # ── Save artifact ─────────────────────────────────────────────────────────
    save_path = model_path(ticker)
    joblib.dump(model, save_path)

    print(f"  Train+Val ROC-AUC : {train_roc:.4f}  (self-check only, not test performance)")
    print(f"  Saved → {save_path}")

    return {
        "ticker":         ticker,
        "model":          model_name,
        "trainval_rows":  len(X_trainval),
        "trainval_roc":   round(train_roc, 4),
        "artifact":       str(save_path),
    }


def load_optimized_hyperparameters() -> dict[str, Any]:
    """Load and validate the optimized hyperparameters JSON."""
    if not OPTIMIZED_HP_PATH.exists():
        raise FileNotFoundError(
            f"optimized_hyperparameters.json not found at {OPTIMIZED_HP_PATH}.\n"
            "Run notebook 04 (hyperparameter tuning) first."
        )
    return json.loads(OPTIMIZED_HP_PATH.read_text())


def train_all(tickers: list[str] | None = None) -> list[dict[str, Any]]:
    """
    Train and save models for all (or a subset of) tickers.

    Parameters
    ----------
    tickers : list of ticker strings, or None to train all TICKERS.
    """
    targets       = tickers or TICKERS
    optimized_hp  = load_optimized_hyperparameters()
    results       = []

    print(f"\n{'═'*58}")
    print(f"  EGX Production Model Training")
    print(f"  Training window : 2019-01-01 → {VAL_END} (train + val combined)")
    print(f"  Tickers         : {targets}")
    print(f"{'═'*58}")

    for ticker in targets:
        try:
            summary = train_ticker(ticker, optimized_hp)
            results.append(summary)
        except Exception as exc:
            print(f"\n  ✗ {ticker} failed: {exc}")
            results.append({"ticker": ticker, "error": str(exc)})

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n\n{'═'*58}")
    print(f"  Training complete — {len([r for r in results if 'error' not in r])}/{len(targets)} succeeded")
    print(f"{'═'*58}")
    print(f"  {'Ticker':<12} {'Model':<16} {'Rows':>8} {'TrainVal ROC':>14}  Artifact")
    print(f"  {'─'*58}")
    for r in results:
        if "error" in r:
            print(f"  {r['ticker']:<12} ✗ {r['error']}")
        else:
            print(
                f"  {r['ticker']:<12} {r['model']:<16} {r['trainval_rows']:>8,} "
                f"{r['trainval_roc']:>14.4f}  {Path(r['artifact']).name}"
            )
    return results


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train EGX production models from optimized hyperparameters."
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        choices=TICKERS,
        help="Tickers to train. Defaults to all 5 tickers.",
    )
    args = parser.parse_args()
    train_all(tickers=args.tickers)
