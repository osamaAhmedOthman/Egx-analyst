"""
predict.py — Live inference pipeline.

Single public function: predict(ticker) → PredictionResult

Full flow:
  1. Fetch the last LOOKBACK_DAYS of OHLCV data from yfinance
  2. Run the feature engineering pipeline (add_technical_features)
  3. Select the most recent complete row (today's features)
  4. Load the fitted .joblib model for this ticker
  5. Call predict_proba — get the UP probability
  6. Apply the ticker-specific threshold from optimized_hyperparameters.json
  7. Return direction, confidence, and metadata

Called by:
  - api/routers/predict.py   (FastAPI endpoint)
  - agent/tools.py           (LangGraph tool node)
  - CLI: python -m src.predict --ticker COMI_CA
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import (
    FEATURE_COLUMNS,
    MODELS_DIR,
    OPTIMIZED_HP_PATH,
    PREDICTION_THRESHOLDS,
    TARGET_COLUMN,
    TICKERS,
    model_path,
)
from src.data_loader import fetch_recent_for_inference
from src.features import add_technical_features, select_features


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    """
    Structured output of the predict() function.

    Attributes
    ----------
    ticker          : EGX ticker string, e.g. "COMI_CA"
    direction       : "UP" or "DOWN" — next-day predicted direction
    confidence      : probability of the predicted direction (0.0–1.0)
    up_probability  : raw model output — P(next day UP)
    threshold       : decision boundary used for this ticker
    prediction_date : the trading date whose features were used
    model_name      : architecture that produced this prediction
    feature_snapshot: dict of the feature values fed to the model
    """
    ticker          : str
    direction       : str
    confidence      : float
    up_probability  : float
    threshold       : float
    prediction_date : str
    model_name      : str
    feature_snapshot: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker":           self.ticker,
            "direction":        self.direction,
            "confidence":       round(self.confidence, 4),
            "up_probability":   round(self.up_probability, 4),
            "threshold":        round(self.threshold, 4),
            "prediction_date":  self.prediction_date,
            "model_name":       self.model_name,
            "feature_snapshot": {
                k: round(float(v), 6) if pd.notna(v) else None
                for k, v in self.feature_snapshot.items()
            },
        }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_model(ticker: str) -> Any:
    """Load the fitted joblib model for one ticker."""
    path = model_path(ticker)
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {path}\n"
            "Run 'python -m src.train' to generate .joblib files."
        )
    return joblib.load(path)


def _load_threshold(ticker: str) -> float:
    """
    Load the per-ticker decision threshold.

    Priority order:
      1. optimized_hyperparameters.json (set by notebook 05 / final evaluation)
      2. PREDICTION_THRESHOLDS dict in config.py (hardcoded safe defaults)

    This two-layer fallback ensures predict.py works even if the JSON
    is missing (e.g. during testing or first-run setup).
    """
    if OPTIMIZED_HP_PATH.exists():
        hp = json.loads(OPTIMIZED_HP_PATH.read_text())
        if ticker in hp and "optimal_threshold" in hp[ticker]:
            return float(hp[ticker]["optimal_threshold"])

    # Fallback to config defaults
    if ticker in PREDICTION_THRESHOLDS:
        return PREDICTION_THRESHOLDS[ticker]

    raise KeyError(
        f"No threshold found for {ticker} in optimized_hyperparameters.json "
        "or config.PREDICTION_THRESHOLDS."
    )


def _load_model_name(ticker: str) -> str:
    """Read the model architecture name from optimized_hyperparameters.json."""
    if OPTIMIZED_HP_PATH.exists():
        hp = json.loads(OPTIMIZED_HP_PATH.read_text())
        if ticker in hp and "model" in hp[ticker]:
            return hp[ticker]["model"]
    return "Unknown"


# ── Public API ────────────────────────────────────────────────────────────────

def predict(ticker: str) -> PredictionResult:
    """
    Generate a next-day directional prediction for one EGX ticker.

    Parameters
    ----------
    ticker : Ticker in project format, e.g. "COMI_CA".

    Returns
    -------
    PredictionResult dataclass with all prediction metadata.

    Raises
    ------
    ValueError  : if ticker is not in the configured TICKERS list.
    RuntimeError: if insufficient recent data is available.
    FileNotFoundError: if the model artifact has not been trained yet.
    """
    if ticker not in TICKERS:
        raise ValueError(
            f"Unknown ticker: '{ticker}'. "
            f"Configured tickers: {TICKERS}"
        )

    # ── Step 1: Fetch recent OHLCV ────────────────────────────────────────────
    raw_df = fetch_recent_for_inference(ticker)

    # ── Step 2: Feature engineering ───────────────────────────────────────────
    # add_target=False because we have no next-day close for today.
    engineered = add_technical_features(raw_df, add_target=False)

    # Drop rows with NaN features (from rolling window warm-up period)
    engineered = engineered.dropna(subset=FEATURE_COLUMNS)

    if engineered.empty:
        raise RuntimeError(
            f"{ticker}: all rows dropped after NaN removal. "
            "The downloaded data may be too short to compute rolling features."
        )

    # ── Step 3: Select the most recent complete row ───────────────────────────
    # This represents today's market state — the input for tomorrow's prediction.
    latest_row   = engineered.iloc[[-1]]   # keep as DataFrame (1 row) not Series
    feature_row  = select_features(latest_row, include_target=False)
    prediction_date = str(latest_row.index[-1].date())

    # ── Step 4: Load model and threshold ─────────────────────────────────────
    model      = _load_model(ticker)
    threshold  = _load_threshold(ticker)
    model_name = _load_model_name(ticker)

    # ── Step 5: Predict ───────────────────────────────────────────────────────
    up_probability = float(model.predict_proba(feature_row)[0, 1])
    is_up          = up_probability >= threshold
    direction      = "UP" if is_up else "DOWN"

    # Confidence = probability of the predicted direction
    # If predicting UP:   confidence = P(UP)
    # If predicting DOWN: confidence = P(DOWN) = 1 - P(UP)
    # This makes confidence always represent certainty about the stated direction.
    confidence = up_probability if is_up else (1.0 - up_probability)

    # ── Step 6: Capture feature snapshot for transparency ────────────────────
    feature_snapshot = feature_row.iloc[0].to_dict()

    return PredictionResult(
        ticker          = ticker,
        direction       = direction,
        confidence      = confidence,
        up_probability  = up_probability,
        threshold       = threshold,
        prediction_date = prediction_date,
        model_name      = model_name,
        feature_snapshot= feature_snapshot,
    )


def predict_all() -> list[PredictionResult]:
    """Generate predictions for all 5 tickers. Failures are logged, not raised."""
    results = []
    for ticker in TICKERS:
        try:
            result = predict(ticker)
            results.append(result)
            print(
                f"  {ticker:<12} {result.direction:<5}  "
                f"confidence={result.confidence:.3f}  "
                f"P(UP)={result.up_probability:.3f}  "
                f"threshold={result.threshold:.4f}  "
                f"({result.model_name})"
            )
        except Exception as exc:
            print(f"  {ticker:<12} ✗ {exc}")
    return results


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate next-day directional predictions for EGX tickers."
    )
    parser.add_argument(
        "--ticker",
        choices=TICKERS,
        default=None,
        help="Single ticker to predict. Omit to predict all 5 tickers.",
    )
    args = parser.parse_args()

    print(f"\nEGX Directional Prediction  —  {date.today()}\n{'─'*50}")

    if args.ticker:
        result = predict(args.ticker)
        import json as _json
        print(_json.dumps(result.to_dict(), indent=2))
    else:
        predict_all()
