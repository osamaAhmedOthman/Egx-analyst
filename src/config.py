"""
config.py — Single source of truth for all project constants.

Every other module imports from here. Nothing is hardcoded anywhere else.
If you add a ticker, change a split date, or update a feature, this is
the only file you touch.
"""
from __future__ import annotations

from pathlib import Path

# ── Tickers ───────────────────────────────────────────────────────────────────
# EKHW_CA was delisted — replaced by HRHO_CA (EFG Holding)
TICKERS: list[str] = [
    "COMI_CA",  # Commercial International Bank
    "HRHO_CA",  # EFG Holding (replaces delisted EKHW_CA)
    "TMGH_CA",  # Talaat Moustafa Group
    "SWDY_CA",  # Elsewedy Electric
    "ORWE_CA",  # Oriental Weavers
]

# ── Feature contract ──────────────────────────────────────────────────────────
# Must match exactly what notebook 02 / src/features.py produces.
# Order matters — models were trained on this exact column sequence.
FEATURE_COLUMNS: list[str] = [
    # Momentum & trend
    "RSI_14",
    "MACD_Norm",
    "MACD_Signal_Norm",
    "MACD_Hist_Norm",
    # Volatility
    "Bollinger_Width",
    "Bollinger_Position",
    "ATR_Pct",
    # Price memory (returns, not price levels)
    "Return_Lag_1",
    "Return_Lag_5",
    "Return_Lag_10",
    "Return_Lag_21",
    # Rolling structure
    "Close_MA5_Ratio",
    "Close_MA21_Ratio",
    "Close_CV5",
    "Close_CV21",
    # Volume
    "Volume_Ratio",
    "Volume_Spike",
    # Calendar / seasonality
    "Day_Of_Week",
    "Month",
    "is_Ramadan",
]

TARGET_COLUMN: str = "Target"

# ── Chronological split anchors ───────────────────────────────────────────────
# These are fixed — changing them invalidates all saved models.
TRAIN_END: str = "2023-12-31"   # training window upper bound (inclusive)
VAL_END:   str = "2024-12-31"   # validation window upper bound (inclusive)
# Test set: everything after VAL_END — used only for final evaluation

# ── Inference data requirements ───────────────────────────────────────────────
# Minimum lookback days needed to compute all rolling features.
# The longest window is 26-day EMA (MACD) + 21-day rolling.
# We fetch LOOKBACK_DAYS to guarantee enough rows after market holidays.
LOOKBACK_DAYS: int = 90

# ── Per-ticker prediction thresholds ─────────────────────────────────────────
# Probability threshold above which the model predicts UP (1).
# Determined via precision_recall_curve on the 2024 validation set.
# ORWE uses a tuned threshold (0.4743); all others use the default 0.50
# because their "optimal" threshold from PR curve was degenerate
# (recall ≈ 1.0, meaning predict-all-UP).
PREDICTION_THRESHOLDS: dict[str, float] = {
    "COMI_CA": 0.50,
    "HRHO_CA": 0.50,
    "TMGH_CA": 0.50,
    "SWDY_CA": 0.50,
    "ORWE_CA": 0.4743,
}

# ── Random state ──────────────────────────────────────────────────────────────
RANDOM_STATE: int = 42

# ── Paths ─────────────────────────────────────────────────────────────────────
# All paths are derived from the project root so they work regardless of
# whether code is called from notebooks/, src/, api/, or the project root.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

RAW_DIR:       Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
MODELS_DIR:    Path = PROJECT_ROOT / "models"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

WINNER_CONFIG_PATH:  Path = MODELS_DIR / "winner_config.json"
OPTIMIZED_HP_PATH:   Path = MODELS_DIR / "optimized_hyperparameters.json"

def model_path(ticker: str) -> Path:
    """Return the path to the saved .joblib model for a given ticker."""
    return MODELS_DIR / f"{ticker}_best_model.joblib"

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI:        str = "mlruns"
MLFLOW_EXPERIMENT_BASELINE: str = "EGX_Directional_Prediction"
MLFLOW_EXPERIMENT_TUNING:   str = "EGX_Hyperparameter_Tuning"

# ── Data download ─────────────────────────────────────────────────────────────
DATA_START_DATE: str = "2019-01-01"   # full history download start
YFINANCE_SUFFIX: str = ".CA"          # EGX ticker suffix on yfinance

# ── Ramadan windows ───────────────────────────────────────────────────────────
# Used by features.py to build the is_Ramadan calendar flag.
# Extend this dict each year.
RAMADAN_RANGES: dict[int, tuple[str, str]] = {
    2019: ("2019-05-06", "2019-06-04"),
    2020: ("2020-04-24", "2020-05-23"),
    2021: ("2021-04-13", "2021-05-12"),
    2022: ("2022-04-02", "2022-05-01"),
    2023: ("2023-03-23", "2023-04-21"),
    2024: ("2024-03-11", "2024-04-09"),
    2025: ("2025-03-01", "2025-03-29"),
    2026: ("2026-02-18", "2026-03-19"),
}
