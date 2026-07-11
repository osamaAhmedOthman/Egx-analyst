"""
features.py — Feature engineering pipeline.

Single public function: add_technical_features(frame) → pd.DataFrame

Consumed by:
  - src/train.py        (training on historical data)
  - src/predict.py      (live inference on freshly downloaded data)
  - api/routers/predict.py (indirectly via src/predict.py)

Design contract:
  - Input : raw OHLCV DataFrame with DatetimeIndex, columns
            [Close, High, Low, Open, Volume], sorted ascending.
  - Output: same DataFrame with FEATURE_COLUMNS appended.
            All features are scale-invariant (ratios, returns, bounded
            indices) — never raw EGP values.
  - The Target column is only added during training (add_target=True).
    During live inference there is no next-day close, so add_target=False.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, RAMADAN_RANGES, TARGET_COLUMN


# ── Internal helpers ──────────────────────────────────────────────────────────

def _calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Wilder-smoothed RSI. Returns values in [0, 100].
    NaNs (first `window` rows) are filled with 50 (neutral midpoint)
    so they do not propagate into downstream features.
    """
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)


def _calculate_atr(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    14-day Average True Range in raw price units.
    Used internally only — normalised by Close before being kept as ATR_Pct.
    """
    prev_close = frame["Close"].shift(1)
    true_range = pd.concat(
        [
            frame["High"] - frame["Low"],
            (frame["High"] - prev_close).abs(),
            (frame["Low"]  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window=window, min_periods=window).mean()


def _build_ramadan_flag(index: pd.DatetimeIndex) -> pd.Series:
    """Binary flag: 1 if the trading day falls inside a Ramadan window."""
    mask = pd.Series(False, index=index)
    for start_str, end_str in RAMADAN_RANGES.values():
        mask = mask | (
            (index >= pd.Timestamp(start_str)) &
            (index <= pd.Timestamp(end_str))
        )
    return mask.astype("Int64")


# ── Public API ────────────────────────────────────────────────────────────────

def add_technical_features(
    frame: pd.DataFrame,
    add_target: bool = False,
) -> pd.DataFrame:
    """
    Compute all 20 scale-invariant features from raw OHLCV data.

    Parameters
    ----------
    frame : pd.DataFrame
        Raw OHLCV data with DatetimeIndex sorted ascending.
        Required columns: Close, High, Low, Open, Volume.
    add_target : bool
        If True, appends the binary Target column (1 = next-day UP).
        Set True during training, False during live inference.

    Returns
    -------
    pd.DataFrame
        Original frame plus all feature columns (and optionally Target).
        Caller should dropna() after this call during training.
    """
    f = frame.copy()

    # Guard against division by zero across all price-normalisation steps
    close_safe = f["Close"].replace(0, np.nan)

    # ── Target (training only) ────────────────────────────────────────────────
    # 1 if tomorrow's close is strictly above today's close, else 0.
    # shift(-1): the last row gets NaN → dropped by caller.
    if add_target:
        f[TARGET_COLUMN] = (f["Close"].shift(-1) > f["Close"]).astype("Int64")

    # ── RSI ───────────────────────────────────────────────────────────────────
    # Already bounded [0, 100] — no normalisation needed.
    f["RSI_14"] = _calculate_rsi(f["Close"], window=14)

    # ── MACD — normalised by Close ────────────────────────────────────────────
    # Raw MACD is in EGP. Dividing by Close converts to a fraction of price
    # so it's comparable across tickers and across a ticker's own history.
    ema_12   = f["Close"].ewm(span=12, adjust=False).mean()
    ema_26   = f["Close"].ewm(span=26, adjust=False).mean()
    macd_raw = ema_12 - ema_26
    sig_raw  = macd_raw.ewm(span=9, adjust=False).mean()
    hist_raw = macd_raw - sig_raw
    f["MACD_Norm"]        = macd_raw / close_safe
    f["MACD_Signal_Norm"] = sig_raw  / close_safe
    f["MACD_Hist_Norm"]   = hist_raw / close_safe  # positive = bullish crossover

    # ── Bollinger Bands — position and width, not raw price levels ────────────
    # Bollinger_Width    : (Upper - Lower) / Mid  — band width as % of price
    # Bollinger_Position : (Close - Lower) / (Upper - Lower)
    #                      0 = at lower band, 1 = at upper band, >1 = breakout
    roll20    = f["Close"].rolling(window=20, min_periods=20)
    bb_mid    = roll20.mean()
    bb_std    = roll20.std()
    bb_upper  = bb_mid + 2 * bb_std
    bb_lower  = bb_mid - 2 * bb_std
    bb_width  = bb_upper - bb_lower
    f["Bollinger_Width"]    = bb_width  / bb_mid.replace(0, np.nan)
    f["Bollinger_Position"] = (f["Close"] - bb_lower) / (bb_width + 1e-9)

    # ── ATR — normalised by Close ─────────────────────────────────────────────
    # ATR in EGP is meaningless across tickers at different price levels.
    # ATR_Pct expresses daily volatility as a fraction of current price.
    atr_raw   = _calculate_atr(f, window=14)
    f["ATR_Pct"] = atr_raw / close_safe

    # ── Return lags — percentage returns, not price levels ────────────────────
    # pct_change(n) = Close_today / Close_{today-n} - 1
    # These replace raw Close_Lag features which are price-denominated.
    for lag in [1, 5, 10, 21]:
        f[f"Return_Lag_{lag}"] = f["Close"].pct_change(lag)

    # ── Rolling structure — ratios and coefficients of variation ──────────────
    # Close_MAn_Ratio : how far price has extended above its n-day average
    #                   >1 = above average (bullish extension)
    # Close_CVn       : std / mean over n days — relative price dispersion
    for window in [5, 21]:
        roll      = f["Close"].rolling(window=window, min_periods=window)
        roll_mean = roll.mean()
        f[f"Close_MA{window}_Ratio"] = f["Close"] / roll_mean.replace(0, np.nan)
        f[f"Close_CV{window}"]       = roll.std()  / roll_mean.replace(0, np.nan)

    # ── Volume features ───────────────────────────────────────────────────────
    # Volume_Ratio : today's volume vs its 21-day average (scale-invariant)
    # Volume_Spike : binary flag — 1 if volume > 2.5× 21-day average
    #                Indicates unusual institutional activity.
    vol_ma21         = f["Volume"].rolling(window=21, min_periods=21).mean()
    f["Volume_Ratio"] = f["Volume"] / vol_ma21.replace(0, np.nan)
    f["Volume_Spike"] = (f["Volume"] > 2.5 * vol_ma21).astype("Int64")

    # ── Calendar / seasonality ────────────────────────────────────────────────
    # Day_Of_Week: EGX trades Sun–Thu; pandas dayofweek gives 0=Mon…6=Sun
    #              so EGX sessions land on 0,1,2,3,6.
    # Month      : captures macro seasonality (post-Ramadan, fiscal year-end).
    # is_Ramadan : EGX-specific liquidity regime flag.
    f["Day_Of_Week"] = f.index.dayofweek.astype("Int64")
    f["Month"]       = f.index.month.astype("Int64")
    f["is_Ramadan"]  = _build_ramadan_flag(f.index)

    return f


def select_features(
    frame: pd.DataFrame,
    include_target: bool = False,
) -> pd.DataFrame:
    """
    Return only the model feature columns (and optionally Target)
    from a fully-engineered DataFrame.

    Raises ValueError if any expected column is missing so errors
    surface early rather than silently producing wrong predictions.
    """
    cols = (FEATURE_COLUMNS + [TARGET_COLUMN]) if include_target else FEATURE_COLUMNS
    missing = set(cols) - set(frame.columns)
    if missing:
        raise ValueError(
            f"Feature contract violation — missing columns: {sorted(missing)}\n"
            "Ensure add_technical_features() was called before select_features()."
        )
    return frame[cols].copy()
