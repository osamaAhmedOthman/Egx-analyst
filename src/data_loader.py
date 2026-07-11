"""
data_loader.py — Download, clean, and chronologically split EGX stock data.

Public functions:
  download_raw_data(ticker)             → pd.DataFrame  (OHLCV, clean)
  load_processed_data(ticker)           → pd.DataFrame  (feature-engineered parquet)
  chronological_split(df)               → train/val/test splits
  fetch_recent_for_inference(ticker)    → pd.DataFrame  (live OHLCV for predict.py)
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Tuple

import pandas as pd
import yfinance as yf

from src.config import (
    DATA_START_DATE,
    LOOKBACK_DAYS,
    PROCESSED_DIR,
    RAW_DIR,
    TICKERS,
    TRAIN_END,
    VAL_END,
    YFINANCE_SUFFIX,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance >= 0.2.40 returns a MultiIndex column header:
        level 0: Price  (Close, High, Low, Open, Volume)
        level 1: Ticker (COMI.CA, COMI.CA, ...)

    This flattens it to a single level with just the price names.
    If the columns are already flat, this is a no-op.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _validate_ohlcv(df: pd.DataFrame, ticker: str) -> None:
    """Raise if required OHLCV columns are missing."""
    required = {"Close", "High", "Low", "Open", "Volume"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{ticker}: downloaded data is missing columns {sorted(missing)}. "
            "The ticker may be delisted or the yfinance API response has changed."
        )


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply standard cleaning steps to raw OHLCV data:
    1. Keep only the five OHLCV columns.
    2. Remove rows where Volume == 0.
       These are exchange holidays / weekends that sneak through yfinance.
       They carry no trading signal and would corrupt rolling-window features.
    3. Drop any remaining NaN rows.
    4. Sort by date ascending (yfinance is usually sorted but not guaranteed).
    """
    df = df.loc[:, ["Close", "High", "Low", "Open", "Volume"]].copy()
    df = df[df["Volume"] > 0]
    df = df.dropna()
    df = df.sort_index()
    return df


# ── Public API ────────────────────────────────────────────────────────────────

def download_raw_data(
    ticker: str,
    start: str = DATA_START_DATE,
    save_csv: bool = True,
) -> pd.DataFrame:
    """
    Download full OHLCV history for one EGX ticker from yfinance.

    Parameters
    ----------
    ticker    : Ticker in project format, e.g. "COMI_CA".
                The function converts underscores to dots and appends .CA suffix.
    start     : ISO date string — download start date.
    save_csv  : If True, saves cleaned data to data/raw/{ticker}.csv.

    Returns
    -------
    pd.DataFrame with DatetimeIndex and columns [Close, High, Low, Open, Volume].
    """
    yf_ticker = ticker.replace("_", ".") + ""   # already has .CA: "COMI_CA" → "COMI.CA"
    # COMI_CA → COMI.CA  (underscore was used instead of dot in filenames)
    yf_ticker = ticker.replace("_", ".")

    print(f"  Downloading {yf_ticker} from {start} ...")
    raw = yf.download(yf_ticker, start=start, auto_adjust=True, progress=False)

    if raw.empty:
        raise RuntimeError(
            f"yfinance returned empty data for {yf_ticker}. "
            "The ticker may be delisted. Check TICKERS in config.py."
        )

    raw = _flatten_multiindex_columns(raw)
    _validate_ohlcv(raw, ticker)
    df  = _clean_ohlcv(raw)

    print(f"  {ticker}: {len(df)} rows  ({df.index[0].date()} → {df.index[-1].date()})")

    if save_csv:
        path = RAW_DIR / f"{ticker}.csv"
        df.to_csv(path)
        print(f"  Saved → {path}")

    return df


def download_all_tickers(start: str = DATA_START_DATE) -> dict[str, pd.DataFrame]:
    """Download and save raw data for all tickers in TICKERS."""
    results: dict[str, pd.DataFrame] = {}
    print(f"Downloading {len(TICKERS)} tickers from {start} ...\n")
    for ticker in TICKERS:
        try:
            results[ticker] = download_raw_data(ticker, start=start, save_csv=True)
        except Exception as exc:
            print(f"  ✗ {ticker}: {exc}")
    print(f"\n✓ Downloaded {len(results)}/{len(TICKERS)} tickers successfully.")
    return results


def load_raw_csv(ticker: str) -> pd.DataFrame:
    """
    Load a previously saved raw CSV from data/raw/.
    Use this instead of re-downloading during development.
    """
    path = RAW_DIR / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Raw CSV not found: {path}\n"
            "Run download_raw_data() or download_all_tickers() first."
        )
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    df = _clean_ohlcv(df)
    return df


def load_processed_data(ticker: str) -> pd.DataFrame:
    """
    Load the feature-engineered parquet for one ticker from data/processed/.
    This is what train.py and the split functions consume.
    """
    path = PROCESSED_DIR / f"{ticker}_features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed parquet not found: {path}\n"
            "Run notebooks/02_feature_engineering.ipynb first."
        )
    df = pd.read_parquet(path).sort_index()

    # Schema validation — catch drift early
    expected = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing  = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"{ticker}: parquet is missing columns {sorted(missing)}.\n"
            "Re-run notebook 02 to regenerate the processed data."
        )
    return df


def chronological_split(
    df: pd.DataFrame,
    train_end: str = TRAIN_END,
    val_end:   str = VAL_END,
) -> Tuple[
    pd.DataFrame, pd.Series,   # X_train, y_train
    pd.DataFrame, pd.Series,   # X_val,   y_val
    pd.DataFrame, pd.Series,   # X_test,  y_test
]:
    """
    Strict date-anchored train / val / test split.

    No shuffling. No leakage. The split boundaries come from config.py
    so they are consistent across every module that calls this function.

    Parameters
    ----------
    df        : Feature-engineered DataFrame with DatetimeIndex.
    train_end : Upper bound of training window (inclusive).
    val_end   : Upper bound of validation window (inclusive).
                Everything after val_end is the test set.

    Returns
    -------
    Six objects: X_train, y_train, X_val, y_val, X_test, y_test
    """
    train_mask = df.index <= train_end
    val_mask   = (df.index > train_end) & (df.index <= val_end)
    test_mask  = df.index > val_end

    def _split_xy(mask: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        subset = df.loc[mask]
        X = subset[FEATURE_COLUMNS].copy()
        y = subset[TARGET_COLUMN].astype(int).copy()
        return X, y

    X_train, y_train = _split_xy(train_mask)
    X_val,   y_val   = _split_xy(val_mask)
    X_test,  y_test  = _split_xy(test_mask)

    return X_train, y_train, X_val, y_val, X_test, y_test


def fetch_recent_for_inference(ticker: str) -> pd.DataFrame:
    """
    Fetch the most recent LOOKBACK_DAYS of OHLCV data for live inference.

    The feature pipeline needs at least 26 days for MACD EMA and 21 days
    for rolling windows. LOOKBACK_DAYS (90) provides a comfortable buffer
    that accounts for weekends and Egyptian market holidays.

    Returns a clean OHLCV DataFrame ready to be passed to
    add_technical_features() in src/features.py.
    """
    import datetime

    yf_ticker = ticker.replace("_", ".")
    end_date  = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=LOOKBACK_DAYS)

    raw = yf.download(
        yf_ticker,
        start=str(start_date),
        end=str(end_date),
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise RuntimeError(
            f"yfinance returned no recent data for {yf_ticker}. "
            "Check your internet connection or whether the market is closed."
        )

    raw = _flatten_multiindex_columns(raw)
    _validate_ohlcv(raw, ticker)
    df  = _clean_ohlcv(raw)

    if len(df) < 30:
        raise RuntimeError(
            f"{ticker}: only {len(df)} rows in the last {LOOKBACK_DAYS} days. "
            "Not enough data to compute rolling features reliably."
        )

    return df
