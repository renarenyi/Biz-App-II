"""
schemas.py
----------
Data contracts for the Trading Agent.

Every module in the data layer must return data conforming to these schemas.
Provider-specific quirks are absorbed here and never leak into strategy code.

Two primary schemas:
  - OHLCVBar     : one row of normalized market price data
  - NewsArticle  : one normalized financial news article
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, TypedDict

import pandas as pd

logger = logging.getLogger(__name__)


# =========================================================================== #
# TypedDicts — used as in-code documentation and for type checkers            #
# =========================================================================== #

class OHLCVBar(TypedDict, total=False):
    """
    Single OHLCV bar row.

    Required fields (total=False means all are optional at dict creation,
    but validate_ohlcv_df enforces the required subset on DataFrames).
    """
    timestamp: datetime          # timezone-aware
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    source: str                  # "alpaca" | "yfinance"
    # Optional / derived
    timeframe: Optional[str]     # "1Day" | "1Hour" | etc.
    currency: Optional[str]
    ingested_at: Optional[datetime]
    is_fallback: Optional[bool]


class NewsArticle(TypedDict, total=False):
    """
    Normalized financial news article.
    Every NewsArticle must have at minimum: ticker, headline, published_at, provider.
    """
    ticker: str
    headline: str
    summary: Optional[str]
    source: Optional[str]        # publication name, e.g. "Reuters"
    published_at: datetime       # timezone-aware
    url: Optional[str]
    provider: str                # "alpaca_news" | "rss" | "finviz"
    # Optional enrichment
    author: Optional[str]
    language: Optional[str]
    related_tickers: Optional[list[str]]
    article_id: Optional[str]
    ingested_at: Optional[datetime]


# =========================================================================== #
# Required column sets                                                         #
# =========================================================================== #

OHLCV_REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume", "symbol", "source"}
OHLCV_NUMERIC_COLUMNS = {"open", "high", "low", "close", "volume"}

NEWS_REQUIRED_FIELDS = {"ticker", "headline", "published_at", "provider"}


# =========================================================================== #
# Validation helpers                                                           #
# =========================================================================== #

def validate_ohlcv_df(df: pd.DataFrame, raise_on_error: bool = True) -> bool:
    """
    Validate a normalized OHLCV DataFrame.

    Checks:
    - required columns are present
    - numeric columns are numeric
    - no fully empty DataFrame
    - timestamps are timezone-aware

    Parameters
    ----------
    df : pd.DataFrame
        Candidate OHLCV DataFrame.
    raise_on_error : bool
        If True, raise ValueError on failure. Otherwise log and return False.

    Returns
    -------
    bool
        True if valid.
    """
    errors: list[str] = []

    if df is None or df.empty:
        errors.append("DataFrame is None or empty.")

    else:
        missing_cols = OHLCV_REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required OHLCV columns: {missing_cols}")

        for col in OHLCV_NUMERIC_COLUMNS:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Column '{col}' is not numeric (dtype={df[col].dtype}).")

        if "timestamp" in df.columns:
            ts_col = df["timestamp"]
            if hasattr(ts_col, "dt"):
                if ts_col.dt.tz is None:
                    errors.append("'timestamp' column is timezone-naive. Convert to UTC or a fixed tz.")
            elif len(ts_col) > 0 and isinstance(ts_col.iloc[0], datetime):
                if ts_col.iloc[0].tzinfo is None:
                    errors.append("'timestamp' column contains tz-naive datetime objects.")

    if errors:
        msg = "OHLCV validation failed:\n  " + "\n  ".join(errors)
        if raise_on_error:
            raise ValueError(msg)
        logger.warning(msg)
        return False

    return True


def validate_article(article: dict, raise_on_error: bool = False) -> bool:
    """
    Validate a single normalized article dict.

    Parameters
    ----------
    article : dict
    raise_on_error : bool

    Returns
    -------
    bool
    """
    missing = NEWS_REQUIRED_FIELDS - set(article.keys())
    if missing:
        msg = f"Article missing required fields: {missing}. Article keys: {set(article.keys())}"
        if raise_on_error:
            raise ValueError(msg)
        logger.debug(msg)
        return False

    if not article.get("headline", "").strip():
        msg = "Article has empty headline."
        if raise_on_error:
            raise ValueError(msg)
        logger.debug(msg)
        return False

    return True


# =========================================================================== #
# Normalisation helpers                                                        #
# =========================================================================== #

def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lowercase and rename common provider-specific column variants to the
    standard OHLCV schema.

    Handles quirks from Alpaca (uses 'vw', 'n') and yfinance
    (uses 'Open', 'High', etc. with capital letters).
    """
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    rename_map = {
        # yfinance capitalized
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adj close": "adj_close",
        "volume": "volume",
        # Alpaca extras we don't need at strategy level
        "vw": "vwap",
        "n": "trade_count",
        "t": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def make_empty_ohlcv_df() -> pd.DataFrame:
    """Return an empty DataFrame with the correct OHLCV schema."""
    return pd.DataFrame(
        columns=list(OHLCV_REQUIRED_COLUMNS) + ["timeframe", "currency", "ingested_at", "is_fallback"]
    )


def make_empty_articles_list() -> list[dict]:
    """Return an empty list — type signal for callers."""
    return []
