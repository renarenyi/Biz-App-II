"""
utils.py
--------
Shared utility functions for the Trading Agent data layer.

These are stateless helpers used by providers, handlers, and tests.
No business logic lives here — only plumbing.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
import pytz


# =========================================================================== #
# Logging                                                                      #
# =========================================================================== #

def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Return a named logger with a consistent format.

    Parameters
    ----------
    name : str
        Usually __name__ from the calling module.
    level : str, optional
        Override log level (DEBUG, INFO, WARNING, ERROR). Falls back to
        the LOG_LEVEL environment variable or INFO.
    """
    from src.config.settings import settings  # late import to avoid circular dep

    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(settings.LOG_FORMAT)
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    effective_level = level or settings.LOG_LEVEL
    logger.setLevel(getattr(logging, effective_level.upper(), logging.INFO))
    logger.propagate = False
    return logger


# =========================================================================== #
# Timestamp helpers                                                            #
# =========================================================================== #

def to_utc(dt: Any) -> Optional[datetime]:
    """
    Coerce any datetime-like value to a UTC-aware datetime.

    Accepts:
    - datetime (naive → assumed UTC; aware → converted)
    - pandas Timestamp
    - ISO 8601 string
    - Unix timestamp (int / float)
    - None → returns None

    Returns
    -------
    datetime with tzinfo=UTC, or None if input is None / unparseable.
    """
    if dt is None:
        return None

    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()

    if isinstance(dt, (int, float)):
        return datetime.fromtimestamp(dt, tz=timezone.utc)

    if isinstance(dt, str):
        try:
            dt = pd.Timestamp(dt)
            dt = dt.to_pydatetime()
        except Exception:
            return None

    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    return None


def localize_to_tz(dt: datetime, tz_name: str = "America/New_York") -> datetime:
    """
    Convert a UTC-aware datetime to the given local timezone.

    Parameters
    ----------
    dt : datetime
        Must be timezone-aware.
    tz_name : str
        pytz timezone name.
    """
    tz = pytz.timezone(tz_name)
    return dt.astimezone(tz)


def now_utc() -> datetime:
    """Return the current moment as UTC-aware datetime."""
    return datetime.now(tz=timezone.utc)


def to_pandas_timestamp_utc(dt: Any) -> Optional[pd.Timestamp]:
    """Convert to pandas Timestamp with UTC tz."""
    result = to_utc(dt)
    if result is None:
        return None
    return pd.Timestamp(result)


# =========================================================================== #
# Numeric coercion                                                             #
# =========================================================================== #

def safe_float(val: Any, default: float = float("nan")) -> float:
    """Safely cast to float, returning default on failure."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def safe_int(val: Any, default: int = 0) -> int:
    """Safely cast to int, returning default on failure."""
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


# =========================================================================== #
# Text helpers                                                                 #
# =========================================================================== #

def clean_headline(text: str) -> str:
    """
    Strip, normalize unicode, and remove excess whitespace from a headline.

    Does NOT perform stemming or lowercasing — keeps original case for LLM input.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def headline_fingerprint(headline: str) -> str:
    """
    Compute a short fingerprint for deduplication.

    Lowercases, strips punctuation, and hashes the result.
    Two headlines that differ only in whitespace or punctuation
    will share a fingerprint.
    """
    normalized = re.sub(r"[^a-z0-9 ]", "", headline.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return hashlib.md5(normalized.encode()).hexdigest()[:12]


def jaccard_similarity(a: str, b: str) -> float:
    """
    Token-level Jaccard similarity between two strings.

    Used as a cheap near-duplicate detector for headlines.

    Returns
    -------
    float in [0, 1] — 1.0 means identical token sets.
    """
    tokens_a = set(re.findall(r"\w+", a.lower()))
    tokens_b = set(re.findall(r"\w+", b.lower()))
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


# =========================================================================== #
# Collection helpers                                                           #
# =========================================================================== #

def deduplicate_by_key(items: list[dict], key: str) -> list[dict]:
    """
    Return a de-duplicated list preserving first occurrence, keyed by `key`.

    Parameters
    ----------
    items : list[dict]
    key : str
        Dict key to deduplicate on (e.g. "url" or "article_id").
    """
    seen: set = set()
    result: list[dict] = []
    for item in items:
        val = item.get(key)
        if val and val not in seen:
            seen.add(val)
            result.append(item)
        elif not val:
            # No key present — keep the item (can't deduplicate without a key)
            result.append(item)
    return result


def deduplicate_by_fingerprint(articles: list[dict], threshold: float = 0.85) -> list[dict]:
    """
    Remove near-duplicate articles using headline Jaccard similarity.

    Parameters
    ----------
    articles : list[dict]
        Each must have a 'headline' key.
    threshold : float
        Jaccard similarity above which two articles are considered duplicates.
        The later one (by index) is dropped.

    Returns
    -------
    list[dict]
        Deduplicated articles.
    """
    kept: list[dict] = []
    kept_headlines: list[str] = []

    for article in articles:
        headline = article.get("headline", "")
        is_dup = any(
            jaccard_similarity(headline, h) >= threshold for h in kept_headlines
        )
        if not is_dup:
            kept.append(article)
            kept_headlines.append(headline)

    return kept


# =========================================================================== #
# DataFrame helpers                                                            #
# =========================================================================== #

def enforce_column_types(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Cast specified columns to float64, coercing errors to NaN."""
    df = df.copy()
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def sort_by_timestamp(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    """Sort a DataFrame ascending by the given timestamp column."""
    if col in df.columns:
        return df.sort_values(col).reset_index(drop=True)
    return df


def drop_ohlcv_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with duplicate (symbol, timestamp) pairs, keeping first."""
    if {"symbol", "timestamp"}.issubset(df.columns):
        return df.drop_duplicates(subset=["symbol", "timestamp"], keep="first").reset_index(drop=True)
    return df.drop_duplicates().reset_index(drop=True)
