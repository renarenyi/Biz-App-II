"""
nlp/schemas.py
--------------
Phase 2 typed data contracts for the NLP / Sentiment layer.

Three schemas form the data spine of Phase 2:

  NewsItem              : normalized article ready for NLP inference
                          (aliases Phase 1's NewsArticle with explicit
                          headline + snippet field for LLM consumption)

  ArticleSentimentResult: per-article inference output from any provider

  TickerSentimentResult : aggregated ticker-level signal consumed by Phase 3

All TypedDicts use total=False so they can be constructed incrementally;
see validate_* helpers to enforce required fields at runtime.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from typing import Literal, Optional, TypedDict

logger = logging.getLogger(__name__)

# Sentinel for the three allowed sentiment labels
SentimentLabel = Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]

# Allowed coarse event types (non-exhaustive; "unknown" is the safe default)
EventType = Literal[
    "earnings_beat",
    "earnings_miss",
    "guidance_raise",
    "guidance_cut",
    "recall",
    "lawsuit",
    "regulation",
    "merger_acquisition",
    "product_launch",
    "executive_change",
    "macro",
    "unknown",
]


# =========================================================================== #
# NewsItem — input contract for the sentiment pipeline                        #
# =========================================================================== #

class NewsItem(TypedDict, total=False):
    """
    A single news item ready for NLP inference.

    Minimum required fields: ticker, headline, published_at.
    The 'analysis_text' field is computed by preprocessing and
    combines headline + snippet into a single string for the model.
    """
    ticker: str
    headline: str
    snippet: Optional[str]          # summary / body excerpt from Phase 1
    source: Optional[str]           # publication name, e.g. "Reuters"
    published_at: datetime          # UTC-aware
    url: Optional[str]
    analysis_text: Optional[str]    # combined headline + snippet (set by preprocessing)
    content_hash: Optional[str]     # SHA-256[:16] of ticker+headline+snippet+pub_at


NEWS_ITEM_REQUIRED = {"ticker", "headline", "published_at"}


def validate_news_item(item: dict, raise_on_error: bool = False) -> bool:
    """
    Verify that a NewsItem dict meets minimum requirements.

    Parameters
    ----------
    item : dict
    raise_on_error : bool

    Returns
    -------
    bool
    """
    missing = NEWS_ITEM_REQUIRED - set(item.keys())
    if missing:
        msg = f"NewsItem missing required fields: {missing}"
        if raise_on_error:
            raise ValueError(msg)
        logger.debug(msg)
        return False

    if not str(item.get("headline", "")).strip():
        msg = "NewsItem has empty headline."
        if raise_on_error:
            raise ValueError(msg)
        logger.debug(msg)
        return False

    return True


def compute_content_hash(item: dict) -> str:
    """
    Deterministic SHA-256 hash keyed on fields that define article identity.

    Hash inputs: ticker | headline | snippet | published_at (ISO)

    Used as the inference cache key so identical articles are never
    re-classified across sessions.

    Parameters
    ----------
    item : dict

    Returns
    -------
    str : 16-character hex prefix of SHA-256
    """
    parts = [
        str(item.get("ticker", "")),
        str(item.get("headline", "")),
        str(item.get("snippet", "") or ""),
        str(item.get("published_at", "") or ""),
    ]
    raw = "|".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


# =========================================================================== #
# ArticleSentimentResult — per-article inference output                       #
# =========================================================================== #

class ArticleSentimentResult(TypedDict, total=False):
    """
    Sentiment classification for a single news article.

    Produced by any model provider; must match this schema before
    the aggregation layer sees it.
    """
    ticker: str
    headline: str
    sentiment: SentimentLabel           # "POSITIVE" | "NEGATIVE" | "NEUTRAL"
    conviction_score: float             # 0.0 – 10.0
    reasoning: Optional[str]           # short rationale grounded in text
    event_type: Optional[EventType]     # coarse event label
    published_at: Optional[datetime]    # UTC-aware
    source: Optional[str]
    provider: Optional[str]             # which model produced this result
    content_hash: Optional[str]         # links back to NewsItem cache key
    inferred_at: Optional[datetime]     # UTC timestamp of inference


ARTICLE_RESULT_REQUIRED = {"ticker", "headline", "sentiment", "conviction_score"}

VALID_SENTIMENT_LABELS: set[str] = {"POSITIVE", "NEGATIVE", "NEUTRAL"}
VALID_EVENT_TYPES: set[str] = {
    "earnings_beat", "earnings_miss", "guidance_raise", "guidance_cut",
    "recall", "lawsuit", "regulation", "merger_acquisition",
    "product_launch", "executive_change", "macro", "unknown",
}


def validate_article_result(result: dict, raise_on_error: bool = False) -> bool:
    """
    Verify that an ArticleSentimentResult is well-formed.

    Checks required fields, sentiment label validity, and conviction range.

    Parameters
    ----------
    result : dict
    raise_on_error : bool

    Returns
    -------
    bool
    """
    missing = ARTICLE_RESULT_REQUIRED - set(result.keys())
    if missing:
        msg = f"ArticleSentimentResult missing required fields: {missing}"
        if raise_on_error:
            raise ValueError(msg)
        logger.debug(msg)
        return False

    sentiment = result.get("sentiment", "")
    if sentiment not in VALID_SENTIMENT_LABELS:
        msg = f"Invalid sentiment label '{sentiment}'. Must be one of {VALID_SENTIMENT_LABELS}."
        if raise_on_error:
            raise ValueError(msg)
        logger.debug(msg)
        return False

    score = result.get("conviction_score")
    if score is None or not (0.0 <= float(score) <= 10.0):
        msg = f"conviction_score '{score}' out of range [0, 10]."
        if raise_on_error:
            raise ValueError(msg)
        logger.debug(msg)
        return False

    return True


def clamp_conviction(score: float) -> float:
    """Clamp a raw conviction score to [0.0, 10.0]."""
    return max(0.0, min(10.0, float(score)))


# =========================================================================== #
# TickerSentimentResult — aggregated ticker-level output for Phase 3          #
# =========================================================================== #

class TickerSentimentResult(TypedDict, total=False):
    """
    Aggregated sentiment signal for one ticker over a defined time window.

    This is the Phase 3 consumption contract. The strategy engine should
    require only: ticker, sentiment, conviction_score, generated_at.
    All other fields are metadata for explainability and auditing.
    """
    ticker: str
    sentiment: SentimentLabel           # dominant label after aggregation
    conviction_score: float             # 0.0 – 10.0 weighted aggregate
    reasoning: Optional[str]           # human-readable summary of signal basis
    source_count: int                   # total articles analyzed
    unique_event_count: int             # distinct event clusters identified
    analysis_window_hours: int          # window used (e.g. 24)
    generated_at: datetime              # UTC timestamp of aggregation
    provider_used: Optional[str]        # primary provider that handled inference
    article_results: Optional[list]     # list[ArticleSentimentResult] — optional detail


TICKER_RESULT_REQUIRED = {"ticker", "sentiment", "conviction_score", "generated_at"}


def validate_ticker_result(result: dict, raise_on_error: bool = False) -> bool:
    """
    Verify a TickerSentimentResult dict meets minimum requirements.

    Parameters
    ----------
    result : dict
    raise_on_error : bool

    Returns
    -------
    bool
    """
    missing = TICKER_RESULT_REQUIRED - set(result.keys())
    if missing:
        msg = f"TickerSentimentResult missing required fields: {missing}"
        if raise_on_error:
            raise ValueError(msg)
        logger.debug(msg)
        return False

    sentiment = result.get("sentiment", "")
    if sentiment not in VALID_SENTIMENT_LABELS:
        msg = f"Invalid sentiment label '{sentiment}' in TickerSentimentResult."
        if raise_on_error:
            raise ValueError(msg)
        logger.debug(msg)
        return False

    return True


def make_empty_ticker_result(ticker: str, window_hours: int = 24) -> dict:
    """
    Return a neutral TickerSentimentResult for use when no articles exist.

    The strategy engine interprets this as: do not trade.

    Parameters
    ----------
    ticker : str
    window_hours : int

    Returns
    -------
    dict (TickerSentimentResult)
    """
    from datetime import timezone
    return {
        "ticker": ticker,
        "sentiment": "NEUTRAL",
        "conviction_score": 0.0,
        "reasoning": "No articles found for the analysis window. Signal withheld.",
        "source_count": 0,
        "unique_event_count": 0,
        "analysis_window_hours": window_hours,
        "generated_at": datetime.now(tz=timezone.utc),
        "provider_used": None,
        "article_results": [],
    }
