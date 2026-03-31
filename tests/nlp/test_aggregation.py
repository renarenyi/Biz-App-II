"""
tests/nlp/test_aggregation.py
------------------------------
Unit tests for nlp/aggregation.py.

Tests cover:
- Empty input returns neutral result
- All-positive batch → POSITIVE
- All-negative batch → NEGATIVE
- Mixed batch with weak signal → NEUTRAL
- Syndication cluster capping (same event, many sources)
- Recency weighting (fresh items dominate)
- Conviction score range [0, 10]
- Output schema completeness
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.nlp.aggregation import (
    aggregate_to_ticker,
    _label_to_direction,
    _direction_to_label,
    DECAY_LAMBDA,
    SENTIMENT_THRESHOLD,
)
from src.nlp.schemas import VALID_SENTIMENT_LABELS


# =========================================================================== #
# Helpers                                                                      #
# =========================================================================== #

def _now():
    return datetime.now(tz=timezone.utc)


def _result(
    ticker: str = "TSLA",
    sentiment: str = "POSITIVE",
    conviction_score: float = 8.0,
    headline: str = "TSLA earnings beat",
    hours_ago: float = 1.0,
) -> dict:
    return {
        "ticker": ticker,
        "headline": headline,
        "sentiment": sentiment,
        "conviction_score": conviction_score,
        "reasoning": f"Test reasoning for {sentiment}",
        "event_type": "earnings_beat",
        "published_at": _now() - timedelta(hours=hours_ago),
        "provider": "finbert",
    }


# =========================================================================== #
# Edge cases                                                                   #
# =========================================================================== #

def test_empty_input_returns_neutral():
    result = aggregate_to_ticker([], ticker="TSLA")
    assert result["sentiment"] == "NEUTRAL"
    assert result["conviction_score"] == 0.0
    assert result["source_count"] == 0


def test_required_fields_present():
    result = aggregate_to_ticker([], ticker="TSLA")
    required = {"ticker", "sentiment", "conviction_score", "generated_at",
                "source_count", "unique_event_count", "analysis_window_hours"}
    assert required.issubset(result.keys())


# =========================================================================== #
# Sentiment direction tests                                                    #
# =========================================================================== #

def test_all_positive_results_positive():
    results = [_result(sentiment="POSITIVE", conviction_score=9.0, headline=f"Beat news {i}")
               for i in range(5)]
    ticker_result = aggregate_to_ticker(results, ticker="TSLA")
    assert ticker_result["sentiment"] == "POSITIVE"
    assert ticker_result["conviction_score"] > 0


def test_all_negative_results_negative():
    results = [_result(sentiment="NEGATIVE", conviction_score=8.0, headline=f"Recall news {i}")
               for i in range(5)]
    ticker_result = aggregate_to_ticker(results, ticker="TSLA")
    assert ticker_result["sentiment"] == "NEGATIVE"


def test_mixed_weak_signal_neutral():
    """Balanced positive and negative with low conviction → NEUTRAL."""
    results = [
        _result(sentiment="POSITIVE", conviction_score=3.0, headline="Slight uptick"),
        _result(sentiment="NEGATIVE", conviction_score=3.0, headline="Minor concern"),
    ]
    ticker_result = aggregate_to_ticker(results, ticker="TSLA", sentiment_threshold=0.3)
    assert ticker_result["sentiment"] == "NEUTRAL"


# =========================================================================== #
# Cluster / syndication tests                                                  #
# =========================================================================== #

def test_syndication_cluster_does_not_dominate():
    """
    8 sources repeat the same recall headline → should not override
    2 positive earnings articles.
    """
    recall_results = [
        _result(
            sentiment="NEGATIVE",
            conviction_score=7.0,
            headline="Tesla recalls two million vehicles autopilot",
        )
        for _ in range(8)
    ]
    positive_results = [
        _result(
            sentiment="POSITIVE",
            conviction_score=9.0,
            headline="Tesla beats quarterly earnings with record revenue",
        ),
        _result(
            sentiment="POSITIVE",
            conviction_score=9.0,
            headline="Tesla delivery numbers exceed analyst expectations",
        ),
    ]
    all_results = recall_results + positive_results
    ticker_result = aggregate_to_ticker(all_results, ticker="TSLA")

    # The recall cluster is 8 articles but counts as 1 cluster unit
    # while the 2 distinct positive articles count as 2 units
    # → positive should win or it should be close to NEUTRAL
    # (exact outcome depends on weights; we assert it's not a blowout negative)
    assert ticker_result["sentiment"] in ("POSITIVE", "NEUTRAL"), (
        f"Cluster cap failed: got {ticker_result['sentiment']} with "
        f"score {ticker_result['conviction_score']}"
    )


def test_unique_event_count_accurate():
    """Two distinct events should produce unique_event_count=2."""
    results = [
        _result(sentiment="POSITIVE", headline="Tesla earnings beat"),
        _result(sentiment="NEGATIVE", headline="Tesla product recall major issue"),
    ]
    ticker_result = aggregate_to_ticker(results, ticker="TSLA")
    assert ticker_result["unique_event_count"] == 2


# =========================================================================== #
# Recency weighting                                                            #
# =========================================================================== #

def test_fresh_items_dominate_stale():
    """
    A single very fresh high-conviction POSITIVE item should outweigh
    many stale NEGATIVE items when recency decay is significant.
    """
    stale_negatives = [
        _result(sentiment="NEGATIVE", conviction_score=8.0,
                headline=f"Old bad news {i}", hours_ago=48.0)
        for i in range(3)
    ]
    fresh_positive = _result(
        sentiment="POSITIVE", conviction_score=9.5,
        headline="Fresh great earnings beat today", hours_ago=0.1
    )
    results = stale_negatives + [fresh_positive]
    ticker_result = aggregate_to_ticker(results, ticker="TSLA", decay_lambda=0.15)
    # At decay_lambda=0.15, 48h-old items have weight ~e^(-7.2) ≈ 0.07%
    # fresh item has weight ~1.0 → should dominate
    assert ticker_result["sentiment"] == "POSITIVE"


# =========================================================================== #
# Conviction score range                                                       #
# =========================================================================== #

def test_conviction_score_in_range():
    results = [_result(conviction_score=9.9) for _ in range(3)]
    ticker_result = aggregate_to_ticker(results, ticker="TSLA")
    assert 0.0 <= ticker_result["conviction_score"] <= 10.0


def test_neutral_conviction_capped():
    """Neutral signals should have reduced conviction."""
    results = [_result(sentiment="NEUTRAL", conviction_score=5.0)]
    ticker_result = aggregate_to_ticker(results, ticker="TSLA")
    assert ticker_result["sentiment"] == "NEUTRAL"
    # Conviction should be reduced by MIN_CONVICTION floor
    assert ticker_result["conviction_score"] <= 5.0


# =========================================================================== #
# Label helpers                                                                #
# =========================================================================== #

@pytest.mark.parametrize("label,expected", [
    ("POSITIVE", 1.0),
    ("NEGATIVE", -1.0),
    ("NEUTRAL", 0.0),
    ("garbage", 0.0),
])
def test_label_to_direction(label, expected):
    assert _label_to_direction(label) == expected


@pytest.mark.parametrize("direction,threshold,expected", [
    (0.5, 0.15, "POSITIVE"),
    (-0.5, 0.15, "NEGATIVE"),
    (0.05, 0.15, "NEUTRAL"),
    (-0.05, 0.15, "NEUTRAL"),
])
def test_direction_to_label(direction, threshold, expected):
    assert _direction_to_label(direction, threshold) == expected


# =========================================================================== #
# Output field types                                                           #
# =========================================================================== #

def test_ticker_result_types():
    results = [_result()]
    out = aggregate_to_ticker(results, ticker="TSLA")
    assert isinstance(out["ticker"], str)
    assert out["sentiment"] in VALID_SENTIMENT_LABELS
    assert isinstance(out["conviction_score"], float)
    assert isinstance(out["source_count"], int)
    assert isinstance(out["unique_event_count"], int)
    assert isinstance(out["generated_at"], datetime)
