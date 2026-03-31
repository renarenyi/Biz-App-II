"""
tests/nlp/test_schemas.py
--------------------------
Unit tests for nlp/schemas.py.

Tests cover:
- NewsItem validation
- ArticleSentimentResult validation
- TickerSentimentResult validation
- content_hash determinism and uniqueness
- clamp_conviction
- make_empty_ticker_result
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.nlp.schemas import (
    validate_news_item,
    validate_article_result,
    validate_ticker_result,
    compute_content_hash,
    clamp_conviction,
    make_empty_ticker_result,
    VALID_SENTIMENT_LABELS,
)


# =========================================================================== #
# NewsItem validation                                                          #
# =========================================================================== #

def test_valid_news_item_passes():
    item = {
        "ticker": "TSLA",
        "headline": "Tesla recalls vehicles",
        "published_at": datetime.now(tz=timezone.utc),
    }
    assert validate_news_item(item) is True


def test_news_item_missing_ticker_fails():
    item = {"headline": "Tesla recalls vehicles", "published_at": datetime.now(tz=timezone.utc)}
    assert validate_news_item(item) is False


def test_news_item_missing_headline_fails():
    item = {"ticker": "TSLA", "published_at": datetime.now(tz=timezone.utc)}
    assert validate_news_item(item) is False


def test_news_item_empty_headline_fails():
    item = {"ticker": "TSLA", "headline": "   ", "published_at": datetime.now(tz=timezone.utc)}
    assert validate_news_item(item) is False


def test_news_item_missing_published_at_fails():
    item = {"ticker": "TSLA", "headline": "Tesla news"}
    assert validate_news_item(item) is False


def test_news_item_raises_on_error():
    item = {"ticker": "TSLA"}
    with pytest.raises(ValueError):
        validate_news_item(item, raise_on_error=True)


# =========================================================================== #
# ArticleSentimentResult validation                                           #
# =========================================================================== #

def test_valid_article_result_passes():
    result = {
        "ticker": "TSLA",
        "headline": "Tesla recalls vehicles",
        "sentiment": "NEGATIVE",
        "conviction_score": 8.0,
    }
    assert validate_article_result(result) is True


@pytest.mark.parametrize("label", ["POSITIVE", "NEGATIVE", "NEUTRAL"])
def test_all_valid_sentiment_labels(label):
    result = {"ticker": "TSLA", "headline": "News", "sentiment": label, "conviction_score": 5.0}
    assert validate_article_result(result) is True


def test_invalid_sentiment_label_fails():
    result = {"ticker": "TSLA", "headline": "News", "sentiment": "BULLISH", "conviction_score": 5.0}
    assert validate_article_result(result) is False


def test_out_of_range_conviction_fails():
    result = {"ticker": "TSLA", "headline": "News", "sentiment": "POSITIVE", "conviction_score": 11.0}
    assert validate_article_result(result) is False


def test_negative_conviction_fails():
    result = {"ticker": "TSLA", "headline": "News", "sentiment": "POSITIVE", "conviction_score": -1.0}
    assert validate_article_result(result) is False


def test_missing_conviction_fails():
    result = {"ticker": "TSLA", "headline": "News", "sentiment": "POSITIVE"}
    assert validate_article_result(result) is False


# =========================================================================== #
# TickerSentimentResult validation                                            #
# =========================================================================== #

def test_valid_ticker_result_passes():
    result = {
        "ticker": "TSLA",
        "sentiment": "NEGATIVE",
        "conviction_score": 7.5,
        "generated_at": datetime.now(tz=timezone.utc),
    }
    assert validate_ticker_result(result) is True


def test_invalid_ticker_sentiment_fails():
    result = {
        "ticker": "TSLA",
        "sentiment": "STRONGLY_POSITIVE",
        "conviction_score": 7.5,
        "generated_at": datetime.now(tz=timezone.utc),
    }
    assert validate_ticker_result(result) is False


# =========================================================================== #
# Content hash                                                                 #
# =========================================================================== #

def test_content_hash_deterministic():
    item = {"ticker": "TSLA", "headline": "Tesla recall", "snippet": None,
            "published_at": "2026-03-10T12:00:00Z"}
    h1 = compute_content_hash(item)
    h2 = compute_content_hash(item)
    assert h1 == h2


def test_different_items_different_hashes():
    a = {"ticker": "TSLA", "headline": "Tesla recall", "snippet": None, "published_at": "2026-03-10"}
    b = {"ticker": "NVDA", "headline": "NVDA earnings beat", "snippet": None, "published_at": "2026-03-10"}
    assert compute_content_hash(a) != compute_content_hash(b)


def test_snippet_changes_hash():
    base = {"ticker": "TSLA", "headline": "Tesla news", "published_at": "2026-03-10"}
    a = {**base, "snippet": ""}
    b = {**base, "snippet": "Extra details here."}
    assert compute_content_hash(a) != compute_content_hash(b)


def test_content_hash_is_16_chars():
    item = {"ticker": "TSLA", "headline": "Test", "snippet": None, "published_at": "2026"}
    h = compute_content_hash(item)
    assert len(h) == 16


# =========================================================================== #
# clamp_conviction                                                             #
# =========================================================================== #

@pytest.mark.parametrize("score,expected", [
    (5.0, 5.0),
    (0.0, 0.0),
    (10.0, 10.0),
    (-1.0, 0.0),
    (11.0, 10.0),
    (10.001, 10.0),
])
def test_clamp_conviction(score, expected):
    assert clamp_conviction(score) == expected


# =========================================================================== #
# make_empty_ticker_result                                                     #
# =========================================================================== #

def test_make_empty_ticker_result():
    result = make_empty_ticker_result("TSLA", window_hours=24)
    assert result["ticker"] == "TSLA"
    assert result["sentiment"] == "NEUTRAL"
    assert result["conviction_score"] == 0.0
    assert result["source_count"] == 0
    assert result["analysis_window_hours"] == 24
    assert isinstance(result["generated_at"], datetime)
