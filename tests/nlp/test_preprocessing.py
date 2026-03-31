"""
tests/nlp/test_preprocessing.py
---------------------------------
Unit tests for nlp/preprocessing.py.

Tests cover:
- Exact duplicate removal (URL, article_id)
- Near-duplicate headline removal (Jaccard)
- Stale article filtering
- Empty input handling
- analysis_text construction and truncation
- Content hash stability
- Headline cleaning
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.nlp.preprocessing import (
    prepare_news_items,
    MAX_ANALYSIS_CHARS,
    JACCARD_NEAR_DUP_THRESHOLD,
)
from src.nlp.schemas import compute_content_hash


# =========================================================================== #
# Fixtures                                                                     #
# =========================================================================== #

def _now():
    return datetime.now(tz=timezone.utc)


def _article(
    headline: str = "TSLA recalls vehicles",
    ticker: str = "TSLA",
    url: str = "https://example.com/1",
    hours_ago: float = 1.0,
    snippet: str = "",
) -> dict:
    return {
        "ticker": ticker,
        "headline": headline,
        "summary": snippet or None,
        "url": url,
        "published_at": _now() - timedelta(hours=hours_ago),
        "provider": "test",
    }


# =========================================================================== #
# Empty input                                                                  #
# =========================================================================== #

def test_empty_articles_returns_empty():
    result = prepare_news_items([], ticker="TSLA")
    assert result == []


# =========================================================================== #
# Deduplication — exact URL                                                    #
# =========================================================================== #

def test_exact_url_dedup():
    articles = [
        _article(headline="TSLA recalls two million vehicles over software bug",
                 url="https://example.com/1"),
        _article(headline="TSLA recalls two million vehicles over software bug",
                 url="https://example.com/1"),   # exact duplicate URL — drop
        _article(headline="Tesla shares rally after strong quarterly earnings report",
                 url="https://example.com/2"),   # distinct URL AND distinct headline
    ]
    result = prepare_news_items(articles, ticker="TSLA")
    urls = [r["url"] for r in result]
    assert len(result) == 2, f"Expected 2 after URL-dedup, got {len(result)}: {urls}"
    assert urls.count("https://example.com/1") == 1


# =========================================================================== #
# Deduplication — near-duplicate headlines                                     #
# =========================================================================== #

def test_near_duplicate_headline_dedup():
    # Identical headlines (Jaccard = 1.0 > threshold 0.85) → one survives
    shared_headline = "Tesla recalls vehicles autopilot concerns safety regulators watchdog"
    articles = [
        _article(headline=shared_headline, url="https://example.com/1"),
        _article(headline=shared_headline, url="https://example.com/2"),
        _article(headline="NVDA beats quarterly earnings expectations", url="https://example.com/3"),
    ]
    result = prepare_news_items(articles, ticker="TSLA")
    assert len(result) == 2
    headlines = [r["headline"] for r in result]
    assert any("Tesla recalls" in h for h in headlines)
    assert any("NVDA" in h for h in headlines)


def test_distinct_headlines_all_kept():
    articles = [
        _article(headline="Tesla recalls vehicles", url="https://a.com/1"),
        _article(headline="NVDA earnings beat expectations", url="https://a.com/2"),
        _article(headline="Apple launches new product line", url="https://a.com/3"),
    ]
    result = prepare_news_items(articles, ticker="TSLA")
    assert len(result) == 3


# =========================================================================== #
# Stale article filtering                                                      #
# =========================================================================== #

def test_stale_articles_filtered():
    articles = [
        _article(hours_ago=1, url="https://a.com/1"),   # fresh
        _article(hours_ago=25, url="https://a.com/2"),  # stale (beyond 24h window)
    ]
    result = prepare_news_items(articles, ticker="TSLA", window_hours=24)
    assert len(result) == 1
    assert "1" in result[0]["url"]


def test_article_at_window_boundary_kept():
    articles = [
        _article(hours_ago=23.9, url="https://a.com/1"),
    ]
    result = prepare_news_items(articles, ticker="TSLA", window_hours=24)
    assert len(result) == 1


def test_no_published_at_retained():
    """Articles with missing timestamps are kept (can't be filtered)."""
    articles = [{
        "ticker": "TSLA",
        "headline": "Tesla news without timestamp",
        "url": "https://a.com/no-ts",
        "published_at": None,
        "provider": "test",
    }]
    result = prepare_news_items(articles, ticker="TSLA", window_hours=24)
    assert len(result) == 1


# =========================================================================== #
# analysis_text and content hash                                               #
# =========================================================================== #

def test_analysis_text_combines_headline_and_snippet():
    articles = [_article(
        headline="Tesla beats earnings",
        snippet="Revenue up 20% year over year.",
        url="https://a.com/1",
    )]
    result = prepare_news_items(articles, ticker="TSLA")
    assert len(result) == 1
    at = result[0]["analysis_text"]
    assert "Tesla beats earnings" in at
    assert "Revenue" in at


def test_analysis_text_truncated():
    long_snippet = "x" * (MAX_ANALYSIS_CHARS + 200)
    articles = [_article(
        headline="Test headline",
        snippet=long_snippet,
        url="https://a.com/1",
    )]
    result = prepare_news_items(articles, ticker="TSLA")
    assert len(result[0]["analysis_text"]) <= MAX_ANALYSIS_CHARS


def test_content_hash_is_deterministic():
    article = _article(headline="Tesla recall")
    items = prepare_news_items([article], ticker="TSLA")
    items2 = prepare_news_items([article], ticker="TSLA")
    assert items[0]["content_hash"] == items2[0]["content_hash"]


def test_different_articles_different_hashes():
    a1 = _article(headline="Tesla recall", url="https://a.com/1")
    a2 = _article(headline="NVDA earnings", url="https://a.com/2")
    items = prepare_news_items([a1, a2], ticker="TSLA")
    assert items[0]["content_hash"] != items[1]["content_hash"]


# =========================================================================== #
# Headline cleaning                                                            #
# =========================================================================== #

def test_html_tags_stripped():
    articles = [_article(headline="<b>Tesla</b> recalls vehicles", url="https://a.com/1")]
    result = prepare_news_items(articles, ticker="TSLA")
    assert "<b>" not in result[0]["headline"]
    assert "Tesla" in result[0]["headline"]


def test_whitespace_collapsed():
    articles = [_article(headline="Tesla   recalls     vehicles", url="https://a.com/1")]
    result = prepare_news_items(articles, ticker="TSLA")
    assert "  " not in result[0]["headline"]


# =========================================================================== #
# Sorted newest-first                                                          #
# =========================================================================== #

def test_results_sorted_newest_first():
    articles = [
        _article(headline="Old news", hours_ago=5, url="https://a.com/old"),
        _article(headline="New news", hours_ago=1, url="https://a.com/new"),
    ]
    result = prepare_news_items(articles, ticker="TSLA")
    assert result[0]["headline"] == "New news"
    assert result[1]["headline"] == "Old news"


# =========================================================================== #
# Ticker enforcement                                                           #
# =========================================================================== #

def test_ticker_overridden():
    articles = [_article(ticker="WRONG")]
    result = prepare_news_items(articles, ticker="TSLA")
    assert result[0]["ticker"] == "TSLA"
