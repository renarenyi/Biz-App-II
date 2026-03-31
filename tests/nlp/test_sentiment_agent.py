"""
tests/nlp/test_sentiment_agent.py
-----------------------------------
Integration-style unit tests for the SentimentAgent.

These tests use a StubProvider (no real model calls) to verify:
- End-to-end pipeline flow
- Cache integration (no double inference)
- FallbackRouter behaviour
- Empty article handling
- Batch analyze_batch interface

No model downloads required.  All model calls are stubbed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest

from src.nlp.providers.base_provider import (
    BaseSentimentProvider,
    ProviderUnavailableError,
)
from src.nlp.fallback_router import FallbackRouter
from src.nlp.cache import SentimentCache
from src.nlp.sentiment_agent import SentimentAgent
from src.nlp.schemas import VALID_SENTIMENT_LABELS


# =========================================================================== #
# Stub providers                                                               #
# =========================================================================== #

class _StubProvider(BaseSentimentProvider):
    """
    Always-available provider that returns a fixed sentiment label.
    Records how many articles were classified to detect double-inference.
    """

    def __init__(self, name: str = "stub", label: str = "POSITIVE", score: float = 8.0):
        self._name = name
        self._label = label
        self._score = score
        self.call_count = 0
        self.classified_count = 0

    @property
    def provider_name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return True

    def classify_articles(self, news_items: list[dict]) -> list[dict]:
        self.call_count += 1
        self.classified_count += len(news_items)
        return [
            self._build_result(item, self._label, self._score, "Stub reasoning")
            for item in news_items
        ]


class _UnavailableProvider(BaseSentimentProvider):
    @property
    def provider_name(self) -> str:
        return "unavailable"

    def is_available(self) -> bool:
        return False

    def classify_articles(self, news_items: list[dict]) -> list[dict]:
        raise ProviderUnavailableError("This provider is always unavailable.")


# =========================================================================== #
# Helpers                                                                      #
# =========================================================================== #

def _now():
    return datetime.now(tz=timezone.utc)


def _article(
    headline: str = "TSLA earnings beat",
    url: str = "https://a.com/1",
    hours_ago: float = 1.0,
    ticker: str = "TSLA",
) -> dict:
    return {
        "ticker": ticker,
        "headline": headline,
        "summary": "Some relevant snippet.",
        "url": url,
        "published_at": _now() - timedelta(hours=hours_ago),
        "provider": "test",
    }


def _agent(label: str = "POSITIVE", score: float = 8.0) -> SentimentAgent:
    stub = _StubProvider(label=label, score=score)
    return SentimentAgent(
        router=FallbackRouter([stub]),
        cache=SentimentCache(use_disk=False),
        window_hours=24,
    )


# =========================================================================== #
# Basic pipeline                                                               #
# =========================================================================== #

def test_analyze_returns_ticker_sentiment_result():
    agent = _agent()
    articles = [_article(), _article(headline="Different news", url="https://a.com/2")]
    result = agent.analyze("TSLA", articles)

    assert result["ticker"] == "TSLA"
    assert result["sentiment"] in VALID_SENTIMENT_LABELS
    assert 0.0 <= result["conviction_score"] <= 10.0
    assert isinstance(result["generated_at"], datetime)
    assert result["source_count"] >= 1


def test_empty_articles_returns_neutral():
    agent = _agent()
    result = agent.analyze("TSLA", [])
    assert result["sentiment"] == "NEUTRAL"
    assert result["conviction_score"] == 0.0
    assert result["source_count"] == 0


def test_all_positive_articles_produces_positive():
    agent = _agent(label="POSITIVE", score=9.0)
    articles = [_article(headline=f"Good news {i}", url=f"https://a.com/{i}") for i in range(4)]
    result = agent.analyze("TSLA", articles)
    assert result["sentiment"] == "POSITIVE"


def test_all_negative_articles_produces_negative():
    agent = _agent(label="NEGATIVE", score=8.0)
    articles = [_article(headline=f"Bad news {i}", url=f"https://a.com/{i}") for i in range(4)]
    result = agent.analyze("TSLA", articles)
    assert result["sentiment"] == "NEGATIVE"


# =========================================================================== #
# Cache — no double inference                                                  #
# =========================================================================== #

def test_cache_prevents_double_inference():
    stub = _StubProvider()
    cache = SentimentCache(use_disk=False)
    agent = SentimentAgent(
        router=FallbackRouter([stub]),
        cache=cache,
    )
    articles = [_article(url="https://a.com/cached")]

    # First call — should trigger inference
    agent.analyze("TSLA", articles)
    first_count = stub.classified_count

    # Second call with same articles — should use cache
    agent.analyze("TSLA", articles)
    second_count = stub.classified_count

    assert second_count == first_count, (
        f"Cache miss: stub classified {second_count - first_count} extra articles "
        "on the second call instead of using the cache."
    )


def test_new_article_triggers_inference_after_cache_hit():
    stub = _StubProvider()
    cache = SentimentCache(use_disk=False)
    agent = SentimentAgent(router=FallbackRouter([stub]), cache=cache)

    articles_v1 = [_article(url="https://a.com/1")]
    agent.analyze("TSLA", articles_v1)
    count_after_first = stub.classified_count

    # Add a fresh article with a new URL
    articles_v2 = articles_v1 + [_article(headline="Brand new article", url="https://a.com/2")]
    agent.analyze("TSLA", articles_v2)
    count_after_second = stub.classified_count

    # Only the new article should be classified
    assert count_after_second == count_after_first + 1


# =========================================================================== #
# Fallback router                                                              #
# =========================================================================== #

def test_fallback_to_second_provider():
    unavailable = _UnavailableProvider()
    stub = _StubProvider(name="backup", label="POSITIVE")
    agent = SentimentAgent(
        router=FallbackRouter([unavailable, stub]),
        cache=SentimentCache(use_disk=False),
    )
    articles = [_article()]
    result = agent.analyze("TSLA", articles)
    assert result["sentiment"] == "POSITIVE"
    assert result.get("provider_used") == "backup"


def test_all_providers_exhausted_returns_neutral():
    """When all providers fail, agent should return neutral signal, not raise."""
    agent = SentimentAgent(
        router=FallbackRouter([_UnavailableProvider()]),
        cache=SentimentCache(use_disk=False),
    )
    articles = [_article()]
    result = agent.analyze("TSLA", articles)
    # Should degrade gracefully
    assert result["sentiment"] == "NEUTRAL"


# =========================================================================== #
# Stale articles                                                               #
# =========================================================================== #

def test_all_stale_articles_returns_neutral():
    agent = _agent()
    stale_articles = [
        _article(hours_ago=30, url=f"https://a.com/{i}") for i in range(3)
    ]
    result = agent.analyze("TSLA", stale_articles, window_hours=24)
    # All articles fall outside 24h window → preprocessing returns 0 items
    assert result["sentiment"] == "NEUTRAL"
    assert result["source_count"] == 0


# =========================================================================== #
# Batch analysis                                                               #
# =========================================================================== #

def test_analyze_batch():
    agent = _agent()
    ticker_articles = {
        "TSLA": [_article(ticker="TSLA")],
        "NVDA": [_article(headline="NVDA earnings", ticker="NVDA", url="https://b.com/1")],
    }
    results = agent.analyze_batch(ticker_articles)
    assert set(results.keys()) == {"TSLA", "NVDA"}
    for ticker, result in results.items():
        assert result["ticker"] == ticker
        assert result["sentiment"] in VALID_SENTIMENT_LABELS


# =========================================================================== #
# Cache stats                                                                  #
# =========================================================================== #

def test_cache_stats_returns_dict():
    agent = _agent()
    agent.analyze("TSLA", [_article()])
    stats = agent.cache_stats()
    assert "memory_entries" in stats
    assert stats["memory_entries"] >= 1
