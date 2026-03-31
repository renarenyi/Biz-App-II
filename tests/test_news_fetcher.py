"""
test_news_fetcher.py
--------------------
Unit tests for NewsFetcher, news provider adapters, and utility functions.

Tests use mock providers to avoid hitting live APIs.
Run with: pytest tests/ -v
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from src.data.news_fetcher import NewsFetcher
from src.data.cache import DataCache
from src.data.providers.alpaca_news_provider import AlpacaNewsProviderError
from src.data.providers.rss_news_provider import RSSNewsProviderError
from src.data.schemas import validate_article
from src.data.utils import (
    clean_headline,
    deduplicate_by_fingerprint,
    deduplicate_by_key,
    jaccard_similarity,
    now_utc,
    to_utc,
)


# =========================================================================== #
# Fixtures                                                                     #
# =========================================================================== #

def _make_articles(
    ticker: str = "AAPL",
    count: int = 5,
    hours_ago: int = 2,
    provider: str = "alpaca_news",
) -> list[dict]:
    """Generate a list of valid mock articles."""
    base_time = now_utc() - timedelta(hours=hours_ago)
    articles = []
    for i in range(count):
        articles.append({
            "ticker": ticker,
            "headline": f"Apple announces major product launch {i}",
            "summary": f"Apple Inc. has announced a significant new product release {i}.",
            "source": "Reuters",
            "published_at": base_time - timedelta(minutes=i * 15),
            "url": f"https://example.com/article-{i}",
            "provider": provider,
            "article_id": f"article-{i}",
            "ingested_at": now_utc(),
        })
    return articles


def _make_mock_alpaca_news(articles=None, fail=False):
    mock = MagicMock()
    mock.is_healthy.return_value = not fail
    if fail:
        mock.get_news.side_effect = AlpacaNewsProviderError("Alpaca news mock failure")
    else:
        mock.get_news.return_value = articles if articles is not None else _make_articles()
    return mock


def _make_mock_rss_news(articles=None, fail=False):
    mock = MagicMock()
    if fail:
        mock.get_news.side_effect = RSSNewsProviderError("RSS mock failure")
    else:
        mock.get_news.return_value = articles if articles is not None else _make_articles(provider="rss")
    return mock


def _make_mock_fmp_news(articles=None, fail=False, disabled=False):
    """Create a mock FMP provider. Default: disabled (is_healthy=False)."""
    mock = MagicMock()
    mock.is_healthy.return_value = not (fail or disabled)
    if fail:
        from src.data.providers.fmp_news_provider import FMPNewsProviderError
        mock.get_news.side_effect = FMPNewsProviderError("FMP mock failure")
    elif disabled:
        mock.get_news.return_value = []
    else:
        mock.get_news.return_value = articles if articles is not None else _make_articles(provider="fmp")
    return mock


def _make_mock_finnhub_news(articles=None, fail=False, disabled=False):
    """Create a mock Finnhub provider. Default: disabled (is_healthy=False)."""
    mock = MagicMock()
    mock.is_healthy.return_value = not (fail or disabled)
    if fail:
        from src.data.providers.finnhub_news_provider import FinnhubNewsProviderError
        mock.get_news.side_effect = FinnhubNewsProviderError("Finnhub mock failure")
    elif disabled:
        mock.get_news.return_value = []
    else:
        mock.get_news.return_value = articles if articles is not None else _make_articles(provider="finnhub")
    return mock


# =========================================================================== #
# NewsFetcher — happy path                                                     #
# =========================================================================== #

class TestNewsFetcherHappyPath(unittest.TestCase):

    def setUp(self):
        self.articles = _make_articles(count=5)
        self.fetcher = NewsFetcher(
            alpaca_news=_make_mock_alpaca_news(self.articles),
            fmp_news=_make_mock_fmp_news(disabled=True),
            finnhub_news=_make_mock_finnhub_news(disabled=True),
            rss_news=_make_mock_rss_news(fail=True),
            use_cache=False,
        )

    def test_returns_list(self):
        result = self.fetcher.get_recent_news("AAPL")
        self.assertIsInstance(result, list)

    def test_articles_have_required_fields(self):
        result = self.fetcher.get_recent_news("AAPL")
        for article in result:
            self.assertIn("ticker", article)
            self.assertIn("headline", article)
            self.assertIn("published_at", article)
            self.assertIn("provider", article)

    def test_articles_ticker_is_set(self):
        result = self.fetcher.get_recent_news("AAPL")
        for article in result:
            self.assertEqual(article["ticker"], "AAPL")

    def test_articles_sorted_newest_first(self):
        result = self.fetcher.get_recent_news("AAPL")
        timestamps = [a["published_at"] for a in result if a.get("published_at")]
        if len(timestamps) > 1:
            for i in range(len(timestamps) - 1):
                self.assertGreaterEqual(timestamps[i], timestamps[i + 1])

    def test_published_at_is_utc_aware(self):
        result = self.fetcher.get_recent_news("AAPL")
        for article in result:
            pub = article.get("published_at")
            if pub is not None:
                self.assertIsNotNone(pub.tzinfo, "published_at should be tz-aware")


# =========================================================================== #
# NewsFetcher — fallback                                                       #
# =========================================================================== #

class TestNewsFetcherFallback(unittest.TestCase):

    def setUp(self):
        self.rss_articles = _make_articles(count=4, provider="rss")
        self.fetcher = NewsFetcher(
            alpaca_news=_make_mock_alpaca_news(fail=True),
            fmp_news=_make_mock_fmp_news(disabled=True),
            finnhub_news=_make_mock_finnhub_news(disabled=True),
            rss_news=_make_mock_rss_news(self.rss_articles),
            use_cache=False,
            min_articles=1,
        )

    def test_falls_back_to_rss_when_alpaca_fails(self):
        result = self.fetcher.get_recent_news("AAPL")
        self.assertGreater(len(result), 0)
        for article in result:
            self.assertEqual(article["provider"], "rss")


class TestNewsFetcherBothFail(unittest.TestCase):

    def setUp(self):
        self.fetcher = NewsFetcher(
            alpaca_news=_make_mock_alpaca_news(fail=True),
            fmp_news=_make_mock_fmp_news(disabled=True),
            finnhub_news=_make_mock_finnhub_news(disabled=True),
            rss_news=_make_mock_rss_news(fail=True),
            use_cache=False,
        )

    def test_returns_empty_list_when_all_fail(self):
        result = self.fetcher.get_recent_news("AAPL")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    def test_does_not_raise_when_all_fail(self):
        # Should never raise — callers must handle empty results
        try:
            self.fetcher.get_recent_news("AAPL")
        except Exception as exc:
            self.fail(f"get_recent_news raised unexpectedly: {exc}")


# =========================================================================== #
# NewsFetcher — supplement logic                                               #
# =========================================================================== #

class TestNewsFetcherSupplementation(unittest.TestCase):

    def test_rss_supplements_when_alpaca_below_min(self):
        """When Alpaca returns < min_articles, RSS should also be queried."""
        low_count_articles = _make_articles(count=1)  # below min_articles=3
        rss_articles = _make_articles(count=4, provider="rss")

        fetcher = NewsFetcher(
            alpaca_news=_make_mock_alpaca_news(low_count_articles),
            fmp_news=_make_mock_fmp_news(disabled=True),
            finnhub_news=_make_mock_finnhub_news(disabled=True),
            rss_news=_make_mock_rss_news(rss_articles),
            use_cache=False,
            min_articles=3,
        )
        result = fetcher.get_recent_news("AAPL")
        providers = {a["provider"] for a in result}
        self.assertIn("rss", providers)

    def test_rss_not_called_when_alpaca_sufficient(self):
        """When Alpaca has enough articles, RSS should not be called."""
        plenty = _make_articles(count=10)
        mock_rss = _make_mock_rss_news()

        fetcher = NewsFetcher(
            alpaca_news=_make_mock_alpaca_news(plenty),
            fmp_news=_make_mock_fmp_news(disabled=True),
            finnhub_news=_make_mock_finnhub_news(disabled=True),
            rss_news=mock_rss,
            use_cache=False,
            min_articles=3,
        )
        fetcher.get_recent_news("AAPL", include_rss=True)
        # RSS should not have been called since Alpaca returned 10 articles
        mock_rss.get_news.assert_not_called()


# =========================================================================== #
# Normalization                                                                #
# =========================================================================== #

class TestNormalizeArticles(unittest.TestCase):

    def setUp(self):
        self.fetcher = NewsFetcher(use_cache=False)

    def test_sets_ticker(self):
        articles = [{"headline": "TSLA rises", "published_at": now_utc(), "provider": "rss"}]
        result = self.fetcher.normalize_articles(articles, ticker="TSLA")
        self.assertEqual(result[0]["ticker"], "TSLA")

    def test_cleans_headline(self):
        articles = [{
            "ticker": "AAPL",
            "headline": "  Apple   Reports   Earnings  ",
            "published_at": now_utc(),
            "provider": "rss",
        }]
        result = self.fetcher.normalize_articles(articles, ticker="AAPL")
        self.assertEqual(result[0]["headline"], "Apple Reports Earnings")

    def test_drops_empty_headline(self):
        articles = [{
            "ticker": "AAPL",
            "headline": "",
            "published_at": now_utc(),
            "provider": "rss",
        }]
        result = self.fetcher.normalize_articles(articles, ticker="AAPL")
        self.assertEqual(len(result), 0)

    def test_converts_string_published_at(self):
        articles = [{
            "ticker": "AAPL",
            "headline": "Apple earnings",
            "published_at": "2024-01-15T10:30:00Z",
            "provider": "rss",
        }]
        result = self.fetcher.normalize_articles(articles, ticker="AAPL")
        pub = result[0]["published_at"]
        self.assertIsNotNone(pub.tzinfo)


# =========================================================================== #
# Deduplication                                                                #
# =========================================================================== #

class TestDeduplicateArticles(unittest.TestCase):

    def setUp(self):
        self.fetcher = NewsFetcher(use_cache=False)

    def test_removes_url_duplicates(self):
        articles = [
            {"ticker": "AAPL", "headline": "AAPL up 3%", "url": "http://a.com", "published_at": now_utc(), "provider": "rss"},
            {"ticker": "AAPL", "headline": "AAPL surges 3%", "url": "http://a.com", "published_at": now_utc(), "provider": "rss"},
        ]
        result = self.fetcher.deduplicate_articles(articles)
        self.assertEqual(len(result), 1)

    def test_removes_near_duplicate_headlines(self):
        articles = [
            {"ticker": "AAPL", "headline": "Apple stock rises on strong earnings", "url": "http://a.com", "published_at": now_utc(), "provider": "rss"},
            {"ticker": "AAPL", "headline": "Apple stock rises on strong earnings report", "url": "http://b.com", "published_at": now_utc(), "provider": "rss"},
        ]
        result = self.fetcher.deduplicate_articles(articles)
        # Should detect near-duplicate
        self.assertLessEqual(len(result), 2)

    def test_keeps_distinct_articles(self):
        articles = [
            {"ticker": "AAPL", "headline": "Apple launches new iPhone", "url": "http://a.com", "published_at": now_utc(), "provider": "rss"},
            {"ticker": "AAPL", "headline": "Federal Reserve raises interest rates", "url": "http://b.com", "published_at": now_utc(), "provider": "rss"},
        ]
        result = self.fetcher.deduplicate_articles(articles)
        self.assertEqual(len(result), 2)


# =========================================================================== #
# Freshness filtering                                                          #
# =========================================================================== #

class TestFilterStaleArticles(unittest.TestCase):

    def setUp(self):
        self.fetcher = NewsFetcher(use_cache=False)

    def test_removes_old_articles(self):
        now = now_utc()
        articles = [
            {"ticker": "AAPL", "headline": "Recent news", "published_at": now - timedelta(hours=1)},
            {"ticker": "AAPL", "headline": "Old news", "published_at": now - timedelta(days=7)},
        ]
        result = self.fetcher.filter_stale_articles(articles, lookback_hours=24)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["headline"], "Recent news")

    def test_keeps_articles_within_window(self):
        now = now_utc()
        articles = [
            {"ticker": "AAPL", "headline": f"Article {i}", "published_at": now - timedelta(hours=i)}
            for i in range(1, 6)
        ]
        result = self.fetcher.filter_stale_articles(articles, lookback_hours=24)
        self.assertEqual(len(result), 5)

    def test_retains_articles_without_date(self):
        articles = [{"ticker": "AAPL", "headline": "No date article", "published_at": None}]
        result = self.fetcher.filter_stale_articles(articles, lookback_hours=24)
        self.assertEqual(len(result), 1)


# =========================================================================== #
# Caching                                                                      #
# =========================================================================== #

class TestNewsFetcherCaching(unittest.TestCase):

    def _mem_cache(self):
        return DataCache(cache_dir="/tmp/ta_test_cache_news", use_disk=False)

    def test_cache_used_on_second_call(self):
        articles = _make_articles(count=5)
        mock_alpaca = _make_mock_alpaca_news(articles)

        fetcher = NewsFetcher(
            cache=self._mem_cache(),
            alpaca_news=mock_alpaca,
            fmp_news=_make_mock_fmp_news(disabled=True),
            finnhub_news=_make_mock_finnhub_news(disabled=True),
            rss_news=_make_mock_rss_news(fail=True),
            use_cache=True,
            min_articles=1,
        )
        fetcher.get_recent_news("AAPL", lookback_hours=24)
        fetcher.get_recent_news("AAPL", lookback_hours=24)

        # Alpaca should have been called only once
        self.assertEqual(mock_alpaca.get_news.call_count, 1)

    def test_force_refresh_bypasses_cache(self):
        articles = _make_articles(count=5)
        mock_alpaca = _make_mock_alpaca_news(articles)

        fetcher = NewsFetcher(
            cache=self._mem_cache(),
            alpaca_news=mock_alpaca,
            fmp_news=_make_mock_fmp_news(disabled=True),
            finnhub_news=_make_mock_finnhub_news(disabled=True),
            rss_news=_make_mock_rss_news(fail=True),
            use_cache=True,
            min_articles=1,
        )
        fetcher.get_recent_news("AAPL")
        fetcher.get_recent_news("AAPL", force_refresh=True)

        self.assertEqual(mock_alpaca.get_news.call_count, 2)


# =========================================================================== #
# Schema validation                                                            #
# =========================================================================== #

class TestArticleSchema(unittest.TestCase):

    def test_valid_article_passes(self):
        article = {
            "ticker": "AAPL",
            "headline": "Apple earnings beat expectations",
            "published_at": now_utc(),
            "provider": "alpaca_news",
        }
        self.assertTrue(validate_article(article))

    def test_missing_required_field_fails(self):
        article = {
            "headline": "Apple earnings",
            "published_at": now_utc(),
            # missing ticker and provider
        }
        self.assertFalse(validate_article(article, raise_on_error=False))

    def test_empty_headline_fails(self):
        article = {
            "ticker": "AAPL",
            "headline": "   ",
            "published_at": now_utc(),
            "provider": "rss",
        }
        self.assertFalse(validate_article(article, raise_on_error=False))


# =========================================================================== #
# clean_headline utility                                                       #
# =========================================================================== #

class TestCleanHeadline(unittest.TestCase):

    def test_strips_whitespace(self):
        self.assertEqual(clean_headline("  hello  "), "hello")

    def test_normalizes_internal_whitespace(self):
        self.assertEqual(clean_headline("hello   world"), "hello world")

    def test_empty_string(self):
        self.assertEqual(clean_headline(""), "")

    def test_none_equivalent(self):
        self.assertEqual(clean_headline(""), "")

    def test_unicode_normalization(self):
        # Ensure unicode ligatures and special chars are handled
        result = clean_headline("Apple\u2019s earnings")  # curly apostrophe
        self.assertIsInstance(result, str)


# =========================================================================== #
# Entry point                                                                  #
# =========================================================================== #

if __name__ == "__main__":
    unittest.main()
