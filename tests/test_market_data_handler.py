"""
test_market_data_handler.py
---------------------------
Unit tests for MarketDataHandler and its supporting infrastructure.

Tests use mock providers to avoid hitting live APIs.
Run with: pytest tests/ -v
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.market_data_handler import MarketDataHandler
from src.data.providers.alpaca_provider import AlpacaProviderError
from src.data.providers.yfinance_provider import YFinanceProviderError
from src.data.cache import DataCache, TTLCache
from src.data.schemas import validate_ohlcv_df, normalize_ohlcv_columns
from src.data.utils import (
    to_utc, clean_headline, headline_fingerprint, jaccard_similarity,
    deduplicate_by_key, sort_by_timestamp, drop_ohlcv_duplicates,
)


# =========================================================================== #
# Fixtures / helpers                                                           #
# =========================================================================== #

def _make_ohlcv_df(ticker: str = "AAPL", n_rows: int = 5, source: str = "alpaca") -> pd.DataFrame:
    """Create a minimal valid OHLCV DataFrame for testing."""
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="D", tz="UTC")
    return pd.DataFrame({
        "timestamp": dates,
        "open":   [150.0 + i for i in range(n_rows)],
        "high":   [155.0 + i for i in range(n_rows)],
        "low":    [148.0 + i for i in range(n_rows)],
        "close":  [152.0 + i for i in range(n_rows)],
        "volume": [1_000_000 + i * 10_000 for i in range(n_rows)],
        "symbol": ticker,
        "source": source,
        "timeframe": "1Day",
        "is_fallback": source != "alpaca",
    })


def _make_mock_alpaca(df: pd.DataFrame = None, fail: bool = False) -> MagicMock:
    mock = MagicMock()
    mock.is_healthy.return_value = not fail
    if fail:
        mock.get_bars.side_effect = AlpacaProviderError("Alpaca mock failure")
        mock.get_latest_quote.side_effect = AlpacaProviderError("Alpaca mock failure")
    else:
        mock.get_bars.return_value = df if df is not None else _make_ohlcv_df()
        mock.get_latest_quote.return_value = {
            "ticker": "AAPL",
            "last_price": 152.5,
            "ask_price": 153.0,
            "bid_price": 152.0,
            "timestamp": datetime(2024, 1, 5, 16, 0, tzinfo=timezone.utc),
            "source": "alpaca",
        }
    return mock


def _make_mock_yfinance(df: pd.DataFrame = None, fail: bool = False) -> MagicMock:
    mock = MagicMock()
    if fail:
        mock.get_bars.side_effect = YFinanceProviderError("yfinance mock failure")
        mock.get_latest_price.side_effect = YFinanceProviderError("yfinance mock failure")
    else:
        mock.get_bars.return_value = df if df is not None else _make_ohlcv_df(source="yfinance")
        mock.get_latest_price.return_value = {
            "ticker": "AAPL",
            "last_price": 152.0,
            "ask_price": None,
            "bid_price": None,
            "timestamp": datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc),
            "source": "yfinance",
        }
    return mock


# =========================================================================== #
# MarketDataHandler tests                                                      #
# =========================================================================== #

class TestMarketDataHandlerHappyPath(unittest.TestCase):
    """Tests where Alpaca is healthy and returns valid data."""

    def setUp(self):
        self.expected_df = _make_ohlcv_df(source="alpaca")
        self.handler = MarketDataHandler(
            alpaca=_make_mock_alpaca(self.expected_df),
            yfinance=_make_mock_yfinance(fail=True),
            use_cache=False,
        )

    def test_get_historical_bars_returns_dataframe(self):
        df = self.handler.get_historical_bars("AAPL", "2024-01-01", "2024-01-05")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)

    def test_get_historical_bars_has_required_columns(self):
        df = self.handler.get_historical_bars("AAPL", "2024-01-01", "2024-01-05")
        required = {"timestamp", "open", "high", "low", "close", "volume", "symbol", "source"}
        self.assertTrue(required.issubset(set(df.columns)))

    def test_get_historical_bars_source_is_alpaca(self):
        df = self.handler.get_historical_bars("AAPL", "2024-01-01", "2024-01-05")
        self.assertTrue((df["source"] == "alpaca").all())

    def test_get_latest_price_returns_dict(self):
        quote = self.handler.get_latest_price("AAPL")
        self.assertIsInstance(quote, dict)
        self.assertIn("last_price", quote)
        self.assertIn("ticker", quote)

    def test_get_latest_price_source_is_alpaca(self):
        quote = self.handler.get_latest_price("AAPL")
        self.assertEqual(quote["source"], "alpaca")

    def test_timestamps_are_tz_aware(self):
        df = self.handler.get_historical_bars("AAPL", "2024-01-01", "2024-01-05")
        self.assertIsNotNone(df["timestamp"].dt.tz)

    def test_rows_sorted_ascending(self):
        df = self.handler.get_historical_bars("AAPL", "2024-01-01", "2024-01-05")
        self.assertTrue(df["timestamp"].is_monotonic_increasing)


class TestMarketDataHandlerFallback(unittest.TestCase):
    """Tests where Alpaca fails and yfinance is used as fallback."""

    def setUp(self):
        self.yf_df = _make_ohlcv_df(source="yfinance")
        self.handler = MarketDataHandler(
            alpaca=_make_mock_alpaca(fail=True),
            yfinance=_make_mock_yfinance(self.yf_df),
            use_cache=False,
        )

    def test_falls_back_to_yfinance_when_alpaca_fails(self):
        df = self.handler.get_historical_bars("AAPL", "2024-01-01", "2024-01-05")
        self.assertFalse(df.empty)
        self.assertTrue((df["source"] == "yfinance").all())

    def test_fallback_bars_are_flagged(self):
        df = self.handler.get_historical_bars("AAPL", "2024-01-01", "2024-01-05")
        if "is_fallback" in df.columns:
            self.assertTrue(df["is_fallback"].any())

    def test_quote_falls_back_to_yfinance(self):
        quote = self.handler.get_latest_price("AAPL")
        self.assertEqual(quote["source"], "yfinance")


class TestMarketDataHandlerAllFail(unittest.TestCase):
    """Tests where all providers fail."""

    def setUp(self):
        self.handler = MarketDataHandler(
            alpaca=_make_mock_alpaca(fail=True),
            yfinance=_make_mock_yfinance(fail=True),
            use_cache=False,
        )

    def test_returns_empty_df_when_all_providers_fail(self):
        df = self.handler.get_historical_bars("AAPL", "2024-01-01", "2024-01-05")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

    def test_returns_none_price_when_all_fail(self):
        quote = self.handler.get_latest_price("AAPL")
        self.assertIsNone(quote["last_price"])
        self.assertEqual(quote["source"], "none")


class TestMarketDataHandlerClean(unittest.TestCase):
    """Tests for clean_ohlcv and validate_bars."""

    def setUp(self):
        self.handler = MarketDataHandler(
            alpaca=_make_mock_alpaca(),
            use_cache=False,
        )

    def test_clean_ohlcv_removes_all_nan_rows(self):
        df = _make_ohlcv_df(n_rows=5)
        df.loc[2, ["open", "high", "low", "close"]] = float("nan")
        cleaned = self.handler.clean_ohlcv(df)
        self.assertEqual(len(cleaned), 4)

    def test_clean_ohlcv_deduplicates(self):
        df = _make_ohlcv_df(n_rows=3)
        df_dup = pd.concat([df, df]).reset_index(drop=True)
        cleaned = self.handler.clean_ohlcv(df_dup)
        self.assertEqual(len(cleaned), 3)

    def test_clean_ohlcv_sorts_ascending(self):
        df = _make_ohlcv_df(n_rows=5)
        shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        cleaned = self.handler.clean_ohlcv(shuffled)
        self.assertTrue(cleaned["timestamp"].is_monotonic_increasing)

    def test_validate_bars_raises_on_empty(self):
        with self.assertRaises(ValueError):
            self.handler.validate_bars(pd.DataFrame())

    def test_validate_bars_raises_on_missing_columns(self):
        df = pd.DataFrame({"timestamp": [], "close": []})
        with self.assertRaises(ValueError):
            self.handler.validate_bars(df)

    def test_validate_bars_passes_on_valid_df(self):
        df = _make_ohlcv_df()
        # Should not raise
        self.handler.validate_bars(df)

    def test_add_moving_averages(self):
        df = _make_ohlcv_df(n_rows=60)
        result = self.handler.add_moving_averages(df, windows=[20, 50])
        self.assertIn("sma_20", result.columns)
        self.assertIn("sma_50", result.columns)
        self.assertEqual(len(result), 60)

    def test_add_daily_returns(self):
        df = _make_ohlcv_df(n_rows=10)
        result = self.handler.add_daily_returns(df)
        self.assertIn("daily_return", result.columns)


class TestMarketDataHandlerCaching(unittest.TestCase):
    """Tests for cache interaction — uses in-memory-only cache to avoid disk persistence between runs."""

    def _mem_cache(self):
        return DataCache(cache_dir="/tmp/ta_test_cache", use_disk=False)

    def test_cache_is_used_on_second_call(self):
        mock_alpaca = _make_mock_alpaca()
        handler = MarketDataHandler(cache=self._mem_cache(), alpaca=mock_alpaca, use_cache=True)

        handler.get_historical_bars("AAPL", "2024-01-01", "2024-01-05")
        handler.get_historical_bars("AAPL", "2024-01-01", "2024-01-05")

        # Provider should only be called once — second call served from in-memory cache
        self.assertEqual(mock_alpaca.get_bars.call_count, 1)

    def test_force_refresh_bypasses_cache(self):
        mock_alpaca = _make_mock_alpaca()
        handler = MarketDataHandler(cache=self._mem_cache(), alpaca=mock_alpaca, use_cache=True)

        handler.get_historical_bars("AAPL", "2024-01-01", "2024-01-05")
        handler.get_historical_bars("AAPL", "2024-01-01", "2024-01-05", force_refresh=True)

        # Both calls should hit the provider
        self.assertEqual(mock_alpaca.get_bars.call_count, 2)


# =========================================================================== #
# Utility function tests                                                       #
# =========================================================================== #

class TestUtils(unittest.TestCase):

    def test_to_utc_from_naive_datetime(self):
        naive = datetime(2024, 1, 1, 12, 0, 0)
        result = to_utc(naive)
        self.assertIsNotNone(result.tzinfo)

    def test_to_utc_from_iso_string(self):
        result = to_utc("2024-01-15T10:30:00Z")
        self.assertIsNotNone(result)
        self.assertEqual(result.year, 2024)

    def test_to_utc_none_returns_none(self):
        self.assertIsNone(to_utc(None))

    def test_to_utc_from_unix_timestamp(self):
        result = to_utc(0)
        self.assertEqual(result.year, 1970)

    def test_headline_fingerprint_stable(self):
        h = "Apple reports record quarterly earnings"
        self.assertEqual(headline_fingerprint(h), headline_fingerprint(h))

    def test_headline_fingerprint_insensitive_to_punctuation(self):
        h1 = "Apple reports record quarterly earnings!"
        h2 = "apple reports record quarterly earnings"
        self.assertEqual(headline_fingerprint(h1), headline_fingerprint(h2))

    def test_jaccard_identical(self):
        self.assertAlmostEqual(jaccard_similarity("hello world", "hello world"), 1.0)

    def test_jaccard_disjoint(self):
        self.assertAlmostEqual(jaccard_similarity("hello", "world"), 0.0)

    def test_jaccard_partial(self):
        score = jaccard_similarity("apple quarterly earnings", "apple earnings report")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_deduplicate_by_key_removes_dupes(self):
        items = [
            {"url": "http://a.com", "title": "A"},
            {"url": "http://b.com", "title": "B"},
            {"url": "http://a.com", "title": "A duplicate"},
        ]
        result = deduplicate_by_key(items, "url")
        self.assertEqual(len(result), 2)

    def test_deduplicate_by_key_keeps_no_key(self):
        items = [
            {"title": "A"},  # no url
            {"url": "http://b.com", "title": "B"},
        ]
        result = deduplicate_by_key(items, "url")
        self.assertEqual(len(result), 2)


# =========================================================================== #
# TTLCache tests                                                               #
# =========================================================================== #

class TestTTLCache(unittest.TestCase):

    def test_set_and_get(self):
        cache = TTLCache(default_ttl=10)
        cache.set("key1", {"value": 42})
        result = cache.get("key1")
        self.assertEqual(result, {"value": 42})

    def test_miss_returns_none(self):
        cache = TTLCache(default_ttl=10)
        self.assertIsNone(cache.get("nonexistent"))

    def test_expired_returns_none(self):
        import time
        cache = TTLCache(default_ttl=0.01)  # 10ms TTL
        cache.set("key", "value")
        time.sleep(0.05)
        self.assertIsNone(cache.get("key"))

    def test_invalidate(self):
        cache = TTLCache(default_ttl=10)
        cache.set("key", "value")
        cache.invalidate("key")
        self.assertIsNone(cache.get("key"))

    def test_clear(self):
        cache = TTLCache(default_ttl=10)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.clear()
        self.assertEqual(cache.size(), 0)


# =========================================================================== #
# Schema tests                                                                 #
# =========================================================================== #

class TestSchemas(unittest.TestCase):

    def test_validate_ohlcv_df_valid(self):
        df = _make_ohlcv_df()
        self.assertTrue(validate_ohlcv_df(df))

    def test_validate_ohlcv_df_empty(self):
        with self.assertRaises(ValueError):
            validate_ohlcv_df(pd.DataFrame(), raise_on_error=True)

    def test_validate_ohlcv_df_missing_columns(self):
        df = pd.DataFrame({"timestamp": [1], "close": [100]})
        with self.assertRaises(ValueError):
            validate_ohlcv_df(df, raise_on_error=True)

    def test_normalize_ohlcv_columns_uppercase(self):
        df = pd.DataFrame({"Open": [1], "High": [2], "Low": [3], "Close": [4], "Volume": [100]})
        result = normalize_ohlcv_columns(df)
        self.assertIn("open", result.columns)
        self.assertIn("close", result.columns)


# =========================================================================== #
# Entry point                                                                  #
# =========================================================================== #

if __name__ == "__main__":
    unittest.main()
