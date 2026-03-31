"""
alpaca_provider.py
------------------
Primary market data provider using the Alpaca Markets API (alpaca-py SDK).

Responsibilities:
  - Fetch historical OHLCV bars for one or more tickers
  - Fetch the latest quote snapshot for a ticker
  - Maintain a circuit-breaker state: if Alpaca fails repeatedly, mark it as
    degraded and stop hitting the API for PROVIDER_COOLDOWN_SECONDS
  - Return data in the normalized schema; never return provider-specific formats

SDK used: alpaca-py (pip install alpaca-py)
  from alpaca.data import StockHistoricalDataClient
  from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
  from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

Fallback: MarketDataHandler is responsible for the fallback decision.
This class only raises AlpacaProviderError when it cannot serve the request.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from src.config.settings import settings
from src.data.schemas import normalize_ohlcv_columns, OHLCV_NUMERIC_COLUMNS
from src.data.utils import to_utc, safe_float, drop_ohlcv_duplicates, sort_by_timestamp

logger = logging.getLogger(__name__)


# =========================================================================== #
# Custom exceptions                                                            #
# =========================================================================== #

class AlpacaProviderError(Exception):
    """Raised when Alpaca cannot serve a request and no retry should be attempted."""


class AlpacaAuthError(AlpacaProviderError):
    """Raised on invalid credentials."""


class AlpacaRateLimitError(AlpacaProviderError):
    """Raised when rate limit is exceeded."""


# =========================================================================== #
# Timeframe mapping                                                            #
# =========================================================================== #

# Map our internal timeframe strings to alpaca-py TimeFrame objects
def _get_alpaca_timeframe(timeframe_str: str):
    """Convert '1Day', '1Hour', '5Min' etc. to an alpaca-py TimeFrame."""
    try:
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except ImportError:
        raise AlpacaProviderError("alpaca-py is not installed. Run: pip install alpaca-py")

    _map = {
        "1Min":  TimeFrame(1, TimeFrameUnit.Minute),
        "5Min":  TimeFrame(5, TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "30Min": TimeFrame(30, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
        "4Hour": TimeFrame(4, TimeFrameUnit.Hour),
        "1Day":  TimeFrame(1, TimeFrameUnit.Day),
        "1Week": TimeFrame(1, TimeFrameUnit.Week),
        "1Month":TimeFrame(1, TimeFrameUnit.Month),
    }
    tf = _map.get(timeframe_str)
    if tf is None:
        raise AlpacaProviderError(
            f"Unknown timeframe '{timeframe_str}'. Valid options: {list(_map.keys())}"
        )
    return tf


# =========================================================================== #
# AlpacaMarketProvider                                                        #
# =========================================================================== #

class AlpacaMarketProvider:
    """
    Thin adapter around the alpaca-py StockHistoricalDataClient.

    Circuit-breaker:
      - Tracks consecutive failures.
      - After MAX_FAILURES consecutive failures, marks the provider as
        degraded for COOLDOWN_SECONDS.
      - During cooldown, get_bars() / get_latest_quote() raise immediately
        without hitting the API.

    Parameters
    ----------
    api_key : str, optional
        Defaults to settings.ALPACA_API_KEY.
    api_secret : str, optional
        Defaults to settings.ALPACA_API_SECRET.
    """

    MAX_FAILURES: int = 3
    COOLDOWN_SECONDS: float = float(settings.PROVIDER_COOLDOWN_SECONDS)

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ) -> None:
        self._api_key = api_key or settings.ALPACA_API_KEY
        self._api_secret = api_secret or settings.ALPACA_API_SECRET

        self._client = None           # lazy init
        self._failure_count: int = 0
        self._degraded_until: float = 0.0  # monotonic timestamp
        self._available: bool = bool(self._api_key and self._api_secret)

        if not self._available:
            logger.warning(
                "AlpacaMarketProvider: credentials not configured. "
                "This provider will be skipped."
            )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def is_healthy(self) -> bool:
        """True if the provider can currently serve requests."""
        if not self._available:
            return False
        if time.monotonic() < self._degraded_until:
            remaining = self._degraded_until - time.monotonic()
            logger.debug(
                "AlpacaMarketProvider: degraded, %.0fs remaining in cooldown.", remaining
            )
            return False
        return True

    def get_bars(
        self,
        ticker: str,
        start: str | datetime,
        end: str | datetime,
        timeframe: str = "1Day",
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV bars for a single ticker.

        Parameters
        ----------
        ticker : str
            Stock symbol, e.g. "AAPL".
        start : str | datetime
            Start of the window (inclusive). ISO string or datetime.
        end : str | datetime
            End of the window (inclusive). ISO string or datetime.
        timeframe : str
            Bar frequency: '1Day', '1Hour', '5Min', etc.

        Returns
        -------
        pd.DataFrame
            Normalized OHLCV DataFrame with columns:
            timestamp, open, high, low, close, volume, symbol, source

        Raises
        ------
        AlpacaProviderError
            If the request fails or credentials are invalid.
        """
        self._check_healthy()

        try:
            from alpaca.data import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.enums import DataFeed
        except ImportError:
            raise AlpacaProviderError(
                "alpaca-py is not installed. Run: pip install alpaca-py"
            )

        client = self._get_client()
        tf = _get_alpaca_timeframe(timeframe)

        start_dt = to_utc(start)
        end_dt = to_utc(end)

        logger.info(
            "AlpacaMarketProvider.get_bars: ticker=%s, %s → %s, timeframe=%s",
            ticker, start_dt.date(), end_dt.date(), timeframe,
        )

        try:
            t0 = time.monotonic()
            request = StockBarsRequest(
                symbol_or_symbols=ticker,
                start=start_dt,
                end=end_dt,
                timeframe=tf,
                feed=DataFeed.IEX,  # Enforce free-tier compatible feed
            )
            bars_data = client.get_stock_bars(request)
            elapsed = time.monotonic() - t0

            df = bars_data.df.reset_index()
            rows = len(df)
            logger.info(
                "AlpacaMarketProvider: received %d rows in %.2fs", rows, elapsed
            )

            if df.empty:
                logger.warning("AlpacaMarketProvider: empty response for %s", ticker)
                self._record_success()
                return pd.DataFrame()

            df = self._normalise(df, ticker=ticker, timeframe=timeframe)
            self._record_success()
            return df

        except Exception as exc:
            self._record_failure(exc)
            raise AlpacaProviderError(
                f"AlpacaMarketProvider.get_bars failed for {ticker}: {exc}"
            ) from exc

    def get_latest_quote(self, ticker: str) -> dict:
        """
        Fetch the latest quote snapshot for a ticker.

        Returns
        -------
        dict with keys: ticker, ask_price, bid_price, last_price, timestamp, source
        """
        self._check_healthy()

        try:
            from alpaca.data import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestQuoteRequest
            from alpaca.data.enums import DataFeed
        except ImportError:
            raise AlpacaProviderError("alpaca-py is not installed.")

        client = self._get_client()

        logger.info("AlpacaMarketProvider.get_latest_quote: ticker=%s", ticker)

        try:
            t0 = time.monotonic()
            request = StockLatestQuoteRequest(symbol_or_symbols=ticker, feed=DataFeed.IEX)
            quotes = client.get_stock_latest_quote(request)
            elapsed = time.monotonic() - t0

            quote = quotes[ticker]
            result = {
                "ticker": ticker,
                "ask_price": safe_float(quote.ask_price),
                "bid_price": safe_float(quote.bid_price),
                "last_price": safe_float((quote.ask_price + quote.bid_price) / 2),
                "timestamp": to_utc(quote.timestamp),
                "source": "alpaca",
            }
            logger.info(
                "AlpacaMarketProvider: quote for %s = $%.4f (%.2fs)",
                ticker, result["last_price"], elapsed,
            )
            self._record_success()
            return result

        except Exception as exc:
            self._record_failure(exc)
            raise AlpacaProviderError(
                f"AlpacaMarketProvider.get_latest_quote failed for {ticker}: {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _get_client(self):
        """Lazy-initialise the alpaca-py client."""
        if self._client is None:
            from alpaca.data import StockHistoricalDataClient
            self._client = StockHistoricalDataClient(
                api_key=self._api_key,
                secret_key=self._api_secret,
            )
        return self._client

    def _check_healthy(self) -> None:
        if not self.is_healthy():
            raise AlpacaProviderError(
                "AlpacaMarketProvider is degraded or credentials are missing."
            )

    def _record_success(self) -> None:
        self._failure_count = 0

    def _record_failure(self, exc: Exception) -> None:
        self._failure_count += 1
        logger.warning(
            "AlpacaMarketProvider: failure #%d — %s", self._failure_count, exc
        )
        if self._failure_count >= self.MAX_FAILURES:
            self._degraded_until = time.monotonic() + self.COOLDOWN_SECONDS
            logger.error(
                "AlpacaMarketProvider: circuit breaker OPEN. "
                "Will retry after %.0fs. Cause: %s",
                self.COOLDOWN_SECONDS, exc,
            )

    def _normalise(self, df: pd.DataFrame, ticker: str, timeframe: str) -> pd.DataFrame:
        """Convert alpaca-py DataFrame to the standard OHLCV schema."""
        df = df.copy()

        # alpaca-py multi-level index: (symbol, timestamp) or just (timestamp,)
        # After reset_index, columns depend on whether one or multiple tickers were requested
        df = normalize_ohlcv_columns(df)

        # Rename alpaca-specific column 't' → 'timestamp' if present after normalisation
        if "timestamp" not in df.columns:
            for candidate in ["t", "time"]:
                if candidate in df.columns:
                    df = df.rename(columns={candidate: "timestamp"})
                    break

        # Ensure symbol column
        if "symbol" not in df.columns:
            # alpaca-py puts symbol in index level when multiple tickers; after reset_index it's a column
            if "symbol" in df.columns:
                pass
            else:
                df["symbol"] = ticker

        # Normalize timestamps → UTC-aware pandas Timestamps
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Cast numeric columns
        for col in OHLCV_NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Metadata
        df["source"] = "alpaca"
        df["timeframe"] = timeframe
        df["is_fallback"] = False

        # Drop provider extras we don't need at strategy level
        drop_cols = [c for c in ["vwap", "trade_count", "adj_close"] if c in df.columns]
        df = df.drop(columns=drop_cols, errors="ignore")

        # Standard cleanup
        df = drop_ohlcv_duplicates(df)
        df = sort_by_timestamp(df)

        # Keep only required + metadata columns
        keep = ["timestamp", "open", "high", "low", "close", "volume",
                "symbol", "source", "timeframe", "is_fallback"]
        df = df[[c for c in keep if c in df.columns]]

        return df
