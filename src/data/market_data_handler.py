"""
market_data_handler.py
----------------------
Central orchestrator for all market (price) data requests.

This is the single entry point that downstream strategy, backtest, and
execution modules call. It abstracts away which provider was used and
handles the entire Alpaca → yfinance fallback chain.

Public interface
----------------
    handler = MarketDataHandler()
    bars = handler.get_historical_bars("AAPL", "2024-01-01", "2024-12-31")
    quote = handler.get_latest_price("AAPL")

Design
------
- Provider priority: Alpaca first → yfinance fallback
- Caching: in-memory + optional disk (DataCache)
- Validation: every output is checked before being returned
- Logging: every fetch cycle logs source, rows, and latency
- Failure tolerance: provider errors do not propagate unless ALL providers fail
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from src.config.settings import settings
from src.data.cache import DataCache
from src.data.providers.alpaca_provider import AlpacaMarketProvider, AlpacaProviderError
from src.data.providers.yfinance_provider import YFinanceProvider, YFinanceProviderError
from src.data.schemas import (
    validate_ohlcv_df,
    normalize_ohlcv_columns,
    make_empty_ohlcv_df,
    OHLCV_NUMERIC_COLUMNS,
)
from src.data.utils import (
    get_logger,
    to_utc,
    drop_ohlcv_duplicates,
    sort_by_timestamp,
    enforce_column_types,
    now_utc,
)

logger = get_logger(__name__)


# =========================================================================== #
# MarketDataHandler                                                            #
# =========================================================================== #

class MarketDataHandler:
    """
    Provider-agnostic market data orchestrator.

    Parameters
    ----------
    cache : DataCache, optional
        Shared cache instance. If None, a new one is created with defaults
        from settings.
    alpaca : AlpacaMarketProvider, optional
        Injected for testing. Defaults to a fresh provider using settings.
    yfinance : YFinanceProvider, optional
        Injected for testing.
    use_cache : bool
        Set to False to always bypass cache (useful for live execution).
    """

    def __init__(
        self,
        cache: Optional[DataCache] = None,
        alpaca: Optional[AlpacaMarketProvider] = None,
        yfinance: Optional[YFinanceProvider] = None,
        use_cache: bool = True,
    ) -> None:
        self._cache = cache or DataCache(
            cache_dir=settings.CACHE_DIR,
            memory_ttl_bars=settings.CACHE_TTL_DAILY_BARS,
            memory_ttl_quotes=settings.CACHE_TTL_LATEST_PRICE,
        )
        self._alpaca = alpaca or AlpacaMarketProvider()
        self._yfinance = yfinance or YFinanceProvider()
        self._use_cache = use_cache

    # ------------------------------------------------------------------ #
    # Primary public methods                                               #
    # ------------------------------------------------------------------ #

    def get_historical_bars(
        self,
        ticker: str,
        start: str | datetime,
        end: str | datetime,
        timeframe: str = "1Day",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV bars for a ticker.

        Tries Alpaca first. Falls back to yfinance if Alpaca is unavailable
        or returns an empty result. Caches successful results.

        Parameters
        ----------
        ticker : str
            Stock symbol (e.g. "AAPL", "SPY").
        start : str | datetime
            Start date/time (inclusive). Accepts ISO strings or datetime objects.
        end : str | datetime
            End date/time (inclusive).
        timeframe : str
            Bar frequency: '1Day', '1Hour', '5Min', etc.
        force_refresh : bool
            If True, bypass cache and re-fetch from provider.

        Returns
        -------
        pd.DataFrame
            Normalized OHLCV DataFrame. Empty DataFrame if all providers fail.

        Notes
        -----
        - Timestamps are UTC-aware.
        - Columns: timestamp, open, high, low, close, volume, symbol, source, timeframe, is_fallback
        - Rows are sorted ascending by timestamp.
        - No look-ahead bias: end is treated as an inclusive upper bound.
        """
        cache_key = self._bars_cache_key(ticker, start, end, timeframe)

        # --- Try cache first ---
        if self._use_cache and not force_refresh:
            cached = self._cache.get_bars(cache_key)
            if cached is not None:
                logger.info(
                    "MarketDataHandler: cache HIT for %s bars (%d rows)", ticker, len(cached)
                )
                return cached

        # --- Try Alpaca ---
        df = self._try_alpaca_bars(ticker, start, end, timeframe)

        # --- Fall back to yfinance ---
        if df is None or df.empty:
            logger.info(
                "MarketDataHandler: Alpaca unavailable/empty for %s — trying yfinance.", ticker
            )
            df = self._try_yfinance_bars(ticker, start, end, timeframe)

        # --- Handle complete failure ---
        if df is None or df.empty:
            logger.error(
                "MarketDataHandler: ALL providers failed for %s bars (%s → %s).",
                ticker, start, end,
            )
            return make_empty_ohlcv_df()

        # --- Validate and cache ---
        df = self.clean_ohlcv(df)
        try:
            validate_ohlcv_df(df, raise_on_error=True)
        except ValueError as exc:
            logger.error("MarketDataHandler: validation error — %s", exc)
            return make_empty_ohlcv_df()

        if self._use_cache:
            self._cache.set_bars(cache_key, df)

        logger.info(
            "MarketDataHandler: returning %d bars for %s (source=%s)",
            len(df), ticker, df["source"].iloc[0] if not df.empty else "unknown",
        )
        return df

    def get_latest_price(
        self,
        ticker: str,
        force_refresh: bool = False,
    ) -> dict:
        """
        Fetch the latest price quote for a ticker.

        Tries Alpaca first (real-time quote), falls back to yfinance (last close).

        Parameters
        ----------
        ticker : str
        force_refresh : bool
            Bypass cache.

        Returns
        -------
        dict with keys:
            ticker, last_price, ask_price, bid_price, timestamp, source

        Returns a dict with last_price=None if all providers fail.
        """
        # --- Try cache ---
        if self._use_cache and not force_refresh:
            cached = self._cache.get_quote(ticker)
            if cached is not None:
                logger.debug("MarketDataHandler: quote cache HIT for %s", ticker)
                return cached

        # --- Try Alpaca ---
        quote = self._try_alpaca_quote(ticker)

        # --- Fall back to yfinance ---
        if quote is None:
            logger.info(
                "MarketDataHandler: Alpaca quote unavailable for %s — trying yfinance.", ticker
            )
            quote = self._try_yfinance_quote(ticker)

        # --- Handle failure ---
        if quote is None:
            logger.error("MarketDataHandler: ALL providers failed for %s quote.", ticker)
            return {
                "ticker": ticker,
                "last_price": None,
                "ask_price": None,
                "bid_price": None,
                "timestamp": now_utc(),
                "source": "none",
            }

        if self._use_cache:
            self._cache.set_quote(ticker, quote, ttl=settings.CACHE_TTL_LATEST_PRICE)

        return quote

    # ------------------------------------------------------------------ #
    # Data cleaning and validation                                         #
    # ------------------------------------------------------------------ #

    def clean_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the full standard cleaning pipeline to an OHLCV DataFrame.

        Steps
        -----
        1. Lowercase and normalize column names
        2. Cast OHLCV columns to float64
        3. Ensure timestamp is UTC-aware
        4. Drop rows where open/high/low/close are all NaN
        5. Drop duplicate (symbol, timestamp) rows
        6. Sort ascending by timestamp

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        if df is None or df.empty:
            return make_empty_ohlcv_df()

        df = df.copy()

        # Normalize column names
        df = normalize_ohlcv_columns(df)

        # Cast numeric columns
        df = enforce_column_types(df, list(OHLCV_NUMERIC_COLUMNS))

        # Ensure UTC-aware timestamp
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

        # Drop rows with all-NaN OHLC
        ohlc = [c for c in ["open", "high", "low", "close"] if c in df.columns]
        df = df.dropna(subset=ohlc, how="all")

        # Deduplicate and sort
        df = drop_ohlcv_duplicates(df)
        df = sort_by_timestamp(df)

        return df.reset_index(drop=True)

    def validate_bars(self, df: pd.DataFrame) -> None:
        """
        Validate a bars DataFrame. Raises ValueError on failure.

        Call this explicitly when building a backtest dataset to ensure
        data integrity before any analysis.

        Parameters
        ----------
        df : pd.DataFrame

        Raises
        ------
        ValueError
        """
        validate_ohlcv_df(df, raise_on_error=True)

    # ------------------------------------------------------------------ #
    # Technical feature helpers                                            #
    # ------------------------------------------------------------------ #

    def add_moving_averages(
        self,
        df: pd.DataFrame,
        windows: list[int] = [20, 50, 200],
    ) -> pd.DataFrame:
        """
        Add simple moving average columns to a bars DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must have a 'close' column.
        windows : list[int]
            Window sizes in bars.

        Returns
        -------
        pd.DataFrame
            Original DataFrame with added SMA columns (sma_20, sma_50, etc.)
        """
        if "close" not in df.columns:
            logger.warning("add_moving_averages: 'close' column not found.")
            return df

        df = df.copy()
        for w in windows:
            df[f"sma_{w}"] = df["close"].rolling(window=w, min_periods=1).mean()
        return df

    def add_rsi(
        self,
        df: pd.DataFrame,
        period: int = 14,
    ) -> pd.DataFrame:
        """
        Add a Relative Strength Index (RSI) column using Wilder's smoothing.

        Parameters
        ----------
        df : pd.DataFrame
            Must have a 'close' column.
        period : int
            RSI lookback window (default: 14).

        Returns
        -------
        pd.DataFrame
            Original DataFrame with added 'rsi_14' column.
        """
        if "close" not in df.columns:
            logger.warning("add_rsi: 'close' column not found.")
            return df

        df = df.copy()
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, float("nan"))
        df[f"rsi_{period}"] = 100.0 - (100.0 / (1.0 + rs))
        return df

    def add_adx(
        self,
        df: pd.DataFrame,
        period: int = 14,
    ) -> pd.DataFrame:
        """
        Add an Average Directional Index (ADX) column using Wilder's smoothing.

        ADX measures trend strength regardless of direction:
          - ADX > 25 → strong trend (good to trade)
          - ADX < 20 → choppy / ranging (avoid entry)

        Parameters
        ----------
        df : pd.DataFrame
            Must have 'high', 'low', 'close' columns.
        period : int
            ADX lookback window (default: 14).

        Returns
        -------
        pd.DataFrame
            Original DataFrame with added 'adx_{period}' column.
        """
        if not all(c in df.columns for c in ("high", "low", "close")):
            logger.warning("add_adx: 'high', 'low', or 'close' column not found.")
            return df

        df = df.copy()

        # True Range
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Directional Movement
        up_move = df["high"] - df["high"].shift(1)
        down_move = df["low"].shift(1) - df["low"]

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        # Wilder smoothing (EMA with alpha = 1/period)
        alpha = 1.0 / period
        atr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr.replace(0, float("nan"))
        minus_di = 100 * minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr.replace(0, float("nan"))

        # ADX = smoothed absolute DI difference / DI sum
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))
        df[f"adx_{period}"] = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

        return df

    def add_daily_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a 'daily_return' column (percentage change in close).

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        if "close" not in df.columns:
            return df
        df = df.copy()
        df["daily_return"] = df["close"].pct_change()
        return df

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _try_alpaca_bars(
        self,
        ticker: str,
        start,
        end,
        timeframe: str,
    ) -> Optional[pd.DataFrame]:
        """Attempt to fetch bars from Alpaca. Returns None on any failure."""
        if not self._alpaca.is_healthy():
            logger.debug("MarketDataHandler: Alpaca provider not healthy, skipping.")
            return None
        try:
            return self._alpaca.get_bars(ticker, start, end, timeframe)
        except AlpacaProviderError as exc:
            logger.warning("MarketDataHandler: Alpaca bars failed — %s", exc)
            return None

    def _try_yfinance_bars(
        self,
        ticker: str,
        start,
        end,
        timeframe: str,
    ) -> Optional[pd.DataFrame]:
        """Attempt to fetch bars from yfinance. Returns None on any failure."""
        try:
            return self._yfinance.get_bars(ticker, start, end, timeframe)
        except YFinanceProviderError as exc:
            logger.warning("MarketDataHandler: yfinance bars failed — %s", exc)
            return None

    def _try_alpaca_quote(self, ticker: str) -> Optional[dict]:
        if not self._alpaca.is_healthy():
            return None
        try:
            return self._alpaca.get_latest_quote(ticker)
        except AlpacaProviderError as exc:
            logger.warning("MarketDataHandler: Alpaca quote failed — %s", exc)
            return None

    def _try_yfinance_quote(self, ticker: str) -> Optional[dict]:
        try:
            return self._yfinance.get_latest_price(ticker)
        except YFinanceProviderError as exc:
            logger.warning("MarketDataHandler: yfinance quote failed — %s", exc)
            return None

    @staticmethod
    def _bars_cache_key(ticker: str, start, end, timeframe: str) -> str:
        """Generate a stable cache key for a bars request."""
        # Normalise dates to ISO strings for consistent keys
        start_str = to_utc(start).strftime("%Y%m%d") if to_utc(start) else str(start)
        end_str = to_utc(end).strftime("%Y%m%d") if to_utc(end) else str(end)
        return f"bars_{ticker}_{timeframe}_{start_str}_{end_str}"
