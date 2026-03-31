"""
yfinance_provider.py
--------------------
Secondary / fallback market data provider using Yahoo Finance (yfinance).

Used when:
  - Alpaca credentials are missing
  - Alpaca is rate-limited or degraded
  - Extended historical data is needed beyond Alpaca's free-tier coverage

Wraps yfinance.download() and yfinance.Ticker() into the standard OHLCV schema.

SDK: yfinance (pip install yfinance)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from src.data.schemas import normalize_ohlcv_columns, OHLCV_NUMERIC_COLUMNS
from src.data.utils import to_utc, drop_ohlcv_duplicates, sort_by_timestamp

logger = logging.getLogger(__name__)


# =========================================================================== #
# Custom exception                                                             #
# =========================================================================== #

class YFinanceProviderError(Exception):
    """Raised when yfinance cannot serve a request."""


# =========================================================================== #
# Timeframe mapping                                                            #
# =========================================================================== #

# Map internal timeframe strings to yfinance interval strings
_YF_INTERVAL_MAP: dict[str, str] = {
    "1Min":   "1m",
    "5Min":   "5m",
    "15Min":  "15m",
    "30Min":  "30m",
    "1Hour":  "1h",
    "4Hour":  "60m",   # closest available
    "1Day":   "1d",
    "1Week":  "1wk",
    "1Month": "1mo",
}

# yfinance imposes limitations on intraday lookback
_YF_INTRADAY_MAX_DAYS: dict[str, int] = {
    "1m": 7,
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "1h": 730,
    "60m": 730,
}


# =========================================================================== #
# YFinanceProvider                                                             #
# =========================================================================== #

class YFinanceProvider:
    """
    Adapter around yfinance for historical bars and latest price lookups.

    No circuit-breaker needed here — yfinance is stateless and free.
    Errors are wrapped in YFinanceProviderError for uniform upstream handling.
    """

    def get_bars(
        self,
        ticker: str,
        start: str | datetime,
        end: str | datetime,
        timeframe: str = "1Day",
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV bars using yfinance.

        Parameters
        ----------
        ticker : str
        start : str | datetime
        end : str | datetime
        timeframe : str
            Internal timeframe string ('1Day', '1Hour', etc.)

        Returns
        -------
        pd.DataFrame
            Normalized OHLCV DataFrame.

        Raises
        ------
        YFinanceProviderError
        """
        try:
            import yfinance as yf
        except ImportError:
            raise YFinanceProviderError(
                "yfinance is not installed. Run: pip install yfinance"
            )

        interval = _YF_INTERVAL_MAP.get(timeframe)
        if not interval:
            raise YFinanceProviderError(
                f"Unknown timeframe '{timeframe}'. Valid: {list(_YF_INTERVAL_MAP.keys())}"
            )

        start_dt = to_utc(start)
        end_dt = to_utc(end)

        logger.info(
            "YFinanceProvider.get_bars: ticker=%s, %s → %s, interval=%s",
            ticker, start_dt.date(), end_dt.date(), interval,
        )

        try:
            t0 = time.monotonic()
            df = yf.download(
                ticker,
                start=start_dt,
                end=end_dt,
                interval=interval,
                auto_adjust=True,       # adjust for splits/dividends
                progress=False,
            )
            elapsed = time.monotonic() - t0

            if df is None or df.empty:
                logger.warning(
                    "YFinanceProvider: empty response for %s (%s → %s)",
                    ticker, start_dt.date(), end_dt.date(),
                )
                return pd.DataFrame()

            logger.info(
                "YFinanceProvider: received %d rows in %.2fs", len(df), elapsed
            )

            df = self._normalise(df, ticker=ticker, timeframe=timeframe)
            return df

        except Exception as exc:
            raise YFinanceProviderError(
                f"YFinanceProvider.get_bars failed for {ticker}: {exc}"
            ) from exc

    def get_latest_price(self, ticker: str) -> dict:
        """
        Fetch the latest available price using yfinance.

        Returns the most recent daily close from fast_info where possible,
        falling back to the last row of a 5-day 1-hour download.

        Returns
        -------
        dict with keys: ticker, last_price, timestamp, source
        """
        try:
            import yfinance as yf
        except ImportError:
            raise YFinanceProviderError("yfinance is not installed.")

        logger.info("YFinanceProvider.get_latest_price: ticker=%s", ticker)

        try:
            t0 = time.monotonic()
            tkr = yf.Ticker(ticker)

            # Prefer fast_info for speed (single HTTP call)
            try:
                fi = tkr.fast_info
                last_price = float(fi.last_price)
                # fast_info doesn't always have a timestamp
                ts = to_utc(fi.last_volume) if hasattr(fi, "last_trade_time") else None
                if ts is None:
                    ts = datetime.now(tz=timezone.utc)
            except Exception:
                # Fallback: download last 2 days
                df = yf.download(
                    ticker,
                    period="2d",
                    interval="1h",
                    auto_adjust=True,
                    progress=False,
                )
                if df is None or df.empty:
                    raise YFinanceProviderError(f"No price data for {ticker}")
                last_price = float(df["Close"].iloc[-1])
                ts = to_utc(df.index[-1])

            elapsed = time.monotonic() - t0
            logger.info(
                "YFinanceProvider: latest price for %s = $%.4f (%.2fs)",
                ticker, last_price, elapsed,
            )

            return {
                "ticker": ticker,
                "last_price": last_price,
                "ask_price": None,
                "bid_price": None,
                "timestamp": ts,
                "source": "yfinance",
            }

        except YFinanceProviderError:
            raise
        except Exception as exc:
            raise YFinanceProviderError(
                f"YFinanceProvider.get_latest_price failed for {ticker}: {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _normalise(self, df: pd.DataFrame, ticker: str, timeframe: str) -> pd.DataFrame:
        """Convert a yfinance DataFrame to the standard OHLCV schema."""
        df = df.copy()

        # yfinance returns a DatetimeIndex or MultiIndex for multi-ticker
        # Handle MultiIndex columns (when yfinance returns multi-ticker format)
        if isinstance(df.columns, pd.MultiIndex):
            # For single ticker, flatten MultiIndex
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        df = df.reset_index()

        # Rename Date/Datetime index column to timestamp
        for candidate in ["Datetime", "Date", "datetime", "date"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "timestamp"})
                break

        # Normalize column names to lowercase
        df = normalize_ohlcv_columns(df)

        # Drop adj_close if present (auto_adjust=True already incorporates it into close)
        df = df.drop(columns=["adj_close", "adj close"], errors="ignore")

        # Cast numeric columns
        for col in OHLCV_NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Ensure timestamp is timezone-aware UTC
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            # yfinance daily bars come as tz-naive date → localize to UTC
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
            else:
                df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

        # Add metadata
        df["symbol"] = ticker
        df["source"] = "yfinance"
        df["timeframe"] = timeframe
        df["is_fallback"] = True

        # Standard cleanup
        df = df.dropna(subset=["open", "high", "low", "close"], how="all")
        df = drop_ohlcv_duplicates(df)
        df = sort_by_timestamp(df)

        # Keep standard columns
        keep = ["timestamp", "open", "high", "low", "close", "volume",
                "symbol", "source", "timeframe", "is_fallback"]
        df = df[[c for c in keep if c in df.columns]]

        return df
