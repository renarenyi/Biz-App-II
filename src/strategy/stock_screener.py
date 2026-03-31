"""
strategy/stock_screener.py
--------------------------
Dynamic stock screening module that selects the best tickers for trading
from a larger universe based on quantitative criteria.

Supports two modes:
  1. screen()               — Live screening (fetches data from APIs)
  2. screen_from_history()  — Historical screening from pre-fetched data
                              (used for monthly rotation in backtesting)

Screening Criteria (0-25 each, total 0-100)
------------------
  1. Liquidity Score   — Average daily dollar volume (price x volume)
  2. Trend Score       — Price vs SMA-50 (uptrend strength)
  3. Volatility Score  — Prefer moderate volatility (20-35% annualized)
  4. News Coverage     — Number of recent articles (optional, for live mode)

Usage
-----
    from src.strategy.stock_screener import StockScreener
    screener = StockScreener(market_handler, news_fetcher)
    top_tickers = screener.screen(universe=UNIVERSE, top_n=4)

    # For backtest rotation:
    results = screener.screen_from_history(all_market_data, as_of_date, top_n=4)
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Default universe: popular, liquid US stocks across sectors
DEFAULT_UNIVERSE = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "TSLA", "AMD", "CRM", "ORCL",
    # Finance
    "JPM", "GS", "V", "MA",
    # Healthcare
    "JNJ", "UNH", "PFE",
    # Consumer
    "WMT", "COST", "DIS",
    # Energy
    "XOM", "CVX",
    # Industrial
    "BA", "CAT",
]


class StockScreener:
    """
    Screens a universe of stocks and selects the top N candidates
    based on liquidity, trend strength, volatility, and news coverage.
    """

    def __init__(self, market_handler, news_fetcher=None):
        self._market = market_handler
        self._news = news_fetcher

    def screen(
        self,
        universe: Optional[list[str]] = None,
        top_n: int = 4,
        lookback_days: int = 60,
        news_lookback_days: int = 30,
    ) -> list[dict]:
        """
        Screen stocks using live/cached API data.

        Returns list of dicts sorted by total_score descending.
        """
        universe = universe or DEFAULT_UNIVERSE
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=lookback_days)
        news_start = end_date - timedelta(days=news_lookback_days)

        logger.info(
            "StockScreener: screening %d tickers (lookback=%dd, news=%dd)",
            len(universe), lookback_days, news_lookback_days,
        )

        candidates = []
        for ticker in universe:
            try:
                score = self._score_ticker(ticker, start_date, end_date, news_start)
                if score is not None:
                    candidates.append(score)
            except Exception as e:
                logger.warning("  %s: SKIPPED (%s)", ticker, str(e)[:60])

        return self._rank(candidates, top_n)

    def screen_from_history(
        self,
        all_market_data: dict[str, pd.DataFrame],
        as_of_date: datetime,
        lookback_days: int = 60,
        top_n: int = 4,
    ) -> list[dict]:
        """
        Screen stocks using pre-fetched historical data.

        This avoids redundant API calls during backtest rotation.
        Only uses market-data metrics (no news API calls).

        Parameters
        ----------
        all_market_data : dict mapping ticker -> DataFrame of OHLCV bars
        as_of_date      : screen using only data up to this date
        lookback_days   : how many days of history to evaluate
        top_n           : number of top tickers to return
        """
        cutoff = as_of_date
        start = cutoff - timedelta(days=lookback_days)

        candidates = []
        for ticker, df in all_market_data.items():
            try:
                # Filter to only data available as of the screening date
                mask = df["timestamp"] <= cutoff
                if hasattr(cutoff, 'tzinfo') and cutoff.tzinfo:
                    # Handle timezone-aware comparison
                    timestamps = pd.to_datetime(df["timestamp"], utc=True)
                    mask = timestamps <= cutoff
                sub_df = df[mask.values if hasattr(mask, 'values') else mask].copy()

                if len(sub_df) < 20:
                    continue

                # Use last `lookback_days` worth of bars
                sub_df = sub_df.tail(lookback_days)

                score = self._score_dataframe(ticker, sub_df)
                if score is not None:
                    candidates.append(score)
            except Exception as e:
                logger.debug("  %s: SKIPPED (%s)", ticker, str(e)[:60])

        return self._rank(candidates, top_n)

    # ------------------------------------------------------------------ #
    # Scoring helpers                                                      #
    # ------------------------------------------------------------------ #

    def _rank(self, candidates: list[dict], top_n: int) -> list[dict]:
        """Sort by total score and assign ranks."""
        candidates.sort(key=lambda x: x["total_score"], reverse=True)
        for i, c in enumerate(candidates):
            c["rank"] = i + 1

        selected = [c["ticker"] for c in candidates[:top_n]]
        logger.info("StockScreener: selected top %d -> %s", top_n, selected)
        return candidates

    def _score_ticker(
        self, ticker: str, start_date: datetime, end_date: datetime,
        news_start: datetime,
    ) -> Optional[dict]:
        """Fetch data and score a single ticker."""
        df = self._market.get_historical_bars(
            ticker, start=start_date, end=end_date, timeframe="1Day"
        )
        if df is None or len(df) < 20:
            return None

        df = self._market.add_moving_averages(df, windows=[50])
        score = self._score_dataframe(ticker, df)
        if score is None:
            return None

        # Add news coverage score (only in live mode)
        if self._news is not None:
            try:
                articles = self._news.get_recent_news(
                    ticker, start=news_start, end=end_date
                )
                n_articles = len(articles) if articles else 0
                if n_articles >= 500:
                    news_score = 25.0
                elif n_articles >= 200:
                    news_score = 20.0
                elif n_articles >= 100:
                    news_score = 15.0
                elif n_articles >= 50:
                    news_score = 10.0
                elif n_articles >= 20:
                    news_score = 5.0
                else:
                    news_score = 2.0
            except Exception:
                news_score = 0.0
            score["news_score"] = round(news_score, 1)
            score["total_score"] = round(score["total_score"] + news_score, 1)

        return score

    def _score_dataframe(self, ticker: str, df: pd.DataFrame) -> Optional[dict]:
        """
        Score a ticker from a DataFrame of OHLCV bars.
        Returns scores for liquidity, trend, volatility (no news).
        """
        if len(df) < 20:
            return None

        # Ensure SMA-50 is computed
        if "SMA_50" not in df.columns:
            df = self._market.add_moving_averages(df, windows=[50]) if self._market else df

        closes = df["close"].astype(float)
        volumes = df["volume"].astype(float)

        # ── 1. Liquidity Score (0-33) ─────────────────────────
        avg_dollar_vol = (closes * volumes).mean()
        if avg_dollar_vol > 0:
            liquidity_raw = math.log10(avg_dollar_vol)
            liquidity_score = min(33, max(0, (liquidity_raw - 6) / 4 * 33))
        else:
            liquidity_score = 0.0

        # ── 2. Trend Score (0-33) ─────────────────────────────
        trend_score = 0.0
        if "SMA_50" in df.columns:
            sma = df["SMA_50"].astype(float)
            valid_mask = sma.notna() & (sma > 0)
            if valid_mask.sum() > 0:
                above_sma = (closes[valid_mask] > sma[valid_mask]).mean()
                latest_close = float(closes.iloc[-1])
                latest_sma = float(sma.dropna().iloc[-1]) if sma.notna().any() else latest_close
                trend_pct = (latest_close - latest_sma) / latest_sma if latest_sma > 0 else 0
                trend_score = min(33, max(0,
                    above_sma * 20 +
                    min(13, max(0, trend_pct * 100))
                ))

        # ── 3. Volatility Score (0-34) ────────────────────────
        daily_returns = closes.pct_change().dropna()
        annualized_vol = 0.0
        volatility_score = 0.0
        if len(daily_returns) > 5:
            annualized_vol = float(daily_returns.std() * math.sqrt(252) * 100)
            if 20 <= annualized_vol <= 35:
                volatility_score = 34.0
            elif 15 <= annualized_vol < 20 or 35 < annualized_vol <= 50:
                volatility_score = 24.0
            elif 10 <= annualized_vol < 15 or 50 < annualized_vol <= 70:
                volatility_score = 14.0
            else:
                volatility_score = 5.0

        total_score = liquidity_score + trend_score + volatility_score

        return {
            "ticker": ticker,
            "liquidity_score": round(liquidity_score, 1),
            "trend_score": round(trend_score, 1),
            "volatility_score": round(volatility_score, 1),
            "news_score": 0.0,
            "total_score": round(total_score, 1),
            "avg_dollar_volume": round(avg_dollar_vol, 0),
            "annualized_vol": round(annualized_vol, 1),
        }


# ===================================================================== #
# Rotation Schedule Builder                                               #
# ===================================================================== #

def build_rotation_schedule(
    screener: StockScreener,
    all_market_data: dict[str, pd.DataFrame],
    trading_dates: list[datetime],
    rotation_period_days: int = 21,
    top_n: int = 4,
) -> dict[str, list[str]]:
    """
    Pre-compute which tickers are active on each rotation window.

    Parameters
    ----------
    screener         : StockScreener instance
    all_market_data  : {ticker: DataFrame} for the full universe
    trading_dates    : list of all trading dates in the backtest
    rotation_period_days : re-screen every N trading days (21 ≈ monthly)
    top_n            : number of tickers to select per window

    Returns
    -------
    dict mapping date_str (YYYY-MM-DD) to list of active tickers
    """
    schedule = {}
    current_active = []

    for i, dt in enumerate(trading_dates):
        if i % rotation_period_days == 0:
            # Re-screen at this date
            results = screener.screen_from_history(
                all_market_data, as_of_date=dt, top_n=top_n,
            )
            current_active = [r["ticker"] for r in results[:top_n]]
            logger.info(
                "Rotation [day %d, %s]: %s",
                i, dt.strftime("%Y-%m-%d"), current_active,
            )

        date_str = dt.strftime("%Y-%m-%d")
        schedule[date_str] = list(current_active)

    return schedule


def print_screening_report(results: list[dict], top_n: int = 4) -> None:
    """Pretty-print the screening results to console."""
    print("\n" + "=" * 80)
    print("  STOCK SCREENING REPORT")
    print("=" * 80)
    print(f"\n  {'Rank':<6} {'Ticker':<8} {'Total':<8} {'Liquidity':<11} "
          f"{'Trend':<8} {'Volatility':<12} {'News':<8}")
    print("  " + "-" * 70)

    for r in results:
        marker = " *" if r.get("rank", 99) <= top_n else "  "
        print(f"{marker}{r['rank']:<5} {r['ticker']:<8} {r['total_score']:<8.1f} "
              f"{r['liquidity_score']:<11.1f} {r['trend_score']:<8.1f} "
              f"{r['volatility_score']:<12.1f} {r['news_score']:<8.1f}")

    selected = [r["ticker"] for r in results if r.get("rank", 99) <= top_n]
    print(f"\n  Selected: {', '.join(selected)}")
    print("=" * 80 + "\n")


def print_rotation_summary(schedule: dict[str, list[str]]) -> None:
    """Print a summary of the rotation schedule."""
    print("\n" + "=" * 80)
    print("  ROTATION SCHEDULE")
    print("=" * 80)

    prev_tickers = None
    for date_str in sorted(schedule.keys()):
        tickers = schedule[date_str]
        if tickers != prev_tickers:
            added = set(tickers) - set(prev_tickers or [])
            removed = set(prev_tickers or []) - set(tickers)
            change = ""
            if prev_tickers is not None:
                parts = []
                if added:
                    parts.append(f"+{','.join(sorted(added))}")
                if removed:
                    parts.append(f"-{','.join(sorted(removed))}")
                change = f"  ({' '.join(parts)})" if parts else "  (no change)"
            print(f"  {date_str}: {', '.join(tickers)}{change}")
            prev_tickers = tickers

    print("=" * 80 + "\n")
