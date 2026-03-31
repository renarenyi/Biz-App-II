"""
strategy/stock_screener.py
--------------------------
Dynamic stock screening module that selects the best tickers for trading
from a larger universe based on quantitative criteria.

Screening Criteria
------------------
  1. Liquidity Score   — Average daily dollar volume (price × volume)
  2. Trend Score       — Price vs SMA-50 (uptrend strength)
  3. Volatility Score  — Prefer moderate volatility (not too flat, not too wild)
  4. News Coverage     — Number of recent news articles (more = better NLP signal)

Usage
-----
    from src.strategy.stock_screener import StockScreener
    screener = StockScreener(market_handler, news_fetcher)
    top_tickers = screener.screen(universe=UNIVERSE, top_n=4)
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
        Screen stocks and return the top N ranked candidates.

        Parameters
        ----------
        universe        : list of ticker symbols to evaluate
        top_n           : number of top tickers to select
        lookback_days   : days of price history for screening
        news_lookback_days : days to check for news coverage

        Returns
        -------
        list of dicts with keys: ticker, rank, total_score, and component scores
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
                    logger.info(
                        "  %s: liquidity=%.1f  trend=%.1f  volatility=%.1f  news=%.1f  TOTAL=%.1f",
                        ticker, score["liquidity_score"], score["trend_score"],
                        score["volatility_score"], score["news_score"], score["total_score"],
                    )
            except Exception as e:
                logger.warning("  %s: SKIPPED (%s)", ticker, str(e)[:60])

        # Rank by total score (descending)
        candidates.sort(key=lambda x: x["total_score"], reverse=True)
        for i, c in enumerate(candidates):
            c["rank"] = i + 1

        selected = candidates[:top_n]
        selected_tickers = [c["ticker"] for c in selected]

        logger.info(
            "StockScreener: selected top %d → %s",
            top_n, selected_tickers,
        )

        return selected

    def _score_ticker(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        news_start: datetime,
    ) -> Optional[dict]:
        """Compute screening scores for a single ticker."""

        # ── Fetch price data ──────────────────────────────────
        df = self._market.get_historical_bars(
            ticker, start=start_date, end=end_date, timeframe="1Day"
        )
        if df is None or len(df) < 20:
            return None

        df = self._market.add_moving_averages(df, windows=[50])

        closes = df["close"].astype(float)
        volumes = df["volume"].astype(float)
        highs = df["high"].astype(float)
        lows = df["low"].astype(float)

        # ── 1. Liquidity Score (0-25) ─────────────────────────
        # Average daily dollar volume, log-scaled
        avg_dollar_vol = (closes * volumes).mean()
        # Score: log10(dollar_vol) mapped to 0-25
        # $10M/day → ~17, $1B/day → ~23, $10B/day → ~25
        if avg_dollar_vol > 0:
            liquidity_raw = math.log10(avg_dollar_vol)
            liquidity_score = min(25, max(0, (liquidity_raw - 6) / 4 * 25))
        else:
            liquidity_score = 0.0

        # ── 2. Trend Score (0-25) ─────────────────────────────
        # How consistently is price above SMA-50?
        if "SMA_50" in df.columns:
            sma = df["SMA_50"].astype(float)
            valid_mask = sma.notna() & (sma > 0)
            if valid_mask.sum() > 0:
                above_sma = (closes[valid_mask] > sma[valid_mask]).mean()
                # Current trend strength: how far above SMA-50
                latest_close = float(closes.iloc[-1])
                latest_sma = float(sma.dropna().iloc[-1]) if sma.notna().any() else latest_close
                trend_pct = (latest_close - latest_sma) / latest_sma if latest_sma > 0 else 0
                # Score: 60% based on % of days above SMA, 40% on current trend strength
                trend_score = min(25, max(0,
                    above_sma * 15 +  # up to 15 pts: how often above SMA
                    min(10, max(0, trend_pct * 100))  # up to 10 pts: current strength
                ))
            else:
                trend_score = 0.0
        else:
            trend_score = 0.0

        # ── 3. Volatility Score (0-25) ────────────────────────
        # Prefer moderate volatility: too low = no opportunities, too high = too risky
        daily_returns = closes.pct_change().dropna()
        if len(daily_returns) > 5:
            annualized_vol = float(daily_returns.std() * math.sqrt(252) * 100)
            # Sweet spot: 20-35% annualized vol
            if 20 <= annualized_vol <= 35:
                volatility_score = 25.0
            elif 15 <= annualized_vol < 20 or 35 < annualized_vol <= 50:
                volatility_score = 18.0
            elif 10 <= annualized_vol < 15 or 50 < annualized_vol <= 70:
                volatility_score = 10.0
            else:
                volatility_score = 5.0
        else:
            volatility_score = 0.0

        # ── 4. News Coverage Score (0-25) ─────────────────────
        # More articles = better FinBERT signal quality
        news_score = 0.0
        if self._news is not None:
            try:
                articles = self._news.get_recent_news(
                    ticker, start=news_start, end=end_date
                )
                n_articles = len(articles) if articles else 0
                # Score: 0 articles = 0, 50 = 10, 200 = 20, 500+ = 25
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

        total_score = liquidity_score + trend_score + volatility_score + news_score

        return {
            "ticker": ticker,
            "liquidity_score": round(liquidity_score, 1),
            "trend_score": round(trend_score, 1),
            "volatility_score": round(volatility_score, 1),
            "news_score": round(news_score, 1),
            "total_score": round(total_score, 1),
            "avg_dollar_volume": round(avg_dollar_vol, 0),
            "annualized_vol": round(annualized_vol, 1) if 'annualized_vol' in dir() else None,
        }


def print_screening_report(results: list[dict]) -> None:
    """Pretty-print the screening results to console."""
    print("\n" + "=" * 80)
    print("  STOCK SCREENING REPORT")
    print("=" * 80)
    print(f"\n  {'Rank':<6} {'Ticker':<8} {'Total':<8} {'Liquidity':<11} "
          f"{'Trend':<8} {'Volatility':<12} {'News':<8}")
    print("  " + "-" * 70)

    for r in results:
        marker = " *" if r.get("rank", 99) <= 4 else "  "  # Mark selected
        print(f"{marker}{r['rank']:<5} {r['ticker']:<8} {r['total_score']:<8.1f} "
              f"{r['liquidity_score']:<11.1f} {r['trend_score']:<8.1f} "
              f"{r['volatility_score']:<12.1f} {r['news_score']:<8.1f}")

    selected = [r["ticker"] for r in results if r.get("rank", 99) <= 4]
    print(f"\n  Selected: {', '.join(selected)}")
    print("=" * 80 + "\n")
