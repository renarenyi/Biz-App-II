"""
backtest/benchmark.py
-----------------------
Builds the SPY (or any ticker) buy-and-hold benchmark series.

Convention
----------
  - Buy 1 unit of the benchmark at the first available open price.
  - Hold until the last bar.
  - No transaction costs.
  - Equity at each bar = initial_capital × (close_t / first_close).

This is the canonical, fair benchmark for a long-only strategy.
The same time window as the strategy backtest is used.

Usage
-----
    from src.backtest.benchmark import BenchmarkBuilder
    result = BenchmarkBuilder.build(spy_rows, initial_capital=100_000.0)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from src.backtest.data_alignment import normalise_market_rows
from src.backtest.schemas import BenchmarkResult

logger = logging.getLogger(__name__)


class BenchmarkBuilder:
    """
    Builds a buy-and-hold benchmark from a list of OHLCV bars.
    """

    @staticmethod
    def build(
        market_rows: list[dict],
        initial_capital: float = 100_000.0,
        ticker: str = "SPY",
    ) -> BenchmarkResult:
        """
        Construct a BenchmarkResult from raw market rows.

        Parameters
        ----------
        market_rows     : list[HistoricalMarketRow]  (will be normalised/sorted)
        initial_capital : float
        ticker          : str  (label only, not used for filtering)

        Returns
        -------
        BenchmarkResult

        Notes
        -----
        - Uses `close` prices to compute equity curve.
        - Equity at bar i = initial_capital × (close_i / close_0).
        - Sharpe is computed with daily returns, annualised by √252.
        - Max drawdown is computed over the equity curve.
        """
        rows = normalise_market_rows(market_rows)
        if not rows:
            logger.warning("BenchmarkBuilder: no valid rows for %s", ticker)
            return _empty_result(ticker, initial_capital)

        first_close = float(rows[0].get("close") or 0)
        if first_close <= 0:
            logger.warning("BenchmarkBuilder: first bar has non-positive close for %s", ticker)
            return _empty_result(ticker, initial_capital)

        # Build equity curve
        equity_curve = []
        daily_returns = []
        prev_equity = initial_capital

        for row in rows:
            close  = float(row.get("close") or 0)
            if close <= 0:
                continue
            equity = round(initial_capital * (close / first_close), 4)
            equity_curve.append({
                "timestamp": row["timestamp"],
                "equity":    equity,
                "close":     close,
            })
            if prev_equity > 0:
                daily_returns.append(equity / prev_equity - 1.0)
            prev_equity = equity

        if not equity_curve:
            return _empty_result(ticker, initial_capital)

        final_equity      = equity_curve[-1]["equity"]
        total_return      = (final_equity - initial_capital) / initial_capital
        annualised_return = _annualised_return(total_return, len(equity_curve))
        max_dd            = _max_drawdown([e["equity"] for e in equity_curve])
        sharpe            = _sharpe_ratio(daily_returns)

        start_date = _fmt_date(rows[0]["timestamp"])
        end_date   = _fmt_date(rows[-1]["timestamp"])

        return {
            "ticker":            ticker,
            "start_date":        start_date,
            "end_date":          end_date,
            "initial_capital":   initial_capital,
            "final_equity":      final_equity,
            "total_return":      round(total_return, 6),
            "annualised_return": round(annualised_return, 6) if annualised_return is not None else None,
            "max_drawdown":      round(max_dd, 6),
            "sharpe_ratio":      round(sharpe, 4) if sharpe is not None else None,
            "equity_curve":      equity_curve,
        }


# =========================================================================== #
# Helpers                                                                      #
# =========================================================================== #

def _empty_result(ticker: str, initial_capital: float) -> BenchmarkResult:
    return {
        "ticker":            ticker,
        "start_date":        None,
        "end_date":          None,
        "initial_capital":   initial_capital,
        "final_equity":      initial_capital,
        "total_return":      0.0,
        "annualised_return": None,
        "max_drawdown":      0.0,
        "sharpe_ratio":      None,
        "equity_curve":      [],
    }


def _annualised_return(total_return: float, n_bars: int, bars_per_year: int = 252) -> Optional[float]:
    if n_bars <= 0:
        return None
    years = n_bars / bars_per_year
    if years <= 0:
        return None
    try:
        return (1.0 + total_return) ** (1.0 / years) - 1.0
    except (ValueError, ZeroDivisionError):
        return None


def _max_drawdown(equity_values: list[float]) -> float:
    """Return maximum drawdown (negative number, e.g. -0.15 = -15%)."""
    if not equity_values:
        return 0.0
    peak = equity_values[0]
    max_dd = 0.0
    for v in equity_values:
        if v > peak:
            peak = v
        dd = (v - peak) / peak if peak > 0 else 0.0
        if dd < max_dd:
            max_dd = dd
    return max_dd


def _sharpe_ratio(
    daily_returns: list[float],
    risk_free_rate_annual: float = 0.0,
    bars_per_year: int = 252,
) -> Optional[float]:
    """
    Compute annualised Sharpe ratio from daily return series.
    Returns None if fewer than 2 observations.
    """
    if len(daily_returns) < 2:
        return None
    import math
    n    = len(daily_returns)
    mean = sum(daily_returns) / n
    rf   = risk_free_rate_annual / bars_per_year
    excess = [r - rf for r in daily_returns]
    mean_exc = sum(excess) / n
    var = sum((r - mean_exc) ** 2 for r in excess) / (n - 1)
    std = math.sqrt(var) if var > 0 else 0.0
    if std == 0:
        return None
    return (mean_exc / std) * math.sqrt(bars_per_year)


def _fmt_date(dt) -> Optional[str]:
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%d")
    if isinstance(dt, str):
        return dt[:10]
    return None
