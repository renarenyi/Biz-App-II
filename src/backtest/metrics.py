"""
backtest/metrics.py
---------------------
Computes all performance metrics from a list of closed TradeRecords and an
equity curve.

All metrics are computed from raw lists — no pandas dependency required.

Metrics computed
----------------
  Return metrics:
    total_return          — (final_equity - initial) / initial
    annualised_return     — CAGR over the backtest period
    final_equity          — terminal portfolio value

  Risk metrics:
    max_drawdown          — worst peak-to-trough (negative fraction)
    downside_deviation    — annualised semi-deviation of daily returns
    calmar_ratio          — annualised_return / abs(max_drawdown)

  Risk-adjusted:
    sharpe_ratio          — annualised Sharpe (daily returns, 0% rf)
    profit_factor         — gross_wins / gross_losses

  Trade metrics:
    trade_count
    win_rate              — fraction of winning trades
    avg_win               — mean PnL of winning trades
    avg_loss              — mean PnL of losing trades
    avg_holding_days      — mean days per closed trade
    exposure_pct          — fraction of bars with ≥1 open position

  Benchmark comparison (optional, provided externally):
    benchmark_total_return
    benchmark_sharpe
"""

from __future__ import annotations

import math
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


# =========================================================================== #
# Public API                                                                   #
# =========================================================================== #

def compute_metrics(
    trades:           list[dict],
    equity_curve:     list[dict],   # list of PortfolioSnapshot
    initial_capital:  float,
    bars_per_year:    int   = 252,
    risk_free_rate:   float = 0.0,
    benchmark:        Optional[dict] = None,   # BenchmarkResult
) -> dict:
    """
    Compute all performance metrics and return as a flat dict.

    Parameters
    ----------
    trades         : list of closed TradeRecords
    equity_curve   : list of PortfolioSnapshots (chronological)
    initial_capital: float
    bars_per_year  : int (252 for daily, 52 for weekly, etc.)
    risk_free_rate : float annual (default 0.0)
    benchmark      : BenchmarkResult | None

    Returns
    -------
    dict with all metric keys (None where insufficient data)
    """
    # Deduplicate equity curve by timestamp — the backtester records one
    # snapshot per ticker per bar, so raw len(equity_curve) overcounts.
    # Keep the LAST snapshot for each unique timestamp (final mark-to-market).
    seen_ts: dict[str, dict] = {}
    for s in equity_curve:
        ts_key = str(s.get("timestamp", ""))[:10]  # date-level dedup
        seen_ts[ts_key] = s
    deduped_curve  = list(seen_ts.values())
    n_bars         = len(deduped_curve)
    final_equity   = equity_curve[-1]["equity"] if equity_curve else initial_capital
    equity_values  = [s["equity"] for s in deduped_curve]
    daily_returns  = _daily_returns(equity_values)

    total_ret      = (final_equity - initial_capital) / initial_capital if initial_capital else 0.0
    ann_ret        = _annualised_return(total_ret, n_bars, bars_per_year)
    max_dd         = max_drawdown(equity_values)
    sharpe         = sharpe_ratio(daily_returns, risk_free_rate, bars_per_year)
    downside_dev   = downside_deviation(daily_returns, bars_per_year)
    calmar         = _calmar(ann_ret, max_dd)
    pf             = profit_factor(trades)

    win_trades  = [t for t in trades if (t.get("pnl") or 0) > 0]
    loss_trades = [t for t in trades if (t.get("pnl") or 0) <= 0]
    n_trades    = len(trades)
    win_rt      = len(win_trades) / n_trades if n_trades else 0.0
    avg_win     = _mean([t["pnl"] for t in win_trades if t.get("pnl") is not None]) if win_trades else None
    avg_loss    = _mean([t["pnl"] for t in loss_trades if t.get("pnl") is not None]) if loss_trades else None
    avg_hold    = _mean([t["holding_days"] for t in trades if t.get("holding_days") is not None])
    exposure    = _exposure_pct(equity_curve)

    result = {
        # Return
        "final_equity":       round(final_equity, 2),
        "total_return":       round(total_ret, 6),
        "annualised_return":  round(ann_ret, 6) if ann_ret is not None else None,
        # Risk
        "max_drawdown":       round(max_dd, 6),
        "downside_deviation": round(downside_dev, 6) if downside_dev is not None else None,
        "calmar_ratio":       round(calmar, 4) if calmar is not None else None,
        # Risk-adjusted
        "sharpe_ratio":       round(sharpe, 4) if sharpe is not None else None,
        "profit_factor":      round(pf, 4) if pf is not None else None,
        # Trade
        "trade_count":        n_trades,
        "win_rate":           round(win_rt, 4),
        "avg_win":            round(avg_win, 4) if avg_win is not None else None,
        "avg_loss":           round(avg_loss, 4) if avg_loss is not None else None,
        "avg_holding_days":   round(avg_hold, 2) if avg_hold is not None else None,
        "exposure_pct":       round(exposure, 4),
    }

    if benchmark is not None:
        result["benchmark_total_return"] = benchmark.get("total_return")
        result["benchmark_sharpe"]       = benchmark.get("sharpe_ratio")
        result["benchmark_max_drawdown"] = benchmark.get("max_drawdown")
        result["benchmark_ann_return"]   = benchmark.get("annualised_return")

    return result


# =========================================================================== #
# Individual metric functions (also exported for tests)                        #
# =========================================================================== #

def max_drawdown(equity_values: list[float]) -> float:
    """
    Return max peak-to-trough drawdown as a negative fraction.
    E.g. -0.15 means -15%.
    """
    if not equity_values:
        return 0.0
    peak   = equity_values[0]
    max_dd = 0.0
    for v in equity_values:
        if v > peak:
            peak = v
        dd = (v - peak) / peak if peak > 0 else 0.0
        if dd < max_dd:
            max_dd = dd
    return max_dd


def sharpe_ratio(
    daily_returns:    list[float],
    risk_free_rate:   float = 0.0,
    bars_per_year:    int   = 252,
) -> Optional[float]:
    """
    Annualised Sharpe ratio from daily return series.
    Returns None if < 2 observations or zero volatility.
    """
    if len(daily_returns) < 2:
        return None
    n   = len(daily_returns)
    rf  = risk_free_rate / bars_per_year
    exc = [r - rf for r in daily_returns]
    mu  = sum(exc) / n
    var = sum((r - mu) ** 2 for r in exc) / (n - 1)
    std = math.sqrt(var) if var > 0 else 0.0
    if std == 0:
        return None
    return (mu / std) * math.sqrt(bars_per_year)


def downside_deviation(
    daily_returns: list[float],
    bars_per_year: int = 252,
    threshold: float = 0.0,
) -> Optional[float]:
    """
    Annualised downside deviation (semi-deviation below threshold).
    """
    neg = [r for r in daily_returns if r < threshold]
    if len(neg) < 2:
        return None
    n   = len(neg)
    mu  = sum(neg) / n
    var = sum((r - mu) ** 2 for r in neg) / (n - 1)
    return math.sqrt(var) * math.sqrt(bars_per_year) if var > 0 else 0.0


def profit_factor(trades: list[dict]) -> Optional[float]:
    """
    Gross profits / gross losses.
    Returns None if no losing trades (infinite — reported as None).
    """
    gross_win  = sum(t["pnl"] for t in trades if (t.get("pnl") or 0) > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if (t.get("pnl") or 0) < 0))
    if gross_loss == 0:
        return None   # all winners — infinite profit factor
    return gross_win / gross_loss


def win_rate(trades: list[dict]) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if (t.get("pnl") or 0) > 0)
    return wins / len(trades)


def trade_summary(trades: list[dict]) -> dict:
    """
    Return per-trade summary statistics as a dict.
    """
    n = len(trades)
    if n == 0:
        return {
            "trade_count":  0,
            "win_count":    0,
            "loss_count":   0,
            "win_rate":     0.0,
            "avg_pnl":      None,
            "total_pnl":    0.0,
            "max_win":      None,
            "max_loss":     None,
            "avg_holding":  None,
        }

    pnls     = [t.get("pnl") or 0.0 for t in trades]
    holdings = [t["holding_days"] for t in trades if t.get("holding_days") is not None]

    win_trades  = [p for p in pnls if p > 0]
    loss_trades = [p for p in pnls if p <= 0]

    return {
        "trade_count":  n,
        "win_count":    len(win_trades),
        "loss_count":   len(loss_trades),
        "win_rate":     round(len(win_trades) / n, 4),
        "avg_pnl":      round(sum(pnls) / n, 4),
        "total_pnl":    round(sum(pnls), 4),
        "max_win":      round(max(win_trades), 4) if win_trades else None,
        "max_loss":     round(min(loss_trades), 4) if loss_trades else None,
        "avg_holding":  round(sum(holdings) / len(holdings), 2) if holdings else None,
    }


# =========================================================================== #
# Helpers                                                                      #
# =========================================================================== #

def _daily_returns(equity_values: list[float]) -> list[float]:
    """Compute simple 1-period returns from equity series."""
    returns = []
    for i in range(1, len(equity_values)):
        prev = equity_values[i - 1]
        if prev > 0:
            returns.append(equity_values[i] / prev - 1.0)
    return returns


def _annualised_return(
    total_return: float,
    n_bars:       int,
    bars_per_year: int = 252,
) -> Optional[float]:
    if n_bars <= 0:
        return None
    years = n_bars / bars_per_year
    if years <= 0:
        return None
    try:
        return (1.0 + total_return) ** (1.0 / years) - 1.0
    except (ValueError, ZeroDivisionError):
        return None


def _calmar(ann_return: Optional[float], max_dd: float) -> Optional[float]:
    if ann_return is None:
        return None
    if max_dd == 0:
        return None
    return ann_return / abs(max_dd)


def _mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _exposure_pct(equity_curve: list[dict]) -> float:
    """Fraction of bars where at least one position was open."""
    if not equity_curve:
        return 0.0
    open_bars = sum(1 for s in equity_curve if s.get("open_positions", 0) > 0)
    return open_bars / len(equity_curve)
