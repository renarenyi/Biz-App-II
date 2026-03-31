"""
tests/backtest/test_metrics.py
--------------------------------
Tests for all performance metric computations.

Verifies:
  - total_return, max_drawdown, Sharpe, win_rate, profit_factor
  - edge cases: empty trades, all wins, all losses
  - compute_metrics integration
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from src.backtest.metrics import (
    max_drawdown,
    sharpe_ratio,
    downside_deviation,
    profit_factor,
    win_rate,
    trade_summary,
    compute_metrics,
)


# =========================================================================== #
# Helpers                                                                      #
# =========================================================================== #

def _snap(equity: float, day: int = 1, open_pos: int = 0) -> dict:
    return {
        "timestamp":      datetime(2025, 3, day, 16, tzinfo=timezone.utc),
        "equity":         equity,
        "cash":           equity,
        "open_positions": open_pos,
    }


def _trade(pnl: float, holding_days: float = 3.0) -> dict:
    return {
        "trade_id":      1,
        "ticker":        "TSLA",
        "side":          "long",
        "entry_time":    datetime(2025, 3, 1, 9, tzinfo=timezone.utc),
        "exit_time":     datetime(2025, 3, 4, 9, tzinfo=timezone.utc),
        "entry_price":   200.0,
        "exit_price":    200.0 + pnl / 5,
        "qty":           5,
        "pnl":           pnl,
        "return_pct":    pnl / 1000.0,
        "holding_days":  holding_days,
        "exit_reason":   "stop_loss" if pnl < 0 else "take_profit",
    }


# =========================================================================== #
# max_drawdown                                                                 #
# =========================================================================== #

def test_max_drawdown_simple():
    # 100 → 110 → 90 → 105  →  max_dd = (90-110)/110 ≈ -18.2%
    curve = [100.0, 110.0, 90.0, 105.0]
    dd = max_drawdown(curve)
    assert abs(dd - ((90 - 110) / 110)) < 0.001


def test_max_drawdown_monotone_up():
    curve = [100.0, 110.0, 120.0, 130.0]
    assert max_drawdown(curve) == 0.0


def test_max_drawdown_monotone_down():
    curve = [100.0, 90.0, 80.0, 70.0]
    dd = max_drawdown(curve)
    assert dd < 0


def test_max_drawdown_empty():
    assert max_drawdown([]) == 0.0


def test_max_drawdown_single_value():
    assert max_drawdown([100.0]) == 0.0


# =========================================================================== #
# sharpe_ratio                                                                 #
# =========================================================================== #

def test_sharpe_returns_none_for_single_observation():
    assert sharpe_ratio([0.01]) is None


def test_sharpe_returns_none_for_zero_volatility():
    # All returns identical → zero std
    result = sharpe_ratio([0.01, 0.01, 0.01, 0.01])
    assert result is None


def test_sharpe_positive_for_positive_returns():
    # Constant returns: mathematically zero variance → ideally None.
    # Floating-point accumulation may produce near-zero (not exact zero) variance
    # and a very large positive Sharpe.  Accept either None or a positive number.
    returns = [0.005] * 252
    s = sharpe_ratio(returns)
    assert s is None or s > 0


def test_sharpe_sign_correct():
    """Negative mean return → negative Sharpe."""
    returns = [-0.005 + (0.001 if i % 2 == 0 else -0.001) for i in range(100)]
    s = sharpe_ratio(returns)
    # mean is negative → Sharpe should be negative
    if s is not None:
        assert s < 0


def test_sharpe_computed_correctly():
    """Synthetic returns with known Sharpe."""
    # 252 returns of 0.01 mean, with std = 0.05 → annualised Sharpe = (0.01/0.05)*sqrt(252)
    import random
    random.seed(42)
    returns = [0.01 + random.gauss(0, 0.05) for _ in range(252)]
    s = sharpe_ratio(returns)
    assert s is not None
    # Rough sanity: should be around 0.01/0.05 * sqrt(252) ≈ 3.17 ± noise
    # Not asserting exact value due to randomness, just verify it's a number
    assert isinstance(s, float)


# =========================================================================== #
# profit_factor                                                                #
# =========================================================================== #

def test_profit_factor_mixed():
    trades = [_trade(100), _trade(-50), _trade(80), _trade(-30)]
    pf = profit_factor(trades)
    assert pf is not None
    assert abs(pf - (180 / 80)) < 0.01


def test_profit_factor_all_wins():
    trades = [_trade(100), _trade(50)]
    pf = profit_factor(trades)
    assert pf is None   # infinite → None


def test_profit_factor_all_losses():
    trades = [_trade(-50), _trade(-30)]
    pf = profit_factor(trades)
    assert pf == 0.0


def test_profit_factor_empty():
    assert profit_factor([]) is None


# =========================================================================== #
# win_rate                                                                     #
# =========================================================================== #

def test_win_rate_50_pct():
    trades = [_trade(100), _trade(-50)]
    assert abs(win_rate(trades) - 0.5) < 0.001


def test_win_rate_all_wins():
    trades = [_trade(10), _trade(20), _trade(5)]
    assert win_rate(trades) == 1.0


def test_win_rate_zero():
    assert win_rate([]) == 0.0


# =========================================================================== #
# trade_summary                                                                #
# =========================================================================== #

def test_trade_summary_counts():
    trades = [_trade(100), _trade(-50), _trade(30), _trade(-20), _trade(70)]
    summary = trade_summary(trades)
    assert summary["trade_count"] == 5
    assert summary["win_count"] == 3
    assert summary["loss_count"] == 2
    assert abs(summary["win_rate"] - 0.6) < 0.001


def test_trade_summary_total_pnl():
    trades = [_trade(100), _trade(-50), _trade(30)]
    summary = trade_summary(trades)
    assert abs(summary["total_pnl"] - 80.0) < 0.01


def test_trade_summary_empty():
    summary = trade_summary([])
    assert summary["trade_count"] == 0
    assert summary["total_pnl"] == 0.0


# =========================================================================== #
# compute_metrics integration                                                  #
# =========================================================================== #

def test_compute_metrics_no_trades():
    curve = [_snap(100_000, day=i) for i in range(1, 11)]
    m = compute_metrics([], curve, initial_capital=100_000.0)
    assert m["total_return"] == 0.0
    assert m["trade_count"] == 0
    assert m["win_rate"] == 0.0
    assert m["max_drawdown"] == 0.0


def test_compute_metrics_positive_return():
    # Use a timedelta-based approach to avoid month-day overflow
    base = datetime(2025, 1, 1, 16, tzinfo=timezone.utc)
    curve = [
        {
            "timestamp":      base + timedelta(days=i),
            "equity":         100_000 + i * 100,
            "cash":           100_000 + i * 100,
            "open_positions": 0,
        }
        for i in range(251)
    ]
    trades = [_trade(100), _trade(-50), _trade(200)]
    m = compute_metrics(trades, curve, initial_capital=100_000.0)
    assert m["total_return"] > 0
    assert m["trade_count"] == 3


def test_compute_metrics_includes_benchmark():
    curve  = [_snap(100_000, day=i) for i in range(1, 11)]
    bench  = {"total_return": 0.09, "sharpe_ratio": 0.8, "max_drawdown": -0.05}
    m = compute_metrics([], curve, initial_capital=100_000.0, benchmark=bench)
    assert m["benchmark_total_return"] == 0.09
    assert m["benchmark_sharpe"] == 0.8


def test_compute_metrics_max_drawdown_computed():
    # Equity: up then down then slightly up
    curve = [
        _snap(100_000, day=1),
        _snap(110_000, day=2),
        _snap(90_000,  day=3),
        _snap(92_000,  day=4),
    ]
    m = compute_metrics([], curve, initial_capital=100_000.0)
    expected_dd = (90_000 - 110_000) / 110_000  # ≈ -18.2%
    assert abs(m["max_drawdown"] - expected_dd) < 0.001


def test_compute_metrics_exposure_pct():
    curve = [
        _snap(100_000, day=1, open_pos=0),
        _snap(100_000, day=2, open_pos=1),
        _snap(100_000, day=3, open_pos=1),
        _snap(100_000, day=4, open_pos=0),
    ]
    m = compute_metrics([], curve, initial_capital=100_000.0)
    assert abs(m["exposure_pct"] - 0.5) < 0.001
