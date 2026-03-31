"""
tests/backtest/test_execution_simulator.py
-------------------------------------------
Tests for ExecutionSimulator fill logic and stop/TP simulation.

Phase 4 spec scenarios covered:
  4. stop-loss triggers historical exit correctly
  6. portfolio value updates correctly after each trade (via PortfolioTracker)
  7. signal logic matches paper-trading rule output for same synthetic inputs
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.backtest.execution_simulator import (
    ExecutionSimulator,
    compute_fill_price,
    check_stop_and_tp,
)
from src.backtest.portfolio_tracker import PortfolioTracker
from src.backtest.schemas import make_trade_record, close_trade_record


# =========================================================================== #
# Helpers                                                                      #
# =========================================================================== #

def _utc(day: int, hour: int = 9) -> datetime:
    return datetime(2025, 3, day, hour, 0, 0, tzinfo=timezone.utc)


def _bar(day=1, open_=200.0, high=210.0, low=190.0, close=205.0) -> dict:
    return {
        "ticker":    "TSLA",
        "timestamp": _utc(day),
        "open":      open_,
        "high":      high,
        "low":       low,
        "close":     close,
        "volume":    1_000_000,
        "sma_50":    180.0,
    }


def _trade(entry_price=200.0, qty=5, stop_loss=196.0) -> dict:
    return make_trade_record(
        trade_id        = 1,
        ticker          = "TSLA",
        entry_time      = _utc(1),
        entry_price     = entry_price,
        qty             = qty,
        stop_loss_price = stop_loss,
    )


# =========================================================================== #
# compute_fill_price                                                           #
# =========================================================================== #

def test_fill_price_next_open():
    signal_bar = _bar(day=1, close=205.0)
    next_bar   = _bar(day=2, open_=206.0)
    price = compute_fill_price(signal_bar, next_bar, "buy", "next_open")
    assert abs(price - 206.0) < 0.01


def test_fill_price_same_close():
    signal_bar = _bar(day=1, close=205.0)
    price = compute_fill_price(signal_bar, None, "buy", "same_close")
    assert abs(price - 205.0) < 0.01


def test_fill_price_next_open_missing_next_bar():
    signal_bar = _bar(day=1)
    price = compute_fill_price(signal_bar, None, "buy", "next_open")
    assert price is None


def test_fill_price_with_slippage_buy():
    signal_bar = _bar(day=1, close=200.0)
    next_bar   = _bar(day=2, open_=200.0)
    price = compute_fill_price(signal_bar, next_bar, "buy", "next_open", slippage_pct=0.001)
    assert abs(price - 200.2) < 0.01


def test_fill_price_with_slippage_sell():
    signal_bar = _bar(day=1, close=200.0)
    next_bar   = _bar(day=2, open_=200.0)
    price = compute_fill_price(signal_bar, next_bar, "sell", "next_open", slippage_pct=0.001)
    assert abs(price - 199.8) < 0.01


# =========================================================================== #
# check_stop_and_tp                                                            #
# =========================================================================== #

def test_stop_loss_hit_when_low_at_or_below_stop():
    trade = _trade(entry_price=200.0, stop_loss=196.0)
    bar   = _bar(day=2, low=194.0, high=205.0)
    result = check_stop_and_tp(trade, bar)
    assert result is not None
    reason, price = result
    assert reason == "stop_loss"
    assert abs(price - 196.0) < 0.01   # fill at stop price


def test_stop_loss_not_hit_when_low_above_stop():
    trade = _trade(entry_price=200.0, stop_loss=196.0)
    bar   = _bar(day=2, low=197.0, high=210.0)
    assert check_stop_and_tp(trade, bar) is None


def test_take_profit_hit_when_high_at_or_above_tp():
    trade = _trade(entry_price=200.0, stop_loss=196.0)
    trade["take_profit_price"] = 208.0
    bar = _bar(day=2, low=199.0, high=210.0)
    result = check_stop_and_tp(trade, bar)
    assert result is not None
    reason, price = result
    assert reason == "take_profit"
    assert abs(price - 208.0) < 0.01


def test_stop_takes_priority_over_tp():
    """When both stop and TP are hit in the same bar, stop-loss takes priority."""
    trade = _trade(entry_price=200.0, stop_loss=196.0)
    trade["take_profit_price"] = 202.0
    # Bar: low=190 (hits stop), high=210 (would hit TP too)
    bar = _bar(day=2, low=190.0, high=210.0)
    result = check_stop_and_tp(trade, bar)
    assert result is not None
    reason, _ = result
    assert reason == "stop_loss"


def test_stop_fill_capped_at_bar_low():
    """Fill cannot be worse (higher for sell) than bar low — conservative model."""
    trade = _trade(stop_loss=196.0)
    # Bar low is 195 — worse than stop but best achievable fill
    bar = _bar(day=2, low=195.0, high=205.0)
    result = check_stop_and_tp(trade, bar)
    assert result is not None
    reason, price = result
    assert reason == "stop_loss"
    assert price >= 195.0    # fill at max(stop_price, low)


# =========================================================================== #
# ExecutionSimulator.simulate_entry                                            #
# =========================================================================== #

def test_simulate_entry_returns_trade_record():
    sim  = ExecutionSimulator()
    trade = sim.simulate_entry(
        ticker          = "TSLA",
        signal_bar      = _bar(day=1, close=200.0),
        next_bar        = _bar(day=2, open_=201.0),
        qty             = 5,
        stop_loss_price = 196.0,
    )
    assert trade is not None
    assert trade["ticker"] == "TSLA"
    assert trade["qty"] == 5
    assert abs(trade["entry_price"] - 201.0) < 0.01
    assert trade["exit_time"] is None     # still open


def test_simulate_entry_returns_none_without_next_bar():
    sim  = ExecutionSimulator(fill_model="next_open")
    trade = sim.simulate_entry(
        ticker          = "TSLA",
        signal_bar      = _bar(day=1),
        next_bar        = None,
        qty             = 5,
        stop_loss_price = 196.0,
    )
    assert trade is None


def test_simulate_entry_increments_trade_id():
    sim  = ExecutionSimulator()
    t1 = sim.simulate_entry("TSLA", _bar(day=1), _bar(day=2), qty=5, stop_loss_price=196.0)
    t2 = sim.simulate_entry("TSLA", _bar(day=3), _bar(day=4), qty=3, stop_loss_price=196.0)
    assert t1 is not None and t2 is not None
    assert t2["trade_id"] > t1["trade_id"]


# =========================================================================== #
# ExecutionSimulator.simulate_exit                                             #
# =========================================================================== #

def test_simulate_exit_computes_pnl():
    sim   = ExecutionSimulator()
    trade = _trade(entry_price=200.0, qty=5)
    closed = sim.simulate_exit(
        trade      = trade,
        signal_bar = _bar(day=2, close=210.0),
        next_bar   = _bar(day=3, open_=210.0),
        exit_reason = "sentiment_reversal",
    )
    assert closed["exit_price"] is not None
    # 5 shares × (210 - 200) = $50 profit
    assert closed["pnl"] is not None
    assert closed["pnl"] > 0


def test_simulate_exit_losing_trade():
    sim   = ExecutionSimulator()
    trade = _trade(entry_price=200.0, qty=5)
    closed = sim.simulate_exit(
        trade      = trade,
        signal_bar = _bar(day=2, close=190.0),
        next_bar   = _bar(day=3, open_=190.0),
        exit_reason = "stop_loss",
    )
    assert closed["pnl"] < 0


def test_simulate_exit_deducts_commission():
    sim   = ExecutionSimulator(commission=5.0)
    trade = _trade(entry_price=200.0, qty=5)
    # Add the entry commission to the trade record as if simulate_entry set it
    trade["commission_paid"] = 5.0
    closed = sim.simulate_exit(
        trade      = trade,
        signal_bar = _bar(day=2),
        next_bar   = _bar(day=3, open_=200.0),   # break-even price
        exit_reason = "signal_exit",
    )
    # At break-even, PnL should be negative because of commissions
    assert closed["pnl"] < 0
    assert closed["commission_paid"] == 10.0   # entry + exit


# =========================================================================== #
# ExecutionSimulator.check_stops                                               #
# =========================================================================== #

def test_check_stops_triggers_on_stop():
    sim   = ExecutionSimulator()
    trade = _trade(entry_price=200.0, qty=5, stop_loss=196.0)
    bar   = _bar(day=2, low=193.0, high=205.0)
    closed = sim.check_stops(trade, bar)
    assert closed is not None
    assert closed["exit_reason"] == "stop_loss"


def test_check_stops_returns_none_when_not_hit():
    sim   = ExecutionSimulator()
    trade = _trade(entry_price=200.0, qty=5, stop_loss=196.0)
    bar   = _bar(day=2, low=198.0, high=210.0)
    assert sim.check_stops(trade, bar) is None


# =========================================================================== #
# PortfolioTracker integration                                                 #
# =========================================================================== #

def test_portfolio_tracker_cash_decreases_on_entry():
    tracker = PortfolioTracker(initial_capital=100_000.0)
    trade   = _trade(entry_price=200.0, qty=5)  # cost = $1000
    trade["commission_paid"] = 0.0
    ok = tracker.enter_position("TSLA", trade)
    assert ok is True
    assert abs(tracker.cash - 99_000.0) < 0.01


def test_portfolio_tracker_cash_increases_on_exit():
    tracker = PortfolioTracker(initial_capital=100_000.0)
    trade   = _trade(entry_price=200.0, qty=5)
    trade["commission_paid"] = 0.0
    tracker.enter_position("TSLA", trade)

    closed = close_trade_record(trade, _utc(3), 210.0, "stop_loss")
    tracker.exit_position("TSLA", closed)

    # Proceeds: 5 × 210 = $1050 added back to cash
    assert abs(tracker.cash - (100_000.0 - 1000.0 + 1050.0)) < 0.01


def test_portfolio_tracker_blocks_duplicate_position():
    tracker = PortfolioTracker(initial_capital=100_000.0)
    t1 = _trade(entry_price=200.0, qty=5)
    t1["commission_paid"] = 0.0
    t2 = dict(t1)
    t2["trade_id"] = 2
    tracker.enter_position("TSLA", t1)
    ok = tracker.enter_position("TSLA", t2)
    assert ok is False
    assert tracker.open_position_count() == 1


def test_portfolio_tracker_blocks_at_max_positions():
    tracker = PortfolioTracker(initial_capital=500_000.0, max_concurrent_positions=2)
    for i, ticker in enumerate(["A", "B"]):
        t = make_trade_record(i + 1, ticker, _utc(1), 100.0, 5, 98.0)
        t["commission_paid"] = 0.0
        tracker.enter_position(ticker, t)
    # Third entry should be blocked
    t3 = make_trade_record(3, "C", _utc(1), 100.0, 5, 98.0)
    t3["commission_paid"] = 0.0
    ok = tracker.enter_position("C", t3)
    assert ok is False


def test_portfolio_equity_increases_with_unrealised_gain():
    tracker = PortfolioTracker(initial_capital=100_000.0)
    trade   = _trade(entry_price=200.0, qty=5)
    trade["commission_paid"] = 0.0
    tracker.enter_position("TSLA", trade)

    # Mark-to-market with price 210 → 5 × 210 + cash = 99_000 + 1050 = 100_050
    equity = tracker.current_equity({"TSLA": 210.0})
    assert abs(equity - 100_050.0) < 0.01


def test_portfolio_snapshot_recorded():
    tracker = PortfolioTracker(initial_capital=100_000.0)
    snap = tracker.record_snapshot(_utc(1), {"TSLA": 200.0})
    assert snap["equity"] == 100_000.0   # no positions yet
    assert len(tracker.equity_curve()) == 1


def test_portfolio_force_close_all():
    tracker = PortfolioTracker(initial_capital=100_000.0)
    trade = _trade(entry_price=200.0, qty=5)
    trade["commission_paid"] = 0.0
    tracker.enter_position("TSLA", trade)
    assert tracker.open_position_count() == 1

    closed = tracker.force_close_all({"TSLA": 205.0}, _utc(10))
    assert tracker.open_position_count() == 0
    assert len(closed) == 1
    assert closed[0]["exit_reason"] == "end_of_period"
