"""
tests/strategy/test_eligibility.py
------------------------------------
Tests for eligibility.py — pre-order eligibility checks.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.strategy.eligibility import (
    check_all_eligibility,
    check_market_open,
    check_no_duplicate_position,
    check_decision_freshness,
    check_valid_quantity,
    check_buying_power,
    check_max_positions,
)


def _now():
    return datetime.now(tz=timezone.utc)


def _open_pos(ticker="TSLA"):
    return {ticker: {"ticker": ticker, "status": "open", "qty": 5, "entry_price": 190.0}}


# ── Market open ──────────────────────────────────────────────────────────────

def test_market_open_passes():
    ok, _ = check_market_open(True)
    assert ok is True


def test_market_closed_fails():
    ok, reason = check_market_open(False)
    assert ok is False
    assert "closed" in reason.lower()


# ── Duplicate position ───────────────────────────────────────────────────────

def test_no_duplicate_position_passes():
    ok, _ = check_no_duplicate_position("TSLA", {})
    assert ok is True


def test_existing_open_position_blocked():
    ok, reason = check_no_duplicate_position("TSLA", _open_pos("TSLA"))
    assert ok is False
    assert "duplicate" in reason.lower()


def test_closed_position_not_blocked():
    ok, _ = check_no_duplicate_position("TSLA", {"TSLA": {"status": "closed"}})
    assert ok is True


# ── Decision freshness ───────────────────────────────────────────────────────

def test_fresh_signal_passes():
    ok, _ = check_decision_freshness(_now() - timedelta(seconds=5), 60.0, _now())
    assert ok is True


def test_stale_signal_fails():
    ok, reason = check_decision_freshness(_now() - timedelta(seconds=120), 60.0, _now())
    assert ok is False
    assert "stale" in reason.lower()


def test_none_timestamp_passes():
    ok, _ = check_decision_freshness(None, 60.0, _now())
    assert ok is True


# ── Valid quantity ───────────────────────────────────────────────────────────

def test_valid_quantity_passes():
    ok, _ = check_valid_quantity(5)
    assert ok is True


def test_zero_quantity_fails():
    ok, reason = check_valid_quantity(0)
    assert ok is False
    assert "invalid" in reason.lower()


def test_negative_quantity_fails():
    ok, _ = check_valid_quantity(-1)
    assert ok is False


# ── Buying power ─────────────────────────────────────────────────────────────

def test_sufficient_buying_power_passes():
    ok, _ = check_buying_power(qty=5, price=100.0, buying_power=1000.0)
    assert ok is True


def test_insufficient_buying_power_fails():
    ok, reason = check_buying_power(qty=5, price=100.0, buying_power=400.0)
    assert ok is False
    assert "insufficient" in reason.lower()


# ── Max positions ─────────────────────────────────────────────────────────────

def test_below_max_positions_passes():
    ok, _ = check_max_positions({"TSLA": {"status": "open"}}, max_positions=5)
    assert ok is True


def test_at_max_positions_fails():
    positions = {f"T{i}": {"status": "open"} for i in range(5)}
    ok, reason = check_max_positions(positions, max_positions=5)
    assert ok is False
    assert "max" in reason.lower()


# ── Composite check ───────────────────────────────────────────────────────────

def test_all_pass_returns_passed():
    result = check_all_eligibility(
        ticker="TSLA",
        signal="BUY",
        qty=5,
        price=200.0,
        is_market_open=True,
        open_positions={},
        buying_power=10_000.0,
        signal_timestamp=_now(),
        max_concurrent_positions=5,
    )
    assert result["passed"] is True


def test_market_closed_blocks_composite():
    result = check_all_eligibility(
        ticker="TSLA",
        signal="BUY",
        qty=5,
        price=200.0,
        is_market_open=False,
        open_positions={},
        buying_power=10_000.0,
    )
    assert result["passed"] is False
    assert result["checks"].get("market_open") is False


def test_duplicate_position_blocks_composite():
    result = check_all_eligibility(
        ticker="TSLA",
        signal="BUY",
        qty=5,
        price=200.0,
        is_market_open=True,
        open_positions=_open_pos("TSLA"),
        buying_power=10_000.0,
    )
    assert result["passed"] is False
    assert result["checks"].get("no_duplicate_position") is False


def test_hold_signal_always_passes_eligibility():
    result = check_all_eligibility(
        ticker="TSLA",
        signal="HOLD",
        qty=0,
        price=0.0,
        is_market_open=False,
        open_positions={},
        buying_power=0.0,
    )
    assert result["passed"] is True


def test_no_action_signal_always_passes_eligibility():
    result = check_all_eligibility(
        ticker="TSLA",
        signal="NO_ACTION",
        qty=0,
        price=0.0,
        is_market_open=False,
        open_positions={},
        buying_power=0.0,
    )
    assert result["passed"] is True
