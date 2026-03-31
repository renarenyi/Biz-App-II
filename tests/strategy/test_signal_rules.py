"""
tests/strategy/test_signal_rules.py
-------------------------------------
Tests for signal_rules.py — deterministic entry/exit/hold logic.

Covers all 8 required scenario types from the spec:
  1. bullish trend + positive sentiment + strong conviction → BUY
  2. bullish trend + neutral sentiment → NO_ACTION
  3. bullish trend + positive sentiment + low conviction → NO_ACTION
  4. bearish trend + positive sentiment → NO_ACTION
  5. open long position + stop-loss breach → EXIT
  6. missing market data → NO_ACTION
  7. stale sentiment → NO_ACTION
  8. [Execution rejection — tested in test_strategy_engine.py]
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.strategy.signal_rules import (
    evaluate_entry,
    evaluate_exit,
    DEFAULT_CONVICTION_THRESHOLD,
    DEFAULT_SENTIMENT_MAX_AGE_HOURS,
)


# =========================================================================== #
# Helpers                                                                      #
# =========================================================================== #

def _now():
    return datetime.now(tz=timezone.utc)


def _market(
    ticker="TSLA",
    close=200.0,
    sma_50=180.0,
    is_market_open=True,
) -> dict:
    return {
        "ticker": ticker,
        "timestamp": _now(),
        "close": close,
        "sma_50": sma_50,
        "is_market_open": is_market_open,
    }


def _sentiment(
    sentiment="POSITIVE",
    conviction=8.5,
    age_hours=1.0,
) -> dict:
    return {
        "ticker": "TSLA",
        "sentiment": sentiment,
        "conviction_score": conviction,
        "generated_at": _now() - timedelta(hours=age_hours),
        "source_count": 5,
        "analysis_window_hours": 24,
    }


def _position(
    entry_price=190.0,
    stop_loss_price=186.2,  # 2% below 190
) -> dict:
    return {
        "ticker": "TSLA",
        "side": "long",
        "qty": 5,
        "entry_price": entry_price,
        "stop_loss_price": stop_loss_price,
        "status": "open",
    }


# =========================================================================== #
# Entry rule tests                                                             #
# =========================================================================== #

# Scenario 1: all conditions pass → BUY
def test_entry_all_conditions_pass_returns_buy():
    decision = evaluate_entry(_market(close=200, sma_50=180), _sentiment("POSITIVE", 8.5))
    assert decision["signal"] == "BUY"
    assert decision["technical_pass"] is True
    assert decision["sentiment_pass"] is True


# Scenario 2: neutral sentiment → NO_ACTION
def test_entry_neutral_sentiment_returns_no_action():
    decision = evaluate_entry(_market(), _sentiment("NEUTRAL", 8.5))
    assert decision["signal"] == "NO_ACTION"
    assert decision["sentiment_pass"] is False


# Scenario 3: low conviction → NO_ACTION
def test_entry_low_conviction_returns_no_action():
    decision = evaluate_entry(
        _market(),
        _sentiment("POSITIVE", 3.0),
        conviction_threshold=DEFAULT_CONVICTION_THRESHOLD,
    )
    assert decision["signal"] == "NO_ACTION"
    assert "conviction" in decision["reason"].lower()


# Scenario 4: bearish trend (price < sma_50) → NO_ACTION
def test_entry_price_below_sma50_returns_no_action():
    decision = evaluate_entry(_market(close=170, sma_50=180), _sentiment("POSITIVE", 9.0))
    assert decision["signal"] == "NO_ACTION"
    assert decision["technical_pass"] is False


# Market closed → NO_ACTION
def test_entry_market_closed_returns_no_action():
    decision = evaluate_entry(_market(is_market_open=False), _sentiment())
    assert decision["signal"] == "NO_ACTION"
    assert "closed" in decision["reason"].lower()


# Missing SMA-50 → NO_ACTION
def test_entry_missing_sma50_returns_no_action():
    market = _market()
    market["sma_50"] = None
    decision = evaluate_entry(market, _sentiment())
    assert decision["signal"] == "NO_ACTION"
    assert "sma" in decision["reason"].lower()


# Scenario 6: missing close price → NO_ACTION
def test_entry_missing_close_returns_no_action():
    market = _market()
    market["close"] = None
    decision = evaluate_entry(market, _sentiment())
    assert decision["signal"] == "NO_ACTION"


# Scenario 7: stale sentiment → NO_ACTION
def test_entry_stale_sentiment_returns_no_action():
    decision = evaluate_entry(
        _market(),
        _sentiment(age_hours=25.0),
        sentiment_max_age_hours=24.0,
    )
    assert decision["signal"] == "NO_ACTION"
    assert "stale" in decision["reason"].lower()


# Negative sentiment → NO_ACTION (not POSITIVE)
def test_entry_negative_sentiment_returns_no_action():
    decision = evaluate_entry(_market(), _sentiment("NEGATIVE", 9.0))
    assert decision["signal"] == "NO_ACTION"


# Exactly at conviction threshold (not strictly above) → NO_ACTION
def test_entry_conviction_at_threshold_not_above_returns_no_action():
    decision = evaluate_entry(
        _market(),
        _sentiment("POSITIVE", DEFAULT_CONVICTION_THRESHOLD),
        conviction_threshold=DEFAULT_CONVICTION_THRESHOLD,
    )
    assert decision["signal"] == "NO_ACTION"


# Just above threshold → BUY
def test_entry_conviction_just_above_threshold_returns_buy():
    decision = evaluate_entry(
        _market(),
        _sentiment("POSITIVE", DEFAULT_CONVICTION_THRESHOLD + 0.01),
        conviction_threshold=DEFAULT_CONVICTION_THRESHOLD,
    )
    assert decision["signal"] == "BUY"


# =========================================================================== #
# Exit rule tests                                                              #
# =========================================================================== #

# Scenario 5: stop-loss hit → EXIT
def test_exit_stop_loss_hit_returns_exit():
    pos = _position(entry_price=190, stop_loss_price=186.2)
    # Price drops to exactly stop_loss
    decision = evaluate_exit(_market(close=186.0), _sentiment(), pos)
    assert decision["signal"] == "EXIT"
    assert "stop" in decision["reason"].lower()


def test_exit_stop_loss_exactly_at_stop_returns_exit():
    pos = _position(stop_loss_price=186.2)
    decision = evaluate_exit(_market(close=186.2), _sentiment(), pos)
    assert decision["signal"] == "EXIT"


# Price above stop → HOLD
def test_exit_price_above_stop_returns_hold():
    pos = _position(stop_loss_price=186.2)
    decision = evaluate_exit(_market(close=195.0), _sentiment(), pos)
    assert decision["signal"] == "HOLD"


# Strongly negative sentiment → EXIT
def test_exit_strong_negative_sentiment_returns_exit():
    pos = _position()
    decision = evaluate_exit(
        _market(close=195.0),
        _sentiment("NEGATIVE", 8.0),
        pos,
        neg_conviction_threshold=6.0,
    )
    assert decision["signal"] == "EXIT"
    assert "sentiment" in decision["reason"].lower()


# Weak negative sentiment → HOLD
def test_exit_weak_negative_sentiment_holds():
    pos = _position()
    decision = evaluate_exit(
        _market(close=195.0),
        _sentiment("NEGATIVE", 3.0),
        pos,
        neg_conviction_threshold=6.0,
    )
    assert decision["signal"] == "HOLD"


# Trend failure (close < sma_50) → EXIT when enabled
def test_exit_trend_failure_returns_exit():
    pos = _position(stop_loss_price=160.0)  # stop far away
    decision = evaluate_exit(
        _market(close=170.0, sma_50=180.0),
        _sentiment("POSITIVE", 7.0),
        pos,
        trend_exit_enabled=True,
    )
    assert decision["signal"] == "EXIT"
    assert "trend" in decision["reason"].lower()


# Trend failure disabled → HOLD
def test_exit_trend_failure_disabled_holds():
    pos = _position(stop_loss_price=160.0)
    decision = evaluate_exit(
        _market(close=170.0, sma_50=180.0),
        _sentiment("POSITIVE", 7.0),
        pos,
        trend_exit_enabled=False,
    )
    assert decision["signal"] == "HOLD"


# =========================================================================== #
# Signal decision schema completeness                                          #
# =========================================================================== #

def test_buy_decision_has_required_fields():
    d = evaluate_entry(_market(), _sentiment())
    # Signal may or may not be BUY depending on market setup
    required = {"ticker", "timestamp", "signal", "reason"}
    assert required.issubset(d.keys())


def test_exit_decision_has_required_fields():
    d = evaluate_exit(_market(close=180.0), _sentiment(), _position(stop_loss_price=185.0))
    required = {"ticker", "timestamp", "signal", "reason"}
    assert required.issubset(d.keys())
