"""
tests/strategy/test_strategy_engine.py
----------------------------------------
Integration tests for StrategyEngine using a dry-run ExecutionEngine.

Tests cover all 8 required scenarios from the spec plus:
  - position state after BUY
  - stop-loss exit closes the position
  - stale sentiment blocks new entry
  - duplicate position blocked
  - missing market data returns NO_ACTION
  - batch evaluation
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.strategy.execution_engine import ExecutionEngine
from src.strategy.order_monitor import OrderMonitor
from src.strategy.logger import StrategyLogger
from src.strategy.strategy_engine import StrategyEngine


# =========================================================================== #
# Helpers                                                                      #
# =========================================================================== #

def _now():
    return datetime.now(tz=timezone.utc)


def _engine(stop_loss_pct=0.02, take_profit_enabled=False, trend_exit=True) -> StrategyEngine:
    """Factory: dry-run StrategyEngine, no file logging."""
    return StrategyEngine(
        execution_engine=ExecutionEngine(
            api_key="", secret_key="",
            base_url="https://paper-api.alpaca.markets",
            dry_run=True,
        ),
        strategy_logger=StrategyLogger(use_file=False),
        stop_loss_pct=stop_loss_pct,
        take_profit_enabled=take_profit_enabled,
        equity_fraction=0.05,
        max_concurrent_positions=5,
        trend_exit_enabled=trend_exit,
        use_file_logging=False,
    )


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


# =========================================================================== #
# Scenario 1: bullish trend + positive sentiment + strong conviction → BUY   #
# =========================================================================== #

def test_scenario1_bullish_positive_conviction_buys():
    eng = _engine()
    result = eng.evaluate("TSLA", _market(close=200, sma_50=180), _sentiment("POSITIVE", 8.5))
    assert result["signal"] == "BUY"
    assert result["order"] is not None
    assert result["order"]["status"] == "submitted"
    assert result["order"]["side"] == "buy"


def test_buy_registers_position():
    eng = _engine()
    eng.evaluate("TSLA", _market(close=200, sma_50=180), _sentiment("POSITIVE", 8.5))
    assert eng.monitor.has_open_position("TSLA")


def test_buy_position_has_stop_loss():
    eng = _engine(stop_loss_pct=0.02)
    eng.evaluate("TSLA", _market(close=200, sma_50=180), _sentiment("POSITIVE", 8.5))
    pos = eng.monitor.get_position("TSLA")
    assert pos is not None
    # Stop should be ~2% below 200
    assert pos["stop_loss_price"] < 200.0
    assert pos["stop_loss_price"] > 190.0


# =========================================================================== #
# Scenario 2: bullish trend + neutral sentiment → HOLD / NO_ACTION           #
# =========================================================================== #

def test_scenario2_neutral_sentiment_no_action():
    eng = _engine()
    result = eng.evaluate("TSLA", _market(), _sentiment("NEUTRAL", 8.5))
    assert result["signal"] in ("NO_ACTION", "HOLD")
    assert result["order"] is None


# =========================================================================== #
# Scenario 3: positive sentiment but low conviction → NO_ACTION               #
# =========================================================================== #

def test_scenario3_low_conviction_no_action():
    eng = _engine()
    result = eng.evaluate("TSLA", _market(), _sentiment("POSITIVE", 3.0))
    assert result["signal"] in ("NO_ACTION", "HOLD")
    assert result["order"] is None


# =========================================================================== #
# Scenario 4: bearish trend + positive sentiment → NO_ACTION                 #
# =========================================================================== #

def test_scenario4_bearish_trend_no_action():
    eng = _engine()
    result = eng.evaluate("TSLA", _market(close=170, sma_50=180), _sentiment("POSITIVE", 9.0))
    assert result["signal"] in ("NO_ACTION", "HOLD")
    assert result["order"] is None


# =========================================================================== #
# Scenario 5: open long + stop-loss breach → EXIT                             #
# =========================================================================== #

def test_scenario5_stop_loss_breach_exits():
    eng = _engine(stop_loss_pct=0.02)
    # Open a position at 200 → stop at 196
    eng.evaluate("TSLA", _market(close=200, sma_50=180), _sentiment("POSITIVE", 9.0))
    assert eng.monitor.has_open_position("TSLA")

    # Price drops below stop
    result = eng.evaluate("TSLA", _market(close=190, sma_50=180), _sentiment("POSITIVE", 9.0))
    assert result["signal"] == "EXIT"
    assert result["order"] is not None
    assert result["order"]["side"] == "sell"
    assert not eng.monitor.has_open_position("TSLA")


# =========================================================================== #
# Scenario 6: missing market data → NO_ACTION                                 #
# =========================================================================== #

def test_scenario6_missing_market_data_no_action():
    eng = _engine()
    result = eng.evaluate("TSLA", {}, _sentiment("POSITIVE", 9.0))
    assert result["signal"] == "NO_ACTION"
    assert result["order"] is None


def test_invalid_close_price_no_action():
    eng = _engine()
    result = eng.evaluate("TSLA", _market(close=0), _sentiment("POSITIVE", 9.0))
    assert result["signal"] in ("NO_ACTION", "HOLD")


# =========================================================================== #
# Scenario 7: stale sentiment → NO_ACTION                                     #
# =========================================================================== #

def test_scenario7_stale_sentiment_no_action():
    eng = _engine()
    result = eng.evaluate("TSLA", _market(), _sentiment("POSITIVE", 9.0, age_hours=25.0))
    assert result["signal"] in ("NO_ACTION", "HOLD")
    assert result["order"] is None


# =========================================================================== #
# Scenario 8: Alpaca order rejected → structured failure                      #
# =========================================================================== #

class _RejectingEngine(ExecutionEngine):
    """Simulates a rejected order."""
    def __init__(self):
        super().__init__(api_key="", secret_key="",
                         base_url="https://paper-api.alpaca.markets", dry_run=True)

    def submit_market_order(self, ticker, side, qty, signal_reason=None):
        return {
            "ticker": ticker, "status": "failed",
            "order_id": None, "submitted_at": _now(),
            "side": side, "qty": qty, "fill_price": None,
            "error": "Alpaca: insufficient funds.",
        }

    def get_account(self):
        return {"buying_power": 100_000.0, "equity": 100_000.0, "cash": 100_000.0}


def test_scenario8_order_rejection_structured_failure():
    eng = StrategyEngine(
        execution_engine=_RejectingEngine(),
        strategy_logger=StrategyLogger(use_file=False),
        use_file_logging=False,
    )
    result = eng.evaluate("TSLA", _market(close=200, sma_50=180), _sentiment("POSITIVE", 9.0))
    # The engine should surface the failure without crashing
    assert result is not None
    if result["order"]:
        assert result["order"]["status"] == "failed"
    # Position should NOT be registered on a failed order
    assert not eng.monitor.has_open_position("TSLA")


# =========================================================================== #
# Duplicate position blocked                                                   #
# =========================================================================== #

def test_duplicate_position_blocked():
    eng = _engine()
    # First BUY
    r1 = eng.evaluate("TSLA", _market(close=200, sma_50=180), _sentiment("POSITIVE", 9.0))
    assert r1["signal"] == "BUY"

    # Second attempt with same conditions — should be blocked
    r2 = eng.evaluate("TSLA", _market(close=202, sma_50=180), _sentiment("POSITIVE", 9.0))
    # When there's already a position, we evaluate exit, not entry
    # Price is still above stop → HOLD
    assert r2["signal"] in ("HOLD", "EXIT")
    assert eng.monitor.open_position_count() == 1


# =========================================================================== #
# Sentiment reversal exits position                                            #
# =========================================================================== #

def test_sentiment_reversal_exits_position():
    eng = _engine()
    eng.evaluate("TSLA", _market(close=200, sma_50=180), _sentiment("POSITIVE", 9.0))
    assert eng.monitor.has_open_position("TSLA")

    # Strong negative sentiment reversal
    result = eng.evaluate(
        "TSLA",
        _market(close=198, sma_50=180),
        _sentiment("NEGATIVE", 8.0),
    )
    assert result["signal"] == "EXIT"
    assert not eng.monitor.has_open_position("TSLA")


# =========================================================================== #
# Batch evaluation                                                             #
# =========================================================================== #

def test_batch_evaluation():
    eng = _engine()
    snapshots = [
        (_market("TSLA", 200, 180), _sentiment("POSITIVE", 9.0)),
        (_market("NVDA", 150, 170), _sentiment("POSITIVE", 8.0)),  # bearish trend
        (_market("AAPL", 180, 160), _sentiment("NEUTRAL", 7.0)),   # neutral sentiment
    ]
    results = eng.evaluate_batch(snapshots)
    assert len(results) == 3
    tickers = [r["ticker"] for r in results]
    assert "TSLA" in tickers
    assert "NVDA" in tickers
    assert "AAPL" in tickers

    # Only TSLA should fire BUY
    tsla = next(r for r in results if r["ticker"] == "TSLA")
    nvda = next(r for r in results if r["ticker"] == "NVDA")
    aapl = next(r for r in results if r["ticker"] == "AAPL")

    assert tsla["signal"] == "BUY"
    assert nvda["signal"] in ("NO_ACTION", "HOLD")
    assert aapl["signal"] in ("NO_ACTION", "HOLD")


# =========================================================================== #
# Risk manager: stop-loss computation                                          #
# =========================================================================== #

def test_stop_loss_computed_correctly():
    from src.strategy.risk_manager import compute_stop_loss
    stop = compute_stop_loss(200.0, 0.02, "long")
    assert abs(stop - 196.0) < 0.01


def test_take_profit_disabled_returns_none():
    from src.strategy.risk_manager import compute_take_profit
    tp = compute_take_profit(200.0, 0.04, "long", enabled=False)
    assert tp is None


def test_assess_position_stop_hit():
    from src.strategy.risk_manager import assess_position
    pos = {"ticker": "TSLA", "stop_loss_price": 196.0, "side": "long"}
    result = assess_position(pos, current_price=195.5)
    assert result["stop_hit"] is True
    assert result["action"] == "EXIT"


def test_assess_position_within_bounds():
    from src.strategy.risk_manager import assess_position
    pos = {"ticker": "TSLA", "stop_loss_price": 196.0, "side": "long"}
    result = assess_position(pos, current_price=200.0)
    assert result["passed"] is True
    assert result["action"] is None


# =========================================================================== #
# Position sizer                                                               #
# =========================================================================== #

def test_position_sizer_fixed_dollar():
    from src.strategy.position_sizer import fixed_dollar_size
    qty = fixed_dollar_size(price=200.0, dollar_amount=1000.0)
    assert qty == 5


def test_position_sizer_pct_equity():
    from src.strategy.position_sizer import percent_of_equity
    qty = percent_of_equity(price=200.0, equity=20_000.0, fraction=0.05)
    assert qty == 5  # 5% of 20000 = 1000 / 200 = 5


def test_position_sizer_zero_price_returns_zero():
    from src.strategy.position_sizer import fixed_dollar_size
    assert fixed_dollar_size(price=0, dollar_amount=1000) == 0


def test_position_sizer_small_equity_returns_zero():
    from src.strategy.position_sizer import percent_of_equity
    qty = percent_of_equity(price=1000.0, equity=100.0, fraction=0.05)
    assert qty == 0  # 5% of 100 = $5; can't buy 1 share at $1000
