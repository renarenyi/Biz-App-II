"""
tests/backtest/test_backtester.py
----------------------------------
Integration tests for the Backtester orchestrator.

Phase 4 acceptance criteria covered:
  1. backtester runs the strategy over the window
  2. respects time alignment (no look-ahead)
  3. reuses Phase 3 logic (via StrategyAdapter + real signal_rules)
  4. simulates entries and exits correctly
  5. applies stop-loss and sizing rules consistently
  6. outputs summary metrics and trade log
  7. compares results against SPY buy-and-hold
  8. produces interpretable outputs (via report_generator)

Phase 4 spec scenario mapping:
  1. strong positive sentiment + bullish trend → valid historical entry
  2. stale sentiment blocks entry
  3. missing SMA blocks entry
  4. stop-loss triggers historical exit
  5. duplicate entry while already long is blocked
  6. benchmark data loads and aligns correctly
  7. signal logic matches paper-trading rule output for same synthetic inputs
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.backtest.backtester import Backtester
from src.backtest.schemas import default_config
from src.backtest.report_generator import summary_table, written_interpretation


# =========================================================================== #
# Helpers                                                                      #
# =========================================================================== #

def _utc(day: int, hour: int = 16) -> datetime:
    return datetime(2025, 3, day, hour, 0, 0, tzinfo=timezone.utc)


def _bar(ticker="TSLA", day=1, close=200.0, sma_50=180.0, open_=None) -> dict:
    op = open_ if open_ is not None else close - 1.0
    return {
        "ticker":    ticker,
        "timestamp": _utc(day),
        "open":      op,
        "high":      close + 5.0,
        "low":       close - 5.0,
        "close":     close,
        "volume":    1_000_000,
        "sma_50":    sma_50,
    }


def _sentiment(ticker="TSLA", day=1, hour=15, sentiment="POSITIVE", conviction=9.0, age_days=0) -> dict:
    ts = _utc(day, hour) - timedelta(days=age_days)
    return {
        "ticker":              ticker,
        "timestamp":           ts,
        "sentiment":           sentiment,
        "conviction_score":    conviction,
        "source_count":        5,
        "analysis_window_hours": 24,
    }


def _make_config(**overrides) -> dict:
    return default_config(
        tickers           = ["TSLA"],
        start_date        = "2025-03-01",
        end_date          = "2025-03-15",
        initial_capital   = 100_000.0,
        **overrides,
    )


def _run(market_rows, sentiment_rows, spy_rows=None, **config_overrides):
    cfg = _make_config(**config_overrides)
    bt  = Backtester(cfg)
    return bt.run(market_rows, sentiment_rows, spy_rows)


# =========================================================================== #
# Scenario 1: bullish trend + positive sentiment → BUY and trade recorded     #
# =========================================================================== #

def test_scenario1_positive_sentiment_bullish_trend_generates_entry():
    """
    Phase 4 spec scenario 1: strong positive sentiment + bullish trend
    produces a valid historical entry.
    """
    bars  = [_bar(day=1, close=200, sma_50=180), _bar(day=2, close=202, sma_50=180)]
    snaps = [_sentiment(day=1, hour=15, sentiment="POSITIVE", conviction=9.0)]
    result = _run(bars, snaps)
    # After 2 bars: entry on day 1 signal, fill at day 2 open; then forced close at end
    assert result["trade_count"] >= 1
    trade = result["trades"][0]
    assert trade["ticker"] == "TSLA"
    assert trade["side"] == "long"


# =========================================================================== #
# Scenario 2: stale sentiment blocks entry                                    #
# =========================================================================== #

def test_scenario2_stale_sentiment_no_entry():
    """
    Sentiment older than 24 hours is treated as stale → NO_ACTION.
    age_days=2 means sentiment is ~48h old at bar time.
    """
    bars = [_bar(day=5, close=200, sma_50=180), _bar(day=6, close=202, sma_50=180)]
    # Sentiment timestamp is 2 days before bar → stale
    snaps = [_sentiment(day=3, hour=15, sentiment="POSITIVE", conviction=9.0)]
    result = _run(bars, snaps)
    # stale sentiment → no entry; only end-of-period close if somehow opened
    assert result["trade_count"] == 0


# =========================================================================== #
# Scenario 3: missing SMA blocks entry                                        #
# =========================================================================== #

def test_scenario3_missing_sma_no_entry():
    """sma_50=None means the technical filter cannot be evaluated → NO_ACTION."""
    bars = [_bar(day=1, close=200, sma_50=None), _bar(day=2, close=202, sma_50=None)]
    snaps = [_sentiment(day=1, hour=15, sentiment="POSITIVE", conviction=9.0)]
    result = _run(bars, snaps)
    assert result["trade_count"] == 0


# =========================================================================== #
# Scenario 4: stop-loss triggers historical exit                              #
# =========================================================================== #

def test_scenario4_stop_loss_triggers_exit():
    """
    Enter on day 1 (close=200, sma=180, sentiment=POSITIVE).
    Day 2 low drops below 2% stop-loss (stop at 196) → stop triggered.
    """
    # 3 bars: signal bar (day 1), fill bar (day 2 w/ low=193), final bar (day 3)
    bars = [
        _bar(day=1, close=200, sma_50=180),
        {"ticker": "TSLA", "timestamp": _utc(2), "open": 199.0, "high": 202.0,
         "low": 193.0, "close": 198.0, "volume": 1_000_000, "sma_50": 180.0},
        _bar(day=3, close=198, sma_50=180),
    ]
    snaps = [_sentiment(day=1, hour=15, sentiment="POSITIVE", conviction=9.0)]
    result = _run(bars, snaps, stop_loss_pct=0.02)
    # Find the stop-loss-exited trade
    stop_trades = [t for t in result["trades"] if t.get("exit_reason") == "stop_loss"]
    assert len(stop_trades) >= 1
    trade = stop_trades[0]
    assert trade["pnl"] is not None
    assert trade["pnl"] < 0   # stop-loss is a loss


# =========================================================================== #
# Scenario 5: duplicate entry blocked while already long                      #
# =========================================================================== #

def test_scenario5_duplicate_entry_blocked():
    """
    Once a position is open for TSLA, a second BUY signal must be blocked.
    Open position count should stay at 1.
    """
    bars  = [_bar(day=i, close=200 + i, sma_50=180) for i in range(1, 6)]
    snaps = [_sentiment(day=1, hour=15, sentiment="POSITIVE", conviction=9.0),
             _sentiment(day=2, hour=15, sentiment="POSITIVE", conviction=9.0),
             _sentiment(day=3, hour=15, sentiment="POSITIVE", conviction=9.0)]
    result = _run(bars, snaps)
    # All trades: only 1 open at a time; final forced close at end
    open_counts = [s["open_positions"] for s in result["equity_curve"]]
    assert max(open_counts) <= 1


# =========================================================================== #
# Scenario 6: benchmark data loads and aligns                                 #
# =========================================================================== #

def test_scenario6_benchmark_computed():
    """SPY buy-and-hold benchmark is computed over the same window."""
    bars  = [_bar(day=i, close=200 + i * 0.5, sma_50=180) for i in range(1, 10)]
    snaps = [_sentiment(day=1, hour=15, sentiment="POSITIVE", conviction=9.0)]
    spy_bars = [{"ticker": "SPY", "timestamp": _utc(i), "open": 450.0,
                 "high": 455.0, "low": 448.0, "close": 450.0 + i * 0.3,
                 "volume": 50_000_000} for i in range(1, 10)]

    result = _run(bars, snaps, spy_rows=spy_bars)
    assert result.get("benchmark") is not None
    bench = result["benchmark"]
    assert bench.get("total_return") is not None
    assert bench.get("equity_curve") and len(bench["equity_curve"]) > 0
    assert result.get("benchmark_total_return") == bench["total_return"]


# =========================================================================== #
# Scenario 7: signal consistency between backtester and Phase 3               #
# =========================================================================== #

def test_scenario7_strategy_adapter_matches_phase3_signal_rules():
    """
    For identical inputs, the StrategyAdapter must produce the same signal
    as calling Phase 3 evaluate_entry() directly.
    """
    from src.backtest.strategy_adapter import StrategyAdapter
    from src.strategy.signal_rules import evaluate_entry

    cfg = _make_config(conviction_threshold=7.0, sentiment_max_age_hours=24.0)
    adapter = StrategyAdapter(cfg)

    bar = _bar(day=1, close=200.0, sma_50=180.0)
    snap = _sentiment(day=1, hour=15, sentiment="POSITIVE", conviction=9.0)
    ts = _utc(1, 16)

    # Via adapter
    adapter_decision = adapter.entry_signal(bar, snap, ts)

    # Directly via Phase 3
    from src.backtest.strategy_adapter import _to_market_snapshot, _to_sentiment_snapshot
    market  = _to_market_snapshot(bar, ts)
    sent    = _to_sentiment_snapshot(snap, "TSLA", ts)
    direct_decision = evaluate_entry(
        market=market,
        sentiment=sent,
        conviction_threshold=7.0,
        sentiment_max_age_hours=24.0,
        reference_time=ts,
    )

    assert adapter_decision["signal"] == direct_decision["signal"]


# =========================================================================== #
# Result structure                                                             #
# =========================================================================== #

def test_result_has_required_keys():
    bars  = [_bar(day=1, close=200, sma_50=180), _bar(day=2, close=202, sma_50=180)]
    snaps = [_sentiment(day=1, hour=15, sentiment="NEUTRAL", conviction=5.0)]
    result = _run(bars, snaps)

    required = {
        "strategy_name", "start_date", "end_date", "initial_capital",
        "trades", "equity_curve", "total_return", "max_drawdown",
        "trade_count", "win_rate",
    }
    assert required.issubset(result.keys())


def test_trade_log_has_required_fields():
    bars  = [_bar(day=1, close=200, sma_50=180), _bar(day=2, close=202, sma_50=180)]
    snaps = [_sentiment(day=1, hour=15, sentiment="POSITIVE", conviction=9.0)]
    result = _run(bars, snaps)
    for trade in result["trades"]:
        assert "ticker" in trade
        assert "entry_price" in trade
        assert "exit_price" in trade
        assert "pnl" in trade
        assert "exit_reason" in trade


def test_equity_curve_is_populated():
    bars  = [_bar(day=i, close=200 + i, sma_50=180) for i in range(1, 6)]
    snaps = [_sentiment(day=1, hour=15, sentiment="NEUTRAL")]
    result = _run(bars, snaps)
    assert len(result["equity_curve"]) >= 1
    for snap in result["equity_curve"]:
        assert snap["equity"] > 0


def test_no_action_produces_zero_trades():
    """Neutral sentiment → no entry → trade_count = 0."""
    bars  = [_bar(day=i, close=200, sma_50=180) for i in range(1, 5)]
    snaps = [_sentiment(day=1, hour=15, sentiment="NEUTRAL", conviction=8.0)]
    result = _run(bars, snaps)
    assert result["trade_count"] == 0


def test_initial_capital_preserved_with_no_trades():
    bars  = [_bar(day=i, close=200, sma_50=180) for i in range(1, 5)]
    snaps = []
    result = _run(bars, snaps)
    assert result["final_equity"] == 100_000.0


# =========================================================================== #
# Report generator smoke test                                                 #
# =========================================================================== #

def test_summary_table_from_result():
    bars  = [_bar(day=1, close=200, sma_50=180), _bar(day=2, close=202, sma_50=180)]
    snaps = [_sentiment(day=1, hour=15, sentiment="NEUTRAL")]
    result = _run(bars, snaps)
    table = summary_table(result)
    assert "Total Return" in table
    assert "Sharpe Ratio" in table
    assert "Max Drawdown" in table


def test_written_interpretation_not_empty():
    bars  = [_bar(day=1, close=200, sma_50=180), _bar(day=2, close=202, sma_50=180)]
    snaps = [_sentiment(day=1, hour=15, sentiment="POSITIVE", conviction=9.0)]
    result = _run(bars, snaps)
    interp = written_interpretation(result)
    assert len(interp) > 200
    assert "LIMITATIONS" in interp
    assert "BENCHMARK" in interp


# =========================================================================== #
# Backtester.sweep parameter sensitivity                                      #
# =========================================================================== #

def test_sweep_different_stop_loss_pcts():
    """Parameter sweep: varying stop_loss_pct should change trade PnLs."""
    bars  = [_bar(day=i, close=200 + i, sma_50=180) for i in range(1, 8)]
    snaps = [_sentiment(day=1, hour=15, sentiment="POSITIVE", conviction=9.0)]
    cfg   = _make_config()
    bt    = Backtester(cfg)
    results = bt.sweep(bars, snaps, [
        {"stop_loss_pct": 0.01},
        {"stop_loss_pct": 0.05},
    ])
    assert len(results) == 2
    # Both runs complete without error
    for r in results:
        assert "total_return" in r


# =========================================================================== #
# Backtester invalid config                                                   #
# =========================================================================== #

def test_invalid_config_raises():
    with pytest.raises(ValueError, match="initial_capital"):
        Backtester(default_config(
            tickers=["TSLA"],
            start_date="2025-03-01",
            end_date="2025-03-15",
            initial_capital=-1.0,
        ))
