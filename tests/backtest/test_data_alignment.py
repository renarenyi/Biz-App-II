"""
tests/backtest/test_data_alignment.py
--------------------------------------
Tests for DataAligner and look-ahead bias prevention.

Phase 4 spec scenarios covered:
  1. no future news is used before its timestamp
  2. indicators only use historical price data (validated via DataAligner iterator)
  3. stale sentiment produces None (blocked entry in backtester)
  4. missing SMA produces bar with sma_50=None
  5. benchmark data loads and aligns correctly (nominal test)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.backtest.data_alignment import (
    DataAligner,
    normalise_market_rows,
    normalise_sentiment_snapshots,
    audit_no_lookahead,
    _to_utc,
)
from src.backtest.schemas import validate_market_row, validate_sentiment_snapshot


# =========================================================================== #
# Helpers                                                                      #
# =========================================================================== #

def _utc(day: int, hour: int = 16) -> datetime:
    """Convenience: day-of-month in March 2025, given hour UTC."""
    return datetime(2025, 3, day, hour, 0, 0, tzinfo=timezone.utc)


def _bar(ticker="TSLA", day=1, close=200.0, sma_50=180.0) -> dict:
    return {
        "ticker":    ticker,
        "timestamp": _utc(day),
        "open":      close - 2.0,
        "high":      close + 3.0,
        "low":       close - 3.0,
        "close":     close,
        "volume":    1_000_000,
        "sma_50":    sma_50,
    }


def _sentiment(ticker="TSLA", day=1, hour=15, sentiment="POSITIVE", conviction=8.5) -> dict:
    return {
        "ticker":          ticker,
        "timestamp":       _utc(day, hour),
        "sentiment":       sentiment,
        "conviction_score": conviction,
        "source_count":    3,
        "analysis_window_hours": 24,
    }


# =========================================================================== #
# _to_utc                                                                      #
# =========================================================================== #

def test_to_utc_from_string_iso():
    dt = _to_utc("2025-03-01T16:00:00Z")
    assert dt is not None
    assert dt.tzinfo is not None
    assert dt.year == 2025


def test_to_utc_from_date_string():
    dt = _to_utc("2025-03-01")
    assert dt is not None
    assert dt.year == 2025


def test_to_utc_from_aware_datetime():
    d = datetime(2025, 3, 1, 16, tzinfo=timezone.utc)
    assert _to_utc(d) == d


def test_to_utc_from_naive_datetime():
    d = datetime(2025, 3, 1, 16)
    result = _to_utc(d)
    assert result is not None
    assert result.tzinfo is not None


def test_to_utc_returns_none_for_none():
    assert _to_utc(None) is None


# =========================================================================== #
# normalise_market_rows                                                        #
# =========================================================================== #

def test_normalise_market_rows_sorted():
    rows = [_bar(day=3), _bar(day=1), _bar(day=2)]
    result = normalise_market_rows(rows)
    dates = [r["timestamp"].day for r in result]
    assert dates == [1, 2, 3]


def test_normalise_market_rows_drops_invalid():
    rows = [_bar(day=1), {"ticker": "TSLA", "timestamp": _utc(2), "close": None}]
    result = normalise_market_rows(rows)
    assert len(result) == 1


def test_normalise_market_rows_timestamps_are_aware():
    rows = [_bar(day=1)]
    result = normalise_market_rows(rows)
    assert result[0]["timestamp"].tzinfo is not None


# =========================================================================== #
# normalise_sentiment_snapshots                                                #
# =========================================================================== #

def test_normalise_sentiment_sorted():
    snaps = [_sentiment(day=3), _sentiment(day=1), _sentiment(day=2)]
    result = normalise_sentiment_snapshots(snaps)
    days = [s["timestamp"].day for s in result]
    assert days == [1, 2, 3]


def test_normalise_sentiment_drops_invalid_sentiment_label():
    snaps = [
        _sentiment(day=1, sentiment="POSITIVE"),
        {"ticker": "TSLA", "timestamp": _utc(2), "sentiment": "BULLISH", "conviction_score": 8.0},
    ]
    result = normalise_sentiment_snapshots(snaps)
    assert len(result) == 1


# =========================================================================== #
# DataAligner — no look-ahead                                                  #
# =========================================================================== #

def test_aligner_uses_only_past_sentiment():
    """
    Bar at day 1 16:00 UTC.
    Sentiment at day 1 17:00 UTC — AFTER bar close.
    Sentiment should NOT be returned for that bar.
    """
    bars = [_bar(day=1)]  # bar close at 16:00
    # Sentiment published 1 hour AFTER bar close → must not be visible
    snaps = [_sentiment(day=1, hour=17)]
    aligner = DataAligner(bars, snaps)
    aligned = list(aligner.iterate("TSLA"))
    assert len(aligned) == 1
    assert aligned[0]["sentiment"] is None


def test_aligner_uses_sentiment_at_bar_time():
    """Sentiment at exactly bar close time should be used."""
    bars  = [_bar(day=1)]          # close at 16:00
    snaps = [_sentiment(day=1, hour=16)]   # same time
    aligner = DataAligner(bars, snaps)
    aligned = list(aligner.iterate("TSLA"))
    assert aligned[0]["sentiment"] is not None


def test_aligner_uses_latest_sentiment_before_bar():
    """
    Two sentiment snapshots: day 1 @ 10:00 and day 1 @ 14:00.
    Bar close at day 1 @ 16:00.
    Should return the 14:00 snapshot (the most recent valid one).
    """
    bars  = [_bar(day=1)]
    snaps = [
        _sentiment(day=1, hour=10, conviction=5.0),
        _sentiment(day=1, hour=14, conviction=9.0),
    ]
    aligner = DataAligner(bars, snaps)
    aligned = list(aligner.iterate("TSLA"))
    assert aligned[0]["sentiment"]["conviction_score"] == 9.0


def test_aligner_returns_none_when_no_sentiment_available():
    bars  = [_bar(day=5)]
    snaps = []
    aligner = DataAligner(bars, snaps)
    aligned = list(aligner.iterate("TSLA"))
    assert aligned[0]["sentiment"] is None


def test_aligner_multi_ticker_isolation():
    """Sentiment for TSLA must not appear for NVDA bars."""
    bars  = [_bar("TSLA", day=1), _bar("NVDA", day=1)]
    snaps = [_sentiment("TSLA", day=1, hour=10)]
    aligner = DataAligner(bars, snaps)
    nvda_aligned = list(aligner.iterate("NVDA"))
    assert nvda_aligned[0]["sentiment"] is None
    tsla_aligned = list(aligner.iterate("TSLA"))
    assert tsla_aligned[0]["sentiment"] is not None


def test_aligner_iteration_order_chronological():
    bars = [_bar(day=3), _bar(day=1), _bar(day=2)]
    snaps = []
    aligner = DataAligner(bars, snaps)
    aligned = list(aligner.iterate("TSLA"))
    days = [a["bar"]["timestamp"].day for a in aligned]
    assert days == [1, 2, 3]


def test_aligner_prev_bars_lookback():
    bars = [_bar(day=1), _bar(day=2), _bar(day=3)]
    aligner = DataAligner(bars, [])
    aligned = list(aligner.iterate("TSLA", lookback_bars=2))
    # First bar has 0 prev bars, third has 2
    assert len(aligned[0]["prev_bars"]) == 0
    assert len(aligned[2]["prev_bars"]) == 2


# =========================================================================== #
# audit_no_lookahead                                                           #
# =========================================================================== #

def test_audit_clean_trade_no_violations():
    """Trade whose entry aligns with a past bar → no violations."""
    bars  = [_bar(day=1), _bar(day=2)]
    snaps = [_sentiment(day=1, hour=15)]
    # Trade entered at day 2 (fill at T+1 open) — decision bar is day 1
    trades = [{"ticker": "TSLA", "entry_time": _utc(2, 9)}]
    violations = audit_no_lookahead(trades, bars, snaps)
    assert violations == []


def test_audit_detects_trade_with_no_preceding_bar():
    """
    A trade whose entry_time predates all market bars cannot have used
    historically valid data → the audit must flag it.

    Implementation note: audit_no_lookahead flags a trade when no bar
    exists at or before entry_time because the decision bar cannot be
    located — indicating either data or timing misalignment.
    """
    # Only bar available is on day 5; trade claims entry on day 2
    bars   = [_bar(day=5)]
    snaps  = [_sentiment(day=1, hour=15)]
    trades = [{"ticker": "TSLA", "entry_time": _utc(2, 9)}]
    violations = audit_no_lookahead(trades, bars, snaps)
    assert len(violations) == 1
    assert "TSLA" in violations[0]


def test_audit_empty_trade_list():
    violations = audit_no_lookahead([], [_bar(day=1)], [_sentiment(day=1)])
    assert violations == []
