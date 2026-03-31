"""
backtest/strategy_adapter.py
------------------------------
Bridges Phase 3 signal rules into the backtesting framework.

Design principle
----------------
The Phase 3 `signal_rules.evaluate_entry()` and `evaluate_exit()` are pure
functions.  This adapter:
  1. Translates a `HistoricalMarketRow` + `HistoricalSentimentSnapshot` into
     the dicts expected by Phase 3 functions (same schema, just populated
     from historical data).
  2. Calls those functions with the config thresholds from `BacktestConfig`.
  3. Returns the Phase 3 `SignalDecision` dict unchanged.

No strategy logic is implemented here.  Any divergence between the live
paper-trading signal and the backtest signal is a bug.

Compatibility guarantee
-----------------------
When `evaluate_entry` or `evaluate_exit` are called via this adapter with
identical input dicts, they must return the same signal as calling them
directly from Phase 3 with those same dicts.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from src.strategy.signal_rules import (
    evaluate_entry,
    evaluate_exit,
    DEFAULT_CONVICTION_THRESHOLD,
    DEFAULT_SENTIMENT_MAX_AGE_HOURS,
    DEFAULT_NEG_CONVICTION_THRESHOLD,
)
from src.strategy.risk_manager import compute_stop_loss, compute_take_profit
from src.strategy.position_sizer import percent_of_equity
from src.backtest.schemas import BacktestConfig

logger = logging.getLogger(__name__)

# =========================================================================== #
# Dict translation helpers                                                     #
# =========================================================================== #

def _to_market_snapshot(bar: dict, reference_time: Optional[datetime] = None) -> dict:
    """
    Convert a HistoricalMarketRow into an InputMarketSnapshot (Phase 3 schema).

    The `is_market_open` flag is set to True because historical daily bars
    represent completed trading sessions.
    """
    ts = reference_time or bar.get("timestamp") or datetime.now(tz=timezone.utc)
    return {
        "ticker":          bar.get("ticker", ""),
        "timestamp":       ts,
        "close":           float(bar.get("close", 0) or 0),
        "sma_50":          bar.get("sma_50"),                  # may be None
        "volume":          bar.get("volume"),
        "rolling_vol_20d": bar.get("rolling_vol_20d"),
        "is_market_open":  bar.get("is_market_open", True),
        # pass-through any extra fields
        "open":            bar.get("open"),
        "high":            bar.get("high"),
        "low":             bar.get("low"),
    }


def _to_sentiment_snapshot(snap: Optional[dict], ticker: str, reference_time: Optional[datetime] = None) -> dict:
    """
    Convert a HistoricalSentimentSnapshot into an InputSentimentSnapshot.

    If snap is None, return a NEUTRAL snapshot with zero conviction (will
    always produce NO_ACTION from signal rules).
    """
    ts = reference_time or datetime.now(tz=timezone.utc)
    if snap is None:
        return {
            "ticker":               ticker,
            "sentiment":            "NEUTRAL",
            "conviction_score":     0.0,
            "generated_at":         ts,
            "source_count":         0,
            "analysis_window_hours": 24,
        }
    return {
        "ticker":               snap.get("ticker", ticker),
        "sentiment":            snap.get("sentiment", "NEUTRAL"),
        "conviction_score":     float(snap.get("conviction_score", 0) or 0),
        "generated_at":         snap.get("timestamp") or ts,
        "source_count":         snap.get("source_count", 0),
        "analysis_window_hours": snap.get("analysis_window_hours", 24),
    }


# =========================================================================== #
# Strategy adapter                                                             #
# =========================================================================== #

class StrategyAdapter:
    """
    Wraps Phase 3 signal rules for use inside the backtester.

    Parameters
    ----------
    config : BacktestConfig
        All thresholds are read from this config.  Changing config changes
        the strategy behaviour — useful for parameter sweeps.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------ #
    # Entry signal                                                        #
    # ------------------------------------------------------------------ #

    def entry_signal(
        self,
        bar: dict,
        sentiment_snap: Optional[dict],
        reference_time: Optional[datetime] = None,
    ) -> dict:
        """
        Evaluate whether the strategy would enter a long position.

        Parameters
        ----------
        bar            : HistoricalMarketRow
        sentiment_snap : HistoricalSentimentSnapshot | None
                         If None, NO_ACTION will always result.
        reference_time : datetime, optional
                         UTC anchor for staleness checks.  Defaults to bar timestamp.

        Returns
        -------
        dict (SignalDecision)
            signal : "BUY" | "NO_ACTION"
        """
        ts     = reference_time or bar.get("timestamp") or datetime.now(tz=timezone.utc)
        market = _to_market_snapshot(bar, ts)
        sent   = _to_sentiment_snapshot(sentiment_snap, bar.get("ticker", ""), ts)

        return evaluate_entry(
            market=market,
            sentiment=sent,
            conviction_threshold=self._config.get(
                "conviction_threshold", DEFAULT_CONVICTION_THRESHOLD
            ),
            sentiment_max_age_hours=self._config.get(
                "sentiment_max_age_hours", DEFAULT_SENTIMENT_MAX_AGE_HOURS
            ),
            reference_time=ts,
        )

    # ------------------------------------------------------------------ #
    # Exit signal                                                         #
    # ------------------------------------------------------------------ #

    def exit_signal(
        self,
        bar: dict,
        sentiment_snap: Optional[dict],
        position: dict,
        reference_time: Optional[datetime] = None,
    ) -> dict:
        """
        Evaluate whether the strategy would exit an existing long position.

        Parameters
        ----------
        bar            : HistoricalMarketRow
        sentiment_snap : HistoricalSentimentSnapshot | None
        position       : open PositionState dict (must have stop_loss_price, side, qty)
        reference_time : datetime, optional

        Returns
        -------
        dict (SignalDecision)
            signal : "EXIT" | "HOLD"
        """
        ts     = reference_time or bar.get("timestamp") or datetime.now(tz=timezone.utc)
        market = _to_market_snapshot(bar, ts)
        sent   = _to_sentiment_snapshot(sentiment_snap, bar.get("ticker", ""), ts)

        return evaluate_exit(
            market=market,
            sentiment=sent,
            position=position,
            neg_conviction_threshold=self._config.get(
                "neg_conviction_threshold", DEFAULT_NEG_CONVICTION_THRESHOLD
            ),
            trend_exit_enabled=self._config.get("trend_exit_enabled", True),
            sentiment_max_age_hours=self._config.get(
                "sentiment_max_age_hours", DEFAULT_SENTIMENT_MAX_AGE_HOURS
            ),
            reference_time=ts,
        )

    # ------------------------------------------------------------------ #
    # Position sizing                                                     #
    # ------------------------------------------------------------------ #

    def compute_qty(self, price: float, equity: float) -> int:
        """
        Return integer share quantity for a new entry using percent-of-equity sizing.
        Returns 0 if price is 0 or equity is 0.
        """
        if price <= 0 or equity <= 0:
            return 0
        fraction = self._config.get("equity_fraction", 0.05)
        return percent_of_equity(price, equity, fraction)

    # ------------------------------------------------------------------ #
    # Risk levels                                                         #
    # ------------------------------------------------------------------ #

    def compute_stop(self, entry_price: float, side: str = "long") -> float:
        """Compute stop-loss price using config stop_loss_pct."""
        return compute_stop_loss(
            entry_price,
            self._config.get("stop_loss_pct", 0.02),
            side,
        )

    def compute_take_profit(self, entry_price: float, side: str = "long") -> Optional[float]:
        """Compute take-profit price (or None if disabled)."""
        return compute_take_profit(
            entry_price,
            self._config.get("take_profit_pct", 0.04),
            side,
            enabled=self._config.get("take_profit_enabled", False),
        )
