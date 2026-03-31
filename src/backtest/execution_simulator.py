"""
backtest/execution_simulator.py
---------------------------------
Simulates order fills and stop-loss execution in historical backtesting.

Fill model assumptions (documented)
-------------------------------------
  Entry fill model (default "next_open"):
    - Signal generated at close of bar T using bar T's data.
    - Order filled at open of bar T+1.
    - Slippage: configurable fraction of open price (default 0.0).
    - Commission: flat dollar amount per trade (default 0.0).

  Entry fill model ("same_close"):
    - Order filled at close of bar T.
    - Used for end-of-day signal + end-of-day execution workflows.

  Stop-loss approximation:
    - If low of bar T+1 ≤ stop_loss_price → exit is triggered.
    - Fill price: max(stop_loss_price, low_of_bar) — approximates the
      actual exit price conservatively.  In reality, slippage through the
      stop is common; this is NOT modelled (documented limitation).

  Take-profit approximation:
    - If high of bar T+1 ≥ take_profit_price → exit is triggered.
    - Fill price: min(take_profit_price, high_of_bar).

  Partial fills: NOT modelled.  All fills are assumed complete.

Limitations (explicitly documented)
--------------------------------------
  1. Stop-loss slippage (gap-through) is not modelled.  A stock opening
     well below the stop is filled AT the stop, not at the open.
  2. Commissions and spreads are optional and flat-rate only.
  3. Market impact is not modelled.
  4. Dividends and splits are not adjusted for.
  5. Short selling is not supported (long-only baseline).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from src.backtest.schemas import (
    TradeRecord,
    make_trade_record,
    close_trade_record,
)

logger = logging.getLogger(__name__)


# =========================================================================== #
# Fill price computation                                                       #
# =========================================================================== #

def _apply_slippage(price: float, side: str, slippage_pct: float) -> float:
    """
    Return the price after applying fractional slippage.
    For 'buy':  price * (1 + slippage_pct)
    For 'sell': price * (1 - slippage_pct)
    """
    if slippage_pct == 0.0:
        return price
    if side == "buy":
        return price * (1.0 + slippage_pct)
    return price * (1.0 - slippage_pct)


def compute_fill_price(
    signal_bar: dict,
    next_bar: Optional[dict],
    side: str,
    fill_model: str = "next_open",
    slippage_pct: float = 0.0,
) -> Optional[float]:
    """
    Compute the fill price for an order given the fill model.

    Parameters
    ----------
    signal_bar  : HistoricalMarketRow — bar at which signal was generated
    next_bar    : HistoricalMarketRow | None — bar T+1 (required for "next_open")
    side        : "buy" | "sell"
    fill_model  : "next_open" | "same_close"
    slippage_pct: fraction of price

    Returns
    -------
    float | None
        None if required bar data is missing.
    """
    if fill_model == "next_open":
        if next_bar is None:
            return None
        raw_price = float(next_bar.get("open") or 0)
    elif fill_model == "same_close":
        raw_price = float(signal_bar.get("close") or 0)
    else:
        raise ValueError(f"Unknown fill_model: {fill_model!r}")

    if raw_price <= 0:
        return None
    return _apply_slippage(raw_price, side, slippage_pct)


# =========================================================================== #
# Stop-loss / take-profit check (intra-bar)                                   #
# =========================================================================== #

def check_stop_and_tp(
    position: dict,
    bar: dict,
    slippage_pct: float = 0.0,
) -> Optional[tuple[str, float]]:
    """
    Check whether the stop-loss or take-profit was hit during `bar`.

    Approximation:
      - Stop hit  : low  ≤ stop_loss_price   → fill at stop_loss_price
      - TP hit    : high ≥ take_profit_price → fill at take_profit_price

    Parameters
    ----------
    position : open TradeRecord or PositionState dict with stop_loss_price
    bar      : HistoricalMarketRow for the period to check

    Returns
    -------
    (exit_reason, fill_price) | None
        None → no stop/tp hit this bar.
        Priority: stop-loss is checked before take-profit.
    """
    low        = float(bar.get("low") or 0)
    high       = float(bar.get("high") or 0)
    stop_price = position.get("stop_loss_price")
    tp_price   = position.get("take_profit_price")

    # Stop-loss check
    if stop_price is not None and low <= float(stop_price):
        fill = _apply_slippage(float(stop_price), "sell", slippage_pct)
        # Conservatively: cannot fill worse than the bar low
        fill = max(fill, low)
        return ("stop_loss", round(fill, 6))

    # Take-profit check
    if tp_price is not None and high >= float(tp_price):
        fill = _apply_slippage(float(tp_price), "sell", slippage_pct)
        fill = min(fill, high)
        return ("take_profit", round(fill, 6))

    return None


# =========================================================================== #
# Execution simulator                                                          #
# =========================================================================== #

class ExecutionSimulator:
    """
    Simulates entries, exits, stop-losses, and costs for one ticker.

    State
    -----
    The simulator is stateless about positions — the caller passes in any
    open position and receives updated state.  The `PortfolioTracker` owns
    position state; this module only handles fill math.

    Parameters
    ----------
    fill_model      : "next_open" | "same_close"
    slippage_pct    : float   (fraction, e.g. 0.001 = 0.1%)
    commission      : float   (flat dollar per trade)
    """

    def __init__(
        self,
        fill_model:    str   = "next_open",
        slippage_pct:  float = 0.0,
        commission:    float = 0.0,
    ) -> None:
        self._fill_model   = fill_model
        self._slippage_pct = slippage_pct
        self._commission   = commission
        self._trade_id_seq = 0

    def _next_id(self) -> int:
        self._trade_id_seq += 1
        return self._trade_id_seq

    # ------------------------------------------------------------------ #
    # Entry simulation                                                    #
    # ------------------------------------------------------------------ #

    def simulate_entry(
        self,
        ticker: str,
        signal_bar: dict,
        next_bar: Optional[dict],
        qty: int,
        stop_loss_price: float,
        take_profit_price: Optional[float] = None,
        reference_time: Optional[datetime] = None,
    ) -> Optional[TradeRecord]:
        """
        Simulate a BUY fill.

        Returns a TradeRecord (open) on success, None if fill is not possible
        (e.g., next_bar missing for next_open model or fill price ≤ 0).
        """
        fill_price = compute_fill_price(
            signal_bar, next_bar, "buy",
            self._fill_model, self._slippage_pct,
        )
        if fill_price is None or fill_price <= 0:
            logger.warning(
                "ExecutionSimulator: cannot fill entry for %s (fill_price=%s, next_bar=%s)",
                ticker, fill_price, next_bar is not None,
            )
            return None

        fill_time = reference_time
        if fill_time is None:
            if next_bar is not None:
                fill_time = next_bar.get("timestamp")
            if fill_time is None:
                fill_time = signal_bar.get("timestamp") or datetime.now(tz=timezone.utc)

        record = make_trade_record(
            trade_id        = self._next_id(),
            ticker          = ticker,
            entry_time      = fill_time,
            entry_price     = fill_price,
            qty             = qty,
            stop_loss_price = stop_loss_price,
        )
        if take_profit_price is not None:
            record["take_profit_price"] = take_profit_price

        # Deduct commission from PnL at exit; record gross entry cost for now
        record["commission_paid"] = self._commission

        logger.debug(
            "ExecutionSimulator: BUY %s x%d @ %.4f (stop=%.4f)",
            ticker, qty, fill_price, stop_loss_price,
        )
        return record

    # ------------------------------------------------------------------ #
    # Exit simulation                                                     #
    # ------------------------------------------------------------------ #

    def simulate_exit(
        self,
        trade: TradeRecord,
        signal_bar: dict,
        next_bar: Optional[dict],
        exit_reason: str,
        reference_time: Optional[datetime] = None,
    ) -> TradeRecord:
        """
        Simulate a SELL fill for an open position.

        For signal-driven exits (sentiment reversal, trend exit), the fill
        model applies (next_open or same_close).

        Returns the closed TradeRecord.
        """
        fill_price = compute_fill_price(
            signal_bar, next_bar, "sell",
            self._fill_model, self._slippage_pct,
        )
        if fill_price is None or fill_price <= 0:
            # Fall back to signal bar close if next bar is missing
            fill_price = float(signal_bar.get("close") or trade.get("entry_price") or 0)
            logger.warning(
                "ExecutionSimulator: no next bar for exit of %s — using close %.4f",
                trade.get("ticker"), fill_price,
            )

        fill_time = reference_time
        if fill_time is None:
            if next_bar is not None:
                fill_time = next_bar.get("timestamp")
            if fill_time is None:
                fill_time = signal_bar.get("timestamp") or datetime.now(tz=timezone.utc)

        closed = close_trade_record(trade, fill_time, fill_price, exit_reason)
        # Add commission to costs (subtract from PnL)
        total_commission = (closed.get("commission_paid", 0) or 0) + self._commission
        closed["commission_paid"] = total_commission
        if closed.get("pnl") is not None:
            closed["pnl"] = round(closed["pnl"] - total_commission, 4)

        logger.debug(
            "ExecutionSimulator: SELL %s x%d @ %.4f | reason=%s pnl=%.2f",
            trade.get("ticker"), trade.get("qty", 0), fill_price,
            exit_reason, closed.get("pnl", 0),
        )
        return closed

    # ------------------------------------------------------------------ #
    # Stop/TP check (called each bar while position is open)              #
    # ------------------------------------------------------------------ #

    def check_stops(
        self,
        trade: TradeRecord,
        current_bar: dict,
    ) -> Optional[TradeRecord]:
        """
        Check if the stop-loss or take-profit fires during `current_bar`.

        If triggered, returns a closed TradeRecord.
        Otherwise returns None.
        """
        hit = check_stop_and_tp(trade, current_bar, self._slippage_pct)
        if hit is None:
            return None

        exit_reason, fill_price = hit
        fill_time = current_bar.get("timestamp") or datetime.now(tz=timezone.utc)
        closed = close_trade_record(trade, fill_time, fill_price, exit_reason)

        # Deduct commission
        total_commission = (closed.get("commission_paid", 0) or 0) + self._commission
        closed["commission_paid"] = total_commission
        if closed.get("pnl") is not None:
            closed["pnl"] = round(closed["pnl"] - total_commission, 4)

        logger.debug(
            "ExecutionSimulator: STOP/TP %s @ %.4f | reason=%s pnl=%.2f",
            trade.get("ticker"), fill_price, exit_reason, closed.get("pnl", 0),
        )
        return closed
