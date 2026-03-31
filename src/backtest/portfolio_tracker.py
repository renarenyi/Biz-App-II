"""
backtest/portfolio_tracker.py
-------------------------------
Tracks cash, equity, open positions, and mark-to-market portfolio value
throughout a backtest run.

State model
-----------
  cash           : float  — uninvested capital
  equity         : float  — cash + mark-to-market value of all open positions
  open_positions : dict   — ticker → open TradeRecord

Portfolio value is recomputed each time `record_snapshot(prices)` is called.
The equity_curve list accumulates one PortfolioSnapshot per bar.

Position limits
---------------
The tracker enforces `max_concurrent_positions` at entry time.  Attempts to
open a position when the limit is reached return False.

Multi-ticker support
--------------------
The tracker holds positions across all tickers simultaneously.  The
backtester drives iteration; the tracker only stores state.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from src.backtest.schemas import PortfolioSnapshot, TradeRecord, close_trade_record

logger = logging.getLogger(__name__)


class PortfolioTracker:
    """
    Stateful portfolio tracker for one backtest run.

    Parameters
    ----------
    initial_capital          : float
    max_concurrent_positions : int
    """

    def __init__(
        self,
        initial_capital:           float = 100_000.0,
        max_concurrent_positions:  int   = 5,
    ) -> None:
        self._initial_capital = float(initial_capital)
        self._cash             = float(initial_capital)
        self._max_positions    = max_concurrent_positions

        # ticker → open TradeRecord
        self._open_positions: dict[str, TradeRecord] = {}

        # Completed trades (closed round-trips)
        self._closed_trades: list[TradeRecord] = []

        # Equity curve: list of PortfolioSnapshot
        self._equity_curve: list[PortfolioSnapshot] = []

    # ------------------------------------------------------------------ #
    # Accessors                                                           #
    # ------------------------------------------------------------------ #

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def initial_capital(self) -> float:
        return self._initial_capital

    def get_position(self, ticker: str) -> Optional[TradeRecord]:
        """Return the open TradeRecord for ticker, or None."""
        return self._open_positions.get(ticker)

    def has_open_position(self, ticker: str) -> bool:
        return ticker in self._open_positions

    def open_position_count(self) -> int:
        return len(self._open_positions)

    def all_open_positions(self) -> dict[str, TradeRecord]:
        return dict(self._open_positions)

    def closed_trades(self) -> list[TradeRecord]:
        return list(self._closed_trades)

    def equity_curve(self) -> list[PortfolioSnapshot]:
        return list(self._equity_curve)

    # ------------------------------------------------------------------ #
    # Mark-to-market                                                      #
    # ------------------------------------------------------------------ #

    def current_equity(self, prices: Optional[dict[str, float]] = None) -> float:
        """
        Return current portfolio equity.

        Parameters
        ----------
        prices : dict[ticker, current_price] | None
            If provided, open positions are valued at these prices.
            If None, positions are valued at their entry prices.
        """
        position_value = 0.0
        for ticker, pos in self._open_positions.items():
            qty   = pos.get("qty", 0)
            price = (prices or {}).get(ticker) or pos.get("entry_price") or 0
            position_value += qty * float(price)
        return round(self._cash + position_value, 4)

    def record_snapshot(
        self,
        timestamp: datetime,
        prices: Optional[dict[str, float]] = None,
    ) -> PortfolioSnapshot:
        """
        Compute and record a PortfolioSnapshot at `timestamp`.
        """
        position_value = 0.0
        for ticker, pos in self._open_positions.items():
            qty   = pos.get("qty", 0)
            price = (prices or {}).get(ticker) or pos.get("entry_price") or 0
            position_value += qty * float(price)

        equity = round(self._cash + position_value, 4)

        snapshot: PortfolioSnapshot = {
            "timestamp":      timestamp,
            "cash":           round(self._cash, 4),
            "equity":         equity,
            "gross_exposure": round(position_value, 4),
            "net_exposure":   round(position_value, 4),   # long-only → same
            "open_positions": len(self._open_positions),
        }
        self._equity_curve.append(snapshot)
        return snapshot

    # ------------------------------------------------------------------ #
    # Entry                                                               #
    # ------------------------------------------------------------------ #

    def enter_position(self, ticker: str, trade: TradeRecord) -> bool:
        """
        Register a new open position.  Deducts entry cost from cash.

        Returns True on success, False if:
          - position already open for ticker
          - max concurrent positions reached
          - insufficient cash
        """
        if ticker in self._open_positions:
            logger.warning("PortfolioTracker: duplicate position blocked for %s", ticker)
            return False

        if len(self._open_positions) >= self._max_positions:
            logger.warning(
                "PortfolioTracker: max positions (%d) reached, cannot open %s",
                self._max_positions, ticker,
            )
            return False

        cost       = float(trade["entry_price"]) * int(trade["qty"])
        commission = float(trade.get("commission_paid", 0) or 0)
        total_cost = cost + commission

        if total_cost > self._cash + 1e-6:   # small epsilon for floating point
            logger.warning(
                "PortfolioTracker: insufficient cash (%.2f) for %s trade cost (%.2f)",
                self._cash, ticker, total_cost,
            )
            return False

        self._open_positions[ticker] = trade
        self._cash -= total_cost
        self._cash  = round(self._cash, 4)

        logger.debug(
            "PortfolioTracker: opened %s x%d @ %.4f  cash=%.2f",
            ticker, trade["qty"], trade["entry_price"], self._cash,
        )
        return True

    # ------------------------------------------------------------------ #
    # Exit                                                                #
    # ------------------------------------------------------------------ #

    def exit_position(self, ticker: str, closed_trade: TradeRecord) -> bool:
        """
        Register a completed trade exit.  Adds proceeds to cash.

        Parameters
        ----------
        ticker       : str
        closed_trade : TradeRecord with exit_price, exit_time, pnl filled

        Returns True on success, False if no open position found.
        """
        if ticker not in self._open_positions:
            logger.warning("PortfolioTracker: no open position to close for %s", ticker)
            return False

        open_trade = self._open_positions.pop(ticker)

        # Cash in: gross proceeds of the exit
        qty      = int(open_trade.get("qty", 0))
        exit_prc = float(closed_trade.get("exit_price", 0) or 0)
        proceeds = exit_prc * qty

        # Commission already deducted from pnl; add back raw proceeds
        self._cash += proceeds
        self._cash  = round(self._cash, 4)

        self._closed_trades.append(closed_trade)

        logger.debug(
            "PortfolioTracker: closed %s x%d @ %.4f  pnl=%.2f  cash=%.2f",
            ticker,
            closed_trade.get("qty", 0),
            closed_trade.get("exit_price", 0),
            closed_trade.get("pnl", 0),
            self._cash,
        )
        return True

    # ------------------------------------------------------------------ #
    # End-of-period: close all open positions at last known price        #
    # ------------------------------------------------------------------ #

    def force_close_all(
        self,
        prices: dict[str, float],
        timestamp: datetime,
        exit_reason: str = "end_of_period",
    ) -> list[TradeRecord]:
        """
        Close all remaining open positions at the given prices.
        Used at the end of the backtest window.
        Returns list of closed TradeRecords.
        """
        closed_list = []
        for ticker in list(self._open_positions.keys()):
            trade      = self._open_positions[ticker]
            exit_price = prices.get(ticker) or float(trade.get("entry_price", 0))
            closed_t   = close_trade_record(trade, timestamp, exit_price, exit_reason)
            self.exit_position(ticker, closed_t)
            closed_list.append(closed_t)
        return closed_list

    # ------------------------------------------------------------------ #
    # Summary helpers                                                     #
    # ------------------------------------------------------------------ #

    def trade_count(self) -> int:
        return len(self._closed_trades)

    def all_trades(self) -> list[TradeRecord]:
        """Closed trades only; open trades excluded."""
        return list(self._closed_trades)

    def reset(self) -> None:
        """Reset all state for a new run (useful for parameter sweeps)."""
        self._cash = self._initial_capital
        self._open_positions.clear()
        self._closed_trades.clear()
        self._equity_curve.clear()
