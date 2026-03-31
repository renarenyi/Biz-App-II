"""
strategy/order_monitor.py
--------------------------
In-memory position and order state tracker for Phase 3.

This module is the single source of truth for:
  - which tickers have open long positions
  - what the entry price and stop-loss level are for each position
  - which orders are pending resolution

It is intentionally in-memory only.  Persistence to disk is a Phase 4 concern
(the backtester manages its own state).  For paper trading runs that span
multiple sessions, the caller can checkpoint and restore via
`export_state()` / `restore_state()`.

Thread safety: not thread-safe.  The trading loop is assumed to be single-threaded.

Public interface
----------------
  monitor.add_position(ticker, position_state)
  monitor.get_position(ticker)     → PositionState dict | None
  monitor.close_position(ticker)
  monitor.has_open_position(ticker) → bool
  monitor.all_open_positions()     → dict[str, dict]
  monitor.open_position_count()    → int
  monitor.export_state()           → dict
  monitor.restore_state(state)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class OrderMonitor:
    """
    In-memory tracker for open positions and pending orders.

    Parameters
    ----------
    max_positions : int
        Hard cap on concurrent open positions.  Used for guard logging only —
        the actual enforcement is in eligibility.py.
    """

    def __init__(self, max_positions: int = 5) -> None:
        self._positions: dict[str, dict] = {}
        self._closed_positions: list[dict] = []
        self._max_positions = max_positions

    # ------------------------------------------------------------------ #
    # Position management                                                  #
    # ------------------------------------------------------------------ #

    def add_position(self, ticker: str, position_state: dict) -> None:
        """
        Register a new open position.

        Parameters
        ----------
        ticker : str
        position_state : dict (PositionState)
            Must contain at minimum: entry_price, stop_loss_price, qty, side.
        """
        if ticker in self._positions and self._positions[ticker].get("status") == "open":
            logger.warning(
                "OrderMonitor: add_position called for %s but an open position already exists. "
                "Overwriting — check for duplicate orders.", ticker,
            )

        position_state = dict(position_state)
        position_state["ticker"] = ticker
        position_state.setdefault("status", "open")
        position_state.setdefault("entered_at", datetime.now(tz=timezone.utc))

        self._positions[ticker] = position_state

        logger.info(
            "OrderMonitor: position added for %s (entry=%.2f, stop=%.2f, qty=%d).",
            ticker,
            position_state.get("entry_price", 0),
            position_state.get("stop_loss_price", 0),
            position_state.get("qty", 0),
        )

    def get_position(self, ticker: str) -> Optional[dict]:
        """
        Retrieve the current PositionState for a ticker.

        Returns None if there is no open position.
        """
        pos = self._positions.get(ticker)
        if pos and pos.get("status") == "open":
            return pos
        return None

    def close_position(self, ticker: str, exit_price: Optional[float] = None) -> Optional[dict]:
        """
        Mark a position as closed.  Moves it to the closed log.

        Parameters
        ----------
        ticker : str
        exit_price : float, optional
            The fill price at exit.

        Returns
        -------
        dict | None
            The closed PositionState, or None if no open position existed.
        """
        pos = self._positions.get(ticker)
        if not pos or pos.get("status") != "open":
            logger.debug("OrderMonitor: close_position(%s) — no open position found.", ticker)
            return None

        pos = dict(pos)
        pos["status"] = "closed"
        pos["exited_at"] = datetime.now(tz=timezone.utc)
        if exit_price is not None:
            pos["exit_price"] = exit_price

        self._positions[ticker] = pos
        self._closed_positions.append(pos)

        logger.info(
            "OrderMonitor: position closed for %s (exit_price=%s).",
            ticker, f"{exit_price:.2f}" if exit_price else "unknown",
        )
        return pos

    def update_stop_loss(self, ticker: str, new_stop: float) -> bool:
        """
        Update the stop-loss level for an open position.

        Returns True if updated, False if no open position found.
        """
        pos = self._positions.get(ticker)
        if not pos or pos.get("status") != "open":
            return False
        old_stop = pos.get("stop_loss_price", 0)
        pos["stop_loss_price"] = new_stop
        logger.info(
            "OrderMonitor: stop_loss updated for %s: %.2f → %.2f.", ticker, old_stop, new_stop
        )
        return True

    # ------------------------------------------------------------------ #
    # Queries                                                             #
    # ------------------------------------------------------------------ #

    def has_open_position(self, ticker: str) -> bool:
        pos = self._positions.get(ticker)
        return pos is not None and pos.get("status") == "open"

    def all_open_positions(self) -> dict[str, dict]:
        """Return a copy of all open PositionState dicts."""
        return {k: dict(v) for k, v in self._positions.items() if v.get("status") == "open"}

    def open_position_count(self) -> int:
        return sum(1 for p in self._positions.values() if p.get("status") == "open")

    def closed_position_count(self) -> int:
        return len(self._closed_positions)

    def all_closed_positions(self) -> list[dict]:
        return list(self._closed_positions)

    # ------------------------------------------------------------------ #
    # State export / restore (for session persistence)                   #
    # ------------------------------------------------------------------ #

    def export_state(self) -> dict:
        """
        Export all position state to a serializable dict.

        Datetimes are serialized to ISO strings.
        """
        return {
            "positions": {k: _make_serialisable(v) for k, v in self._positions.items()},
            "closed":    [_make_serialisable(p) for p in self._closed_positions],
        }

    def restore_state(self, state: dict) -> None:
        """
        Restore position state from a previously exported dict.

        Parameters
        ----------
        state : dict
            Output of export_state().
        """
        for ticker, pos in state.get("positions", {}).items():
            self._positions[ticker] = pos
        self._closed_positions = list(state.get("closed", []))
        logger.info(
            "OrderMonitor: restored %d positions (%d closed).",
            len(self._positions), len(self._closed_positions),
        )

    def save_to_file(self, path: str | Path) -> None:
        """Save state to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.export_state(), fh, indent=2, default=str)
        logger.info("OrderMonitor: state saved to %s.", path)

    def load_from_file(self, path: str | Path) -> None:
        """Load state from a previously saved JSON file."""
        p = Path(path)
        if not p.exists():
            logger.info("OrderMonitor: no state file at %s — starting fresh.", path)
            return
        with open(p) as fh:
            state = json.load(fh)
        self.restore_state(state)

    # ------------------------------------------------------------------ #
    # Representation                                                      #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"OrderMonitor(open={self.open_position_count()}, "
            f"closed={self.closed_position_count()})"
        )


# =========================================================================== #
# Serialization helper                                                         #
# =========================================================================== #

def _make_serialisable(obj):
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serialisable(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj
