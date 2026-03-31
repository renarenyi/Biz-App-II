"""
strategy/logger.py
-------------------
Structured decision, order, and risk logging for Phase 3.

This module writes human-readable + machine-parseable records to:
  1. A rotating JSON-lines log file  (one JSON object per line)
  2. The Python logger at INFO level (for console / syslog)

Log categories
--------------
  log_signal_decision(decision)   — every signal evaluation cycle
  log_order_event(result, ...)    — every order submission attempt
  log_risk_event(result, ...)     — stop-loss hits, TP, forced exits
  log_position_opened(position)   — new position registered
  log_position_closed(position)   — position closed

Each record is tagged with a category field so downstream consumers
(dashboards, Phase 4 analyzer) can filter easily.

Record format
-------------
Every record includes at minimum:
  category   : str  (signal | order | risk | position_open | position_close)
  ticker     : str
  timestamp  : ISO 8601 string (UTC)
  data       : dict  (category-specific payload)

File management
---------------
The log file path defaults to "data/logs/strategy_decisions.jsonl".
The file is opened in append mode — never truncated.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class StrategyLogger:
    """
    Structured logger for Phase 3 strategy decisions.

    Parameters
    ----------
    log_path : str | Path
        Path to the JSONL log file.
    use_file : bool
        Set to False in unit tests to suppress file writes.
    """

    def __init__(
        self,
        log_path: str | Path = "data/logs/strategy_decisions.jsonl",
        use_file: bool = True,
    ) -> None:
        self._log_path = Path(log_path)
        self._use_file = use_file

        if use_file:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("StrategyLogger: writing to %s", self._log_path)

    # ------------------------------------------------------------------ #
    # Public log methods                                                  #
    # ------------------------------------------------------------------ #

    def log_signal_decision(self, decision: dict) -> None:
        """
        Log a SignalDecision — called for every evaluated ticker.

        Includes decisions that resulted in NO_ACTION or HOLD,
        not just executed trades.
        """
        self._write("signal", decision.get("ticker", "UNKNOWN"), {
            "signal":           decision.get("signal"),
            "technical_pass":   decision.get("technical_pass"),
            "sentiment_pass":   decision.get("sentiment_pass"),
            "eligibility_pass": decision.get("eligibility_pass"),
            "reason":           decision.get("reason"),
            "price":            decision.get("price"),
            "sma_50":           decision.get("sma_50"),
            "sentiment":        decision.get("sentiment"),
            "conviction_score": decision.get("conviction_score"),
        })
        logger.info(
            "[SIGNAL] %s → %s | %s",
            decision.get("ticker", "?"),
            decision.get("signal", "?"),
            decision.get("reason", ""),
        )

    def log_order_event(
        self,
        result: dict,
        request: Optional[dict] = None,
        signal_reason: Optional[str] = None,
    ) -> None:
        """
        Log an order submission result.

        Parameters
        ----------
        result : dict (OrderResult)
        request : dict (OrderRequest), optional
        signal_reason : str, optional
        """
        payload = {
            "status":       result.get("status"),
            "order_id":     result.get("order_id"),
            "side":         result.get("side"),
            "qty":          result.get("qty"),
            "error":        result.get("error"),
            "signal_reason": signal_reason,
        }
        if request:
            payload["order_type"]    = request.get("order_type")
            payload["time_in_force"] = request.get("time_in_force")

        self._write("order", result.get("ticker", "UNKNOWN"), payload)

        level = logging.INFO if result.get("status") == "submitted" else logging.ERROR
        logger.log(
            level,
            "[ORDER] %s %s x%d → %s (id=%s, err=%s)",
            result.get("side", "?").upper(),
            result.get("ticker", "?"),
            result.get("qty", 0),
            result.get("status", "?"),
            result.get("order_id", "n/a"),
            result.get("error", "none"),
        )

    def log_risk_event(self, risk_result: dict, context: Optional[str] = None) -> None:
        """
        Log a risk event (stop hit, TP hit, forced exit, duplicate block, etc.).

        Parameters
        ----------
        risk_result : dict (RiskCheckResult)
        context : str, optional
            Free-form note (e.g., "assess_position", "portfolio_risk_check")
        """
        self._write("risk", risk_result.get("ticker", "UNKNOWN"), {
            "passed":      risk_result.get("passed"),
            "action":      risk_result.get("action"),
            "stop_hit":    risk_result.get("stop_hit"),
            "tp_hit":      risk_result.get("tp_hit"),
            "reason":      risk_result.get("reason"),
            "context":     context,
        })
        logger.info(
            "[RISK] %s → %s | %s",
            risk_result.get("ticker", "?"),
            risk_result.get("action", "none"),
            risk_result.get("reason", ""),
        )

    def log_position_opened(self, position: dict) -> None:
        """Log when a new position is registered."""
        self._write("position_open", position.get("ticker", "UNKNOWN"), {
            "side":            position.get("side"),
            "qty":             position.get("qty"),
            "entry_price":     position.get("entry_price"),
            "stop_loss_price": position.get("stop_loss_price"),
            "take_profit_price": position.get("take_profit_price"),
            "order_id":        position.get("order_id"),
        })
        logger.info(
            "[POSITION OPEN] %s x%d @ %.2f (stop=%.2f)",
            position.get("ticker", "?"),
            position.get("qty", 0),
            position.get("entry_price", 0),
            position.get("stop_loss_price", 0),
        )

    def log_position_closed(self, position: dict) -> None:
        """Log when a position is closed."""
        entry = position.get("entry_price", 0)
        exit_ = position.get("exit_price", 0)
        pnl   = (exit_ - entry) * position.get("qty", 0) if entry and exit_ else None

        self._write("position_close", position.get("ticker", "UNKNOWN"), {
            "side":        position.get("side"),
            "qty":         position.get("qty"),
            "entry_price": entry,
            "exit_price":  exit_,
            "pnl_est":     round(pnl, 2) if pnl is not None else None,
            "entered_at":  _iso(position.get("entered_at")),
            "exited_at":   _iso(position.get("exited_at")),
        })
        logger.info(
            "[POSITION CLOSE] %s x%d entry=%.2f exit=%s pnl=%s",
            position.get("ticker", "?"),
            position.get("qty", 0),
            entry,
            f"{exit_:.2f}" if exit_ else "n/a",
            f"${pnl:.2f}" if pnl is not None else "n/a",
        )

    # ------------------------------------------------------------------ #
    # Core write                                                          #
    # ------------------------------------------------------------------ #

    def _write(self, category: str, ticker: str, data: dict) -> None:
        record = {
            "category":  category,
            "ticker":    ticker,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "data":      _make_serialisable(data),
        }
        if self._use_file:
            try:
                with open(self._log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record) + "\n")
            except Exception as exc:
                logger.warning("StrategyLogger: write error — %s", exc)


# =========================================================================== #
# Helpers                                                                      #
# =========================================================================== #

def _iso(dt) -> Optional[str]:
    if isinstance(dt, datetime):
        return dt.isoformat()
    return str(dt) if dt else None


def _make_serialisable(obj):
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serialisable(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj
