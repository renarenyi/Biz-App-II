"""
strategy/strategy_engine.py
-----------------------------
Phase 3 top-level orchestrator — the Strategy & Execution Engine.

This is the single public entry point for Phase 3.

Public interface
----------------
    engine = StrategyEngine()
    result = engine.evaluate(ticker, market_snapshot, sentiment_snapshot)

    # result is a StrategyResult dict:
    # {
    #   "ticker":    "TSLA",
    #   "signal":    "BUY",
    #   "order":     { ...OrderResult... },
    #   "position":  { ...PositionState... },
    #   "reason":    "All entry conditions met: ...",
    #   "timestamp": datetime(...)
    # }

Data flow
---------
market_snapshot (Phase 1) + sentiment_snapshot (Phase 2)
    │
    ▼
signal_rules.evaluate_entry()  OR  evaluate_exit()
    │  deterministic rule evaluation
    ▼
eligibility.check_all_eligibility()
    │  market open, dup guard, freshness, qty, buying power, max pos
    ▼
position_sizer.percent_of_equity()
    │  compute integer qty
    ▼
risk_manager.compute_stop_loss()
    │  pre-calculate stop level
    ▼
execution_engine.submit_market_order()
    │  Alpaca Paper Trading
    ▼
order_monitor.add_position()   OR   close_position()
    │  update in-memory state
    ▼
strategy_logger.log_*()
    │  JSONL + Python logger
    ▼
StrategyResult dict  →  Phase 4 backtester / loop

Decision logic summary
-----------------------
For each ticker evaluation:

  1. If there is already an open position → evaluate_exit()
       - If EXIT signal → execute sell order, close position
       - If HOLD signal → return HOLD with no order

  2. If no open position → evaluate_entry()
       - If BUY signal passes eligibility → execute buy order, register position
       - Otherwise → return NO_ACTION with no order

Phase 4 compatibility
----------------------
The same signal_rules functions are callable without the execution and
monitoring side effects, so backtesting can reuse the pure rule logic.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from src.strategy.schemas import (
    InputMarketSnapshot,
    InputSentimentSnapshot,
    SignalDecision,
    make_no_action,
    validate_market_snapshot,
    validate_sentiment_snapshot,
)
from src.strategy.signal_rules import (
    evaluate_entry,
    evaluate_exit,
    DEFAULT_CONVICTION_THRESHOLD,
    DEFAULT_SENTIMENT_MAX_AGE_HOURS,
    DEFAULT_NEG_CONVICTION_THRESHOLD,
)
from src.strategy.eligibility import check_all_eligibility, DEFAULT_MAX_CONCURRENT_POSITIONS
from src.strategy.position_sizer import percent_of_equity
from src.strategy.risk_manager import compute_stop_loss, compute_take_profit, assess_position
from src.strategy.execution_engine import ExecutionEngine
from src.strategy.order_monitor import OrderMonitor
from src.strategy.logger import StrategyLogger
from src.strategy.schemas import make_position_state

logger = logging.getLogger(__name__)


# =========================================================================== #
# StrategyEngine                                                               #
# =========================================================================== #

class StrategyEngine:
    """
    Phase 3 strategy and paper execution orchestrator.

    Parameters
    ----------
    execution_engine : ExecutionEngine, optional
        If None, a dry-run engine is created automatically.
    monitor : OrderMonitor, optional
        Shared position tracker.
    strategy_logger : StrategyLogger, optional
        Structured audit logger.
    conviction_threshold : float
        Minimum sentiment conviction to enter a trade.
    sentiment_max_age_hours : float
        Maximum age of a sentiment signal before it is treated as stale.
    stop_loss_pct : float
        Stop-loss percentage below entry price (e.g. 0.02 = 2%).
    take_profit_pct : float
        Take-profit percentage above entry price.  Set take_profit_enabled=True to use.
    take_profit_enabled : bool
        Whether to calculate and apply take-profit exits.
    equity_fraction : float
        Portfolio fraction to allocate per trade (e.g. 0.05 = 5%).
    max_concurrent_positions : int
        Maximum number of simultaneously open positions.
    trend_exit_enabled : bool
        Whether a break below SMA-50 triggers a position exit.
    """

    def __init__(
        self,
        execution_engine: Optional[ExecutionEngine] = None,
        monitor: Optional[OrderMonitor] = None,
        strategy_logger: Optional[StrategyLogger] = None,
        conviction_threshold:     float = DEFAULT_CONVICTION_THRESHOLD,
        sentiment_max_age_hours:  float = DEFAULT_SENTIMENT_MAX_AGE_HOURS,
        stop_loss_pct:            float = 0.02,
        take_profit_pct:          float = 0.04,
        take_profit_enabled:      bool  = False,
        equity_fraction:          float = 0.05,
        max_concurrent_positions: int   = DEFAULT_MAX_CONCURRENT_POSITIONS,
        trend_exit_enabled:       bool  = True,
        log_path:                 str   = "data/logs/strategy_decisions.jsonl",
        use_file_logging:         bool  = True,
    ) -> None:
        self._engine   = execution_engine or ExecutionEngine(
            api_key="", secret_key="",
            base_url="https://paper-api.alpaca.markets",
            dry_run=True,
        )
        self._monitor  = monitor or OrderMonitor(max_positions=max_concurrent_positions)
        self._log      = strategy_logger or StrategyLogger(log_path, use_file=use_file_logging)

        # Tunable parameters
        self._conviction_threshold     = conviction_threshold
        self._sentiment_max_age_hours  = sentiment_max_age_hours
        self._stop_loss_pct            = stop_loss_pct
        self._take_profit_pct          = take_profit_pct
        self._take_profit_enabled      = take_profit_enabled
        self._equity_fraction          = equity_fraction
        self._max_concurrent_positions = max_concurrent_positions
        self._trend_exit_enabled       = trend_exit_enabled

        logger.info(
            "StrategyEngine: initialized "
            "(conviction_thresh=%.1f, sentiment_max_age=%.0fh, "
            "stop_loss=%.1f%%, equity_fraction=%.1f%%, max_pos=%d)",
            conviction_threshold, sentiment_max_age_hours,
            stop_loss_pct * 100, equity_fraction * 100, max_concurrent_positions,
        )

    # ------------------------------------------------------------------ #
    # Primary public method                                               #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        ticker: str,
        market: dict,
        sentiment: dict,
        reference_time: Optional[datetime] = None,
    ) -> dict:
        """
        Run one full decision cycle for a ticker.

        Parameters
        ----------
        ticker : str
        market : dict (InputMarketSnapshot)
        sentiment : dict (InputSentimentSnapshot)
        reference_time : datetime, optional
            UTC anchor.  Defaults to now.

        Returns
        -------
        dict (StrategyResult)
            Always returns — never raises.
            signal = "NO_ACTION" if inputs are invalid.
        """
        ref = reference_time or datetime.now(tz=timezone.utc)
        ts  = ref

        # ── Validate inputs ───────────────────────────────────────────── #
        if not validate_market_snapshot(market):
            decision = make_no_action(ticker, "Invalid or missing market snapshot.", ts)
            self._log.log_signal_decision(decision)
            return _build_result(ticker, decision, None, None, ts)

        if not validate_sentiment_snapshot(sentiment):
            decision = make_no_action(ticker, "Invalid or missing sentiment snapshot.", ts)
            self._log.log_signal_decision(decision)
            return _build_result(ticker, decision, None, None, ts)

        # ── Branch: existing position → exit logic ────────────────────── #
        existing_position = self._monitor.get_position(ticker)

        if existing_position:
            return self._handle_existing_position(ticker, market, sentiment, existing_position, ts)

        # ── Branch: no position → entry logic ────────────────────────── #
        return self._handle_entry(ticker, market, sentiment, ts)

    # ------------------------------------------------------------------ #
    # Evaluate multiple tickers in one call                               #
    # ------------------------------------------------------------------ #

    def evaluate_batch(
        self,
        snapshots: list[tuple[dict, dict]],
        reference_time: Optional[datetime] = None,
    ) -> list[dict]:
        """
        Evaluate a list of (market_snapshot, sentiment_snapshot) pairs.

        Parameters
        ----------
        snapshots : list[tuple[dict, dict]]
        reference_time : datetime, optional

        Returns
        -------
        list[dict]  (StrategyResult per ticker)
        """
        ref = reference_time or datetime.now(tz=timezone.utc)
        results = []
        for market, sentiment in snapshots:
            ticker = market.get("ticker", "UNKNOWN")
            results.append(self.evaluate(ticker, market, sentiment, reference_time=ref))
        return results

    # ------------------------------------------------------------------ #
    # Entry branch                                                        #
    # ------------------------------------------------------------------ #

    def _handle_entry(
        self,
        ticker: str,
        market: dict,
        sentiment: dict,
        ts: datetime,
    ) -> dict:
        # Step 1 — Signal rules
        decision = evaluate_entry(
            market=market,
            sentiment=sentiment,
            conviction_threshold=self._conviction_threshold,
            sentiment_max_age_hours=self._sentiment_max_age_hours,
            reference_time=ts,
        )

        if decision["signal"] != "BUY":
            decision["eligibility_pass"] = False
            self._log.log_signal_decision(decision)
            return _build_result(ticker, decision, None, None, ts)

        # Step 2 — Sizing
        account = self._engine.get_account() or {}
        equity       = float(account.get("equity", 0))
        buying_power = float(account.get("buying_power", 0))
        price        = float(market.get("close", 0))

        qty = percent_of_equity(price, equity, self._equity_fraction) if equity > 0 else 0

        # Step 3 — Eligibility
        eligibility = check_all_eligibility(
            ticker=ticker,
            signal="BUY",
            qty=qty,
            price=price,
            is_market_open=market.get("is_market_open", False),
            open_positions=self._monitor.all_open_positions(),
            buying_power=buying_power,
            signal_timestamp=ts,
            max_concurrent_positions=self._max_concurrent_positions,
        )

        decision["eligibility_pass"] = eligibility["passed"]

        if not eligibility["passed"]:
            decision["reason"] = decision["reason"] + f" | Eligibility blocked: {eligibility['reason']}"
            self._log.log_signal_decision(decision)
            return _build_result(ticker, decision, None, None, ts)

        # Step 4 — Compute risk levels
        stop_loss   = compute_stop_loss(price, self._stop_loss_pct)
        take_profit = compute_take_profit(price, self._take_profit_pct, enabled=self._take_profit_enabled)

        # Step 5 — Submit order
        order_result = self._engine.submit_market_order(
            ticker=ticker,
            side="buy",
            qty=qty,
            signal_reason=decision["reason"],
        )
        self._log.log_order_event(order_result, signal_reason=decision["reason"])

        if order_result.get("status") != "submitted":
            decision["reason"] += f" | Order FAILED: {order_result.get('error')}"
            self._log.log_signal_decision(decision)
            return _build_result(ticker, decision, order_result, None, ts)

        # Step 6 — Register position
        position = make_position_state(
            ticker=ticker,
            side="long",
            qty=qty,
            entry_price=price,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            order_id=order_result.get("order_id"),
            entered_at=ts,
        )
        self._monitor.add_position(ticker, position)
        self._log.log_position_opened(position)
        self._log.log_signal_decision(decision)

        return _build_result(ticker, decision, order_result, position, ts)

    # ------------------------------------------------------------------ #
    # Exit branch                                                        #
    # ------------------------------------------------------------------ #

    def _handle_existing_position(
        self,
        ticker: str,
        market: dict,
        sentiment: dict,
        position: dict,
        ts: datetime,
    ) -> dict:
        # Step 1 — Assess risk levels first (stop-loss override)
        close = float(market.get("close", 0))
        risk_result = assess_position(position, close)
        if risk_result.get("action") == "EXIT":
            self._log.log_risk_event(risk_result, context="assess_position")
            # Force exit via signal rules result
            decision = {
                "ticker":           ticker,
                "timestamp":        ts,
                "signal":           "EXIT",
                "technical_pass":   False,
                "sentiment_pass":   False,
                "eligibility_pass": True,
                "reason":           risk_result["reason"],
                "price":            close,
                "sma_50":           market.get("sma_50"),
                "sentiment":        sentiment.get("sentiment"),
                "conviction_score": sentiment.get("conviction_score"),
            }
            return self._execute_exit(ticker, decision, position, ts)

        # Step 2 — Evaluate signal rules for exit
        decision = evaluate_exit(
            market=market,
            sentiment=sentiment,
            position=position,
            neg_conviction_threshold=DEFAULT_NEG_CONVICTION_THRESHOLD,
            trend_exit_enabled=self._trend_exit_enabled,
            sentiment_max_age_hours=self._sentiment_max_age_hours,
            reference_time=ts,
        )

        if decision["signal"] == "EXIT":
            return self._execute_exit(ticker, decision, position, ts)

        # HOLD — no action needed
        self._log.log_signal_decision(decision)
        return _build_result(ticker, decision, None, position, ts)

    def _execute_exit(
        self,
        ticker: str,
        decision: dict,
        position: dict,
        ts: datetime,
    ) -> dict:
        qty = position.get("qty", 0)
        order_result = self._engine.submit_market_order(
            ticker=ticker,
            side="sell",
            qty=qty,
            signal_reason=decision["reason"],
        )
        self._log.log_order_event(order_result, signal_reason=decision["reason"])

        exit_price = float(order_result.get("fill_price") or position.get("entry_price") or 0)
        closed = self._monitor.close_position(ticker, exit_price=exit_price or None)
        if closed:
            self._log.log_position_closed(closed)

        self._log.log_signal_decision(decision)
        return _build_result(ticker, decision, order_result, closed or position, ts)

    # ------------------------------------------------------------------ #
    # Introspection                                                       #
    # ------------------------------------------------------------------ #

    @property
    def monitor(self) -> OrderMonitor:
        return self._monitor

    def open_positions(self) -> dict[str, dict]:
        return self._monitor.all_open_positions()


# =========================================================================== #
# Result builder                                                               #
# =========================================================================== #

def _build_result(
    ticker: str,
    decision: dict,
    order: Optional[dict],
    position: Optional[dict],
    ts: datetime,
) -> dict:
    return {
        "ticker":    ticker,
        "signal":    decision.get("signal", "NO_ACTION"),
        "reason":    decision.get("reason", ""),
        "order":     order,
        "position":  position,
        "timestamp": ts,
        "decision":  decision,
    }
