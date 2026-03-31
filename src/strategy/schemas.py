"""
strategy/schemas.py
--------------------
Phase 3 typed data contracts for the Strategy & Execution layer.

Five schemas define the spine of Phase 3:

  InputMarketSnapshot     : technical features from Phase 1
  InputSentimentSnapshot  : structured signal from Phase 2
  SignalDecision          : output of signal_rules — BUY / SELL / HOLD / EXIT / NO_ACTION
  OrderRequest            : payload sent to Alpaca paper execution
  OrderResult             : response from Alpaca (or structured failure)
  PositionState           : live position record with entry price and stop-loss
  RiskCheckResult         : outcome of the risk_manager evaluation

All TypedDicts use total=False for incremental construction.
Runtime validators enforce required fields.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Literal, Optional, TypedDict

logger = logging.getLogger(__name__)

# ── Sentinel types ────────────────────────────────────────────────────────── #

SignalType = Literal["BUY", "SELL", "HOLD", "EXIT", "NO_ACTION"]
OrderSide  = Literal["buy", "sell"]
OrderType  = Literal["market", "limit", "stop", "stop_limit"]
TIF        = Literal["day", "gtc", "ioc", "fok"]
PositionSide = Literal["long", "short"]

VALID_SIGNALS: set[str] = {"BUY", "SELL", "HOLD", "EXIT", "NO_ACTION"}
VALID_SIDES:   set[str] = {"buy", "sell"}


# =========================================================================== #
# InputMarketSnapshot — Phase 1 → Phase 3                                     #
# =========================================================================== #

class InputMarketSnapshot(TypedDict, total=False):
    """
    Technical features from MarketDataHandler, ready for rule evaluation.

    Required: ticker, timestamp, close, is_market_open.
    All others are optional but must be present for the full entry rule.
    """
    ticker:         str
    timestamp:      datetime          # UTC-aware — when the data was fetched
    close:          float             # latest close / last price
    sma_50:         Optional[float]   # 50-day simple moving average
    sma_20:         Optional[float]   # 20-day SMA (optional filter)
    volume:         Optional[float]   # latest bar volume
    volatility_20d: Optional[float]   # 20-day annualised volatility
    is_market_open: bool


MARKET_SNAPSHOT_REQUIRED = {"ticker", "timestamp", "close", "is_market_open"}


def validate_market_snapshot(snap: dict, raise_on_error: bool = False) -> bool:
    missing = MARKET_SNAPSHOT_REQUIRED - set(snap.keys())
    if missing:
        msg = f"InputMarketSnapshot missing required fields: {missing}"
        if raise_on_error: raise ValueError(msg)
        logger.debug(msg); return False
    if snap.get("close") is None or float(snap["close"]) <= 0:
        msg = "InputMarketSnapshot: close price must be positive."
        if raise_on_error: raise ValueError(msg)
        logger.debug(msg); return False
    return True


# =========================================================================== #
# InputSentimentSnapshot — Phase 2 → Phase 3                                  #
# =========================================================================== #

class InputSentimentSnapshot(TypedDict, total=False):
    """
    Ticker-level sentiment result from SentimentAgent.
    Mirrors TickerSentimentResult but only carries the fields Phase 3 needs.
    """
    ticker:               str
    sentiment:            str         # "POSITIVE" | "NEGATIVE" | "NEUTRAL"
    conviction_score:     float       # 0–10
    reasoning:            Optional[str]
    generated_at:         datetime    # UTC-aware — when inference ran
    source_count:         int
    analysis_window_hours: int


SENTIMENT_SNAPSHOT_REQUIRED = {"ticker", "sentiment", "conviction_score", "generated_at"}


def validate_sentiment_snapshot(snap: dict, raise_on_error: bool = False) -> bool:
    missing = SENTIMENT_SNAPSHOT_REQUIRED - set(snap.keys())
    if missing:
        msg = f"InputSentimentSnapshot missing required fields: {missing}"
        if raise_on_error: raise ValueError(msg)
        logger.debug(msg); return False
    return True


# =========================================================================== #
# SignalDecision — output of signal_rules                                      #
# =========================================================================== #

class SignalDecision(TypedDict, total=False):
    """
    The structured output of one signal evaluation cycle.

    Consumed by eligibility.py and execution_engine.py.
    Stored verbatim in the decision log for auditability.
    """
    ticker:           str
    timestamp:        datetime         # UTC — when the decision was made
    signal:           SignalType       # BUY / SELL / HOLD / EXIT / NO_ACTION
    technical_pass:   bool
    sentiment_pass:   bool
    eligibility_pass: bool             # set by eligibility layer (may be None at rule stage)
    reason:           str              # human-readable explanation
    price:            Optional[float]
    sma_50:           Optional[float]
    sentiment:        Optional[str]
    conviction_score: Optional[float]


SIGNAL_DECISION_REQUIRED = {"ticker", "timestamp", "signal", "reason"}


def validate_signal_decision(dec: dict, raise_on_error: bool = False) -> bool:
    missing = SIGNAL_DECISION_REQUIRED - set(dec.keys())
    if missing:
        msg = f"SignalDecision missing required fields: {missing}"
        if raise_on_error: raise ValueError(msg)
        logger.debug(msg); return False
    if dec.get("signal") not in VALID_SIGNALS:
        msg = f"SignalDecision: invalid signal '{dec.get('signal')}'."
        if raise_on_error: raise ValueError(msg)
        logger.debug(msg); return False
    return True


def make_no_action(ticker: str, reason: str, timestamp: Optional[datetime] = None) -> dict:
    """Convenience: construct a NO_ACTION SignalDecision."""
    from datetime import timezone
    return {
        "ticker": ticker,
        "timestamp": timestamp or datetime.now(tz=timezone.utc),
        "signal": "NO_ACTION",
        "technical_pass": False,
        "sentiment_pass": False,
        "eligibility_pass": False,
        "reason": reason,
        "price": None,
        "sma_50": None,
        "sentiment": None,
        "conviction_score": None,
    }


# =========================================================================== #
# OrderRequest — payload sent to Alpaca                                        #
# =========================================================================== #

class OrderRequest(TypedDict, total=False):
    """
    Paper-trade order specification sent to Alpaca via execution_engine.
    """
    ticker:        str
    side:          OrderSide          # "buy" | "sell"
    qty:           int                # integer shares
    order_type:    OrderType          # "market" | "limit" | ...
    time_in_force: TIF                # "day" | "gtc" | ...
    limit_price:   Optional[float]    # required for limit orders
    signal_reason: Optional[str]      # propagated from SignalDecision.reason


ORDER_REQUEST_REQUIRED = {"ticker", "side", "qty", "order_type", "time_in_force"}


def validate_order_request(req: dict, raise_on_error: bool = False) -> bool:
    missing = ORDER_REQUEST_REQUIRED - set(req.keys())
    if missing:
        msg = f"OrderRequest missing required fields: {missing}"
        if raise_on_error: raise ValueError(msg)
        logger.debug(msg); return False
    if req.get("side") not in VALID_SIDES:
        msg = f"OrderRequest: invalid side '{req.get('side')}'."
        if raise_on_error: raise ValueError(msg)
        logger.debug(msg); return False
    if not (isinstance(req.get("qty"), int) and req["qty"] > 0):
        msg = f"OrderRequest: qty must be a positive integer (got {req.get('qty')!r})."
        if raise_on_error: raise ValueError(msg)
        logger.debug(msg); return False
    return True


# =========================================================================== #
# OrderResult — response from Alpaca (or structured failure)                  #
# =========================================================================== #

class OrderResult(TypedDict, total=False):
    """
    Result of one order submission attempt.
    status = "submitted" | "filled" | "failed" | "rejected"
    """
    ticker:       str
    status:       str
    order_id:     Optional[str]
    submitted_at: Optional[datetime]
    side:         Optional[str]
    qty:          Optional[int]
    fill_price:   Optional[float]
    error:        Optional[str]       # set on failure/rejection


# =========================================================================== #
# PositionState — live position record                                         #
# =========================================================================== #

class PositionState(TypedDict, total=False):
    """
    In-flight position tracked by OrderMonitor.

    Each open long or short position has one PositionState entry.
    Used by risk_manager to compute stop-loss/take-profit levels.
    """
    ticker:          str
    side:            PositionSide     # "long" | "short"
    qty:             int
    entry_price:     float
    stop_loss_price: float            # absolute stop level (pre-calculated)
    take_profit_price: Optional[float]
    entered_at:      datetime         # UTC — time of fill / order submission
    order_id:        Optional[str]
    status:          str              # "open" | "closed" | "pending"


def make_position_state(
    ticker: str,
    side: str,
    qty: int,
    entry_price: float,
    stop_loss_price: float,
    order_id: Optional[str] = None,
    take_profit_price: Optional[float] = None,
    entered_at: Optional[datetime] = None,
) -> dict:
    """Construct a PositionState dict."""
    from datetime import timezone
    return {
        "ticker":            ticker,
        "side":              side,
        "qty":               qty,
        "entry_price":       entry_price,
        "stop_loss_price":   stop_loss_price,
        "take_profit_price": take_profit_price,
        "entered_at":        entered_at or datetime.now(tz=timezone.utc),
        "order_id":          order_id,
        "status":            "open",
    }


# =========================================================================== #
# RiskCheckResult — output of risk_manager evaluation                         #
# =========================================================================== #

class RiskCheckResult(TypedDict, total=False):
    """
    Structured output of one risk evaluation.

    passed = True means the trade or position survived all risk checks.
    """
    ticker:     str
    passed:     bool
    reason:     str
    action:     Optional[str]   # e.g. "EXIT" if risk forces a close
    stop_hit:   bool
    tp_hit:     bool
    max_pos_hit: bool
