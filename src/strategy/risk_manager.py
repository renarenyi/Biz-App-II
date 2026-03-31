"""
strategy/risk_manager.py
-------------------------
Risk controls for Phase 3.

Responsibilities
----------------
1. Compute stop-loss and take-profit price levels for a new entry
2. Evaluate whether an open position should be force-exited on risk grounds
3. Enforce max position count and max exposure limits

This module is pure computation — no API calls, no logging side-effects.
All risk levels are pre-calculated at entry time and stored in PositionState.
This makes them deterministic and backtest-reusable.

Default risk parameters (all overridable)
------------------------------------------
STOP_LOSS_PCT       = 0.02   (2% below entry for long positions)
TAKE_PROFIT_PCT     = 0.04   (4% above entry — optional, disabled by default)
MAX_PORTFOLIO_RISK  = 0.10   (hard stop: no more than 10% total portfolio in risk)

Risk logic
----------
Entry:
  stop_loss_price  = entry_price * (1 - STOP_LOSS_PCT)
  take_profit_price = entry_price * (1 + TAKE_PROFIT_PCT) [if enabled]

Monitor:
  if close <= stop_loss_price  → EXIT (stop hit)
  if close >= take_profit_price → EXIT (take-profit hit) [if enabled]
  if conviction flips and sentiment is strongly negative → EXIT (signal reversal)

These are the same conditions as in signal_rules.evaluate_exit().
risk_manager.assess_position() is the numerical gateway; signal_rules is the
higher-level coordinator that also checks trend and sentiment.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# =========================================================================== #
# Default risk parameters                                                      #
# =========================================================================== #

DEFAULT_STOP_LOSS_PCT:    float = 0.02   # 2% trailing stop
DEFAULT_TAKE_PROFIT_PCT:  float = 0.04   # 4% take-profit
DEFAULT_TAKE_PROFIT_ENABLED: bool = False


# =========================================================================== #
# Entry: compute risk levels                                                   #
# =========================================================================== #

def compute_stop_loss(
    entry_price: float,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
    side: str = "long",
) -> float:
    """
    Compute the absolute stop-loss price for a new position.

    Parameters
    ----------
    entry_price : float
    stop_loss_pct : float
        Fraction below (long) or above (short) entry_price.
    side : str
        "long" | "short"

    Returns
    -------
    float
        Absolute stop-loss price level.
    """
    if entry_price <= 0 or stop_loss_pct <= 0:
        return 0.0

    if side == "long":
        level = entry_price * (1.0 - stop_loss_pct)
    else:  # short
        level = entry_price * (1.0 + stop_loss_pct)

    logger.debug(
        "risk_manager: stop_loss for %s entry=%.2f (%.0f%% stop) → %.2f",
        side, entry_price, stop_loss_pct * 100, level,
    )
    return round(level, 4)


def compute_take_profit(
    entry_price: float,
    take_profit_pct: float = DEFAULT_TAKE_PROFIT_PCT,
    side: str = "long",
    enabled: bool = DEFAULT_TAKE_PROFIT_ENABLED,
) -> Optional[float]:
    """
    Compute the absolute take-profit price level.

    Returns None if take-profit is disabled.
    """
    if not enabled or entry_price <= 0 or take_profit_pct <= 0:
        return None

    if side == "long":
        level = entry_price * (1.0 + take_profit_pct)
    else:
        level = entry_price * (1.0 - take_profit_pct)

    logger.debug(
        "risk_manager: take_profit for %s entry=%.2f (%.0f%% target) → %.2f",
        side, entry_price, take_profit_pct * 100, level,
    )
    return round(level, 4)


def trailing_stop_update(
    position: dict,
    current_high: float,
    trailing_pct: float = DEFAULT_STOP_LOSS_PCT,
) -> Optional[float]:
    """
    Move stop-loss up when price makes new highs. Never moves down.

    Parameters
    ----------
    position : dict (PositionState or TradeRecord)
        Must contain 'stop_loss_price' and 'entry_price'.
    current_high : float
        Highest price observed in current bar.
    trailing_pct : float
        Trailing stop percentage (e.g. 0.06 = 6%).

    Returns
    -------
    float | None
        New stop-loss price if ratcheted up, else None (no change).
    """
    if current_high <= 0 or trailing_pct <= 0:
        return None

    old_stop = float(position.get("stop_loss_price") or 0)
    new_stop = round(current_high * (1.0 - trailing_pct), 4)

    if new_stop > old_stop:
        logger.debug(
            "trailing_stop_update: ratchet stop %.4f → %.4f (high=%.2f, trail=%.1f%%)",
            old_stop, new_stop, current_high, trailing_pct * 100,
        )
        return new_stop
    return None


# =========================================================================== #
# Monitor: assess a live position                                              #
# =========================================================================== #

def assess_position(
    position: dict,
    current_price: float,
) -> dict:
    """
    Assess whether an open position has breached stop-loss or take-profit.

    Parameters
    ----------
    position : dict (PositionState)
        Must contain: stop_loss_price, and optionally take_profit_price.
    current_price : float
        Latest close or bid price.

    Returns
    -------
    dict (RiskCheckResult)
        passed = True if position is within bounds.
        action = "EXIT" if a limit was breached.
    """
    ticker      = position.get("ticker", "UNKNOWN")
    stop_loss   = position.get("stop_loss_price")
    take_profit = position.get("take_profit_price")
    side        = position.get("side", "long")

    stop_hit = False
    tp_hit   = False
    reason   = "Position within risk bounds."
    action   = None

    if current_price <= 0:
        return {
            "ticker": ticker, "passed": True,
            "reason": "Current price invalid — cannot assess.",
            "action": None, "stop_hit": False, "tp_hit": False, "max_pos_hit": False,
        }

    # Stop-loss check
    if stop_loss is not None:
        if side == "long" and current_price <= stop_loss:
            stop_hit = True
            reason = f"Stop-loss hit: price {current_price:.2f} ≤ stop {stop_loss:.2f}."
            action = "EXIT"

    # Take-profit check (only if no stop hit)
    if not stop_hit and take_profit is not None:
        if side == "long" and current_price >= take_profit:
            tp_hit = True
            reason = f"Take-profit reached: price {current_price:.2f} ≥ target {take_profit:.2f}."
            action = "EXIT"

    passed = not (stop_hit or tp_hit)

    result: dict = {
        "ticker":      ticker,
        "passed":      passed,
        "reason":      reason,
        "action":      action,
        "stop_hit":    stop_hit,
        "tp_hit":      tp_hit,
        "max_pos_hit": False,  # set separately by check_portfolio_risk
    }

    if not passed:
        logger.info("risk_manager.assess_position: %s — %s", ticker, reason)

    return result


# =========================================================================== #
# Portfolio-level risk checks                                                  #
# =========================================================================== #

def check_portfolio_risk(
    open_positions: dict[str, dict],
    portfolio_equity: float,
    max_portfolio_risk_pct: float = 0.10,
    new_position_cost: Optional[float] = None,
) -> dict:
    """
    Check whether total risk exposure is within portfolio limits.

    Parameters
    ----------
    open_positions : dict[str, dict]
        ticker → PositionState mapping.
    portfolio_equity : float
        Total portfolio equity.
    max_portfolio_risk_pct : float
        Maximum fraction of equity allowed at risk simultaneously.
    new_position_cost : float, optional
        Estimated cost of the new position being considered.

    Returns
    -------
    dict with keys: passed, reason, total_risk_dollars, max_risk_dollars
    """
    if portfolio_equity <= 0:
        return {
            "passed": False, "reason": "Portfolio equity is zero or invalid.",
            "total_risk_dollars": 0.0, "max_risk_dollars": 0.0,
        }

    # Total current risk = sum of (entry_price - stop_loss) * qty for all open longs
    total_risk = 0.0
    for pos in open_positions.values():
        if pos.get("status") != "open":
            continue
        entry = pos.get("entry_price", 0)
        stop  = pos.get("stop_loss_price", entry)
        qty   = pos.get("qty", 0)
        risk_per_share = max(entry - stop, 0)
        total_risk += risk_per_share * qty

    if new_position_cost is not None:
        total_risk += new_position_cost

    max_risk = portfolio_equity * max_portfolio_risk_pct

    if total_risk <= max_risk:
        return {
            "passed": True,
            "reason": f"Total risk ${total_risk:.2f} ≤ limit ${max_risk:.2f}.",
            "total_risk_dollars": total_risk,
            "max_risk_dollars": max_risk,
        }

    return {
        "passed": False,
        "reason": f"Portfolio risk limit exceeded: ${total_risk:.2f} > ${max_risk:.2f}.",
        "total_risk_dollars": total_risk,
        "max_risk_dollars": max_risk,
    }
