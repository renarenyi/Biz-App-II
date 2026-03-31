"""
strategy/eligibility.py
------------------------
Pre-order eligibility checks for Phase 3.

This module answers: "Even if the signal says BUY, are we allowed to act?"

Checks (in evaluation order)
-----------------------------
1. Market is open
2. No existing open position for this ticker (duplicate guard)
3. Signal is fresh (decision age)
4. Ticker is tradable (basic sanity — does not call Alpaca)
5. Position size is non-zero (computed externally, passed in)
6. Buying power is sufficient for the computed size
7. Max concurrent positions limit not exceeded

All checks are pure functions with explicit inputs.
No side effects; no API calls.  The caller (strategy_engine) is responsible
for querying Alpaca for account state before calling these checks.

Design notes
------------
- Each check returns a (bool, str) tuple: (passed, reason_message)
- The composite check_all_eligibility() returns an EligibilityResult dict
- Blocking conditions are checked in priority order; first failure short-circuits
- This module MUST NOT import from execution_engine.py (no circular deps)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Maximum age of a signal decision before it is considered stale
DEFAULT_DECISION_MAX_AGE_SECONDS: float = 60.0
# Default max concurrent open positions across all tickers
DEFAULT_MAX_CONCURRENT_POSITIONS: int = 5


# =========================================================================== #
# Composite eligibility check                                                  #
# =========================================================================== #

def check_all_eligibility(
    ticker: str,
    signal: str,
    qty: int,
    price: float,
    is_market_open: bool,
    open_positions: dict[str, dict],
    buying_power: float,
    signal_timestamp: Optional[datetime] = None,
    decision_max_age_seconds: float = DEFAULT_DECISION_MAX_AGE_SECONDS,
    max_concurrent_positions: int = DEFAULT_MAX_CONCURRENT_POSITIONS,
    reference_time: Optional[datetime] = None,
) -> dict:
    """
    Run all eligibility checks for a signal.

    Parameters
    ----------
    ticker : str
    signal : str
        "BUY" | "EXIT" | "HOLD" | "NO_ACTION"
    qty : int
        Computed order quantity.  Pass 0 if sizing returned nothing.
    price : float
        Current price — used to estimate order cost.
    is_market_open : bool
    open_positions : dict[str, dict]
        Mapping of ticker → PositionState for currently open positions.
    buying_power : float
        Available buying power from Alpaca account.
    signal_timestamp : datetime, optional
        UTC timestamp of when the SignalDecision was made.
    decision_max_age_seconds : float
    max_concurrent_positions : int
    reference_time : datetime, optional
        Anchor for freshness checks.  Defaults to now (UTC).

    Returns
    -------
    dict with keys:
        passed:  bool
        reason:  str
        checks:  dict[str, bool]  — individual check results
    """
    ref = reference_time or datetime.now(tz=timezone.utc)
    checks: dict[str, bool] = {}

    # HOLD / NO_ACTION always pass eligibility — nothing to check
    if signal in ("HOLD", "NO_ACTION"):
        return _pass("Signal is HOLD/NO_ACTION — no order needed.", checks)

    # ── 1. Market open ────────────────────────────────────────────────── #
    ok, reason = check_market_open(is_market_open)
    checks["market_open"] = ok
    if not ok:
        return _fail(reason, checks)

    # ── 2. Duplicate position guard (BUY only) ────────────────────────── #
    if signal == "BUY":
        ok, reason = check_no_duplicate_position(ticker, open_positions)
        checks["no_duplicate_position"] = ok
        if not ok:
            return _fail(reason, checks)

    # ── 3. Decision freshness ─────────────────────────────────────────── #
    ok, reason = check_decision_freshness(signal_timestamp, decision_max_age_seconds, ref)
    checks["decision_fresh"] = ok
    if not ok:
        return _fail(reason, checks)

    # ── 4. Quantity is valid ──────────────────────────────────────────── #
    ok, reason = check_valid_quantity(qty)
    checks["valid_quantity"] = ok
    if not ok:
        return _fail(reason, checks)

    # ── 5. Buying power sufficient (BUY only) ────────────────────────── #
    if signal == "BUY":
        ok, reason = check_buying_power(qty, price, buying_power)
        checks["buying_power"] = ok
        if not ok:
            return _fail(reason, checks)

    # ── 6. Max concurrent positions (BUY only) ───────────────────────── #
    if signal == "BUY":
        ok, reason = check_max_positions(open_positions, max_concurrent_positions)
        checks["max_positions"] = ok
        if not ok:
            return _fail(reason, checks)

    return _pass("All eligibility checks passed.", checks)


# =========================================================================== #
# Individual checks (exposed for unit testing)                                 #
# =========================================================================== #

def check_market_open(is_market_open: bool) -> tuple[bool, str]:
    if is_market_open:
        return True, "Market is open."
    return False, "Market is closed — order blocked."


def check_no_duplicate_position(
    ticker: str,
    open_positions: dict[str, dict],
) -> tuple[bool, str]:
    pos = open_positions.get(ticker)
    if pos is None or pos.get("status") != "open":
        return True, f"No open position for {ticker}."
    return False, f"Duplicate position blocked: {ticker} already has an open long."


def check_decision_freshness(
    signal_timestamp: Optional[datetime],
    max_age_seconds: float,
    ref: datetime,
) -> tuple[bool, str]:
    if signal_timestamp is None:
        return True, "No timestamp on signal — allowing (cannot evaluate staleness)."
    try:
        ts = signal_timestamp
        if hasattr(ts, "tzinfo") and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_s = (ref - ts).total_seconds()
        if age_s <= max_age_seconds:
            return True, f"Signal is fresh ({age_s:.1f}s old)."
        return False, f"Signal is stale ({age_s:.1f}s old > max {max_age_seconds:.0f}s)."
    except Exception as exc:
        logger.debug("check_decision_freshness error: %s", exc)
        return True, "Could not evaluate signal freshness — allowing."


def check_valid_quantity(qty: int) -> tuple[bool, str]:
    if isinstance(qty, int) and qty > 0:
        return True, f"Quantity {qty} is valid."
    return False, f"Invalid order quantity: {qty!r} — order blocked."


def check_buying_power(qty: int, price: float, buying_power: float) -> tuple[bool, str]:
    cost = qty * price
    if buying_power >= cost:
        return True, f"Buying power ${buying_power:.2f} ≥ order cost ${cost:.2f}."
    return False, (
        f"Insufficient buying power: ${buying_power:.2f} < "
        f"estimated cost ${cost:.2f} ({qty} × ${price:.2f})."
    )


def check_max_positions(
    open_positions: dict[str, dict],
    max_positions: int,
) -> tuple[bool, str]:
    current = sum(
        1 for p in open_positions.values() if p.get("status") == "open"
    )
    if current < max_positions:
        return True, f"Position count {current} < limit {max_positions}."
    return False, f"Max concurrent positions reached ({current}/{max_positions})."


# =========================================================================== #
# Result builders                                                              #
# =========================================================================== #

def _pass(reason: str, checks: dict) -> dict:
    return {"passed": True, "reason": reason, "checks": checks}


def _fail(reason: str, checks: dict) -> dict:
    return {"passed": False, "reason": reason, "checks": checks}
