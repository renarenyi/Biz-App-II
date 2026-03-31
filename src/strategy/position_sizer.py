"""
strategy/position_sizer.py
---------------------------
Position sizing for Phase 3.

Principles
----------
- Simple, auditable, deterministic
- Returns an INTEGER number of shares (fractional shares not supported)
- Returns 0 if the computed size would be 0, negative, or exceed safety limits
- Caller should block the order if qty == 0

Two sizing methods are provided:

  1. fixed_dollar_size(price, dollar_amount)
       allocate a fixed dollar amount, e.g. always risk $1,000 per trade

  2. percent_of_equity(price, equity, fraction)
       allocate a fixed fraction of the portfolio, e.g. 5% of $20,000 = $1,000

Both methods check that the computed quantity is ≥ 1 and that the order cost
does not exceed a safety cap (MAX_POSITION_FRACTION of portfolio).

Design note
-----------
More sophisticated sizing (volatility-adjusted Kelly sizing, ATR-based position
sizing) belongs in Phase 4 extensions.  For now, simple = auditable = correct.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

# Hard cap: no single position > 20% of portfolio equity
MAX_POSITION_FRACTION: float = 0.20
DEFAULT_FIXED_DOLLAR:  float = 1_000.0
DEFAULT_EQUITY_FRACTION: float = 0.05   # 5% of portfolio per trade


# =========================================================================== #
# Public sizing functions                                                      #
# =========================================================================== #

def fixed_dollar_size(
    price: float,
    dollar_amount: float = DEFAULT_FIXED_DOLLAR,
    portfolio_equity: Optional[float] = None,
) -> int:
    """
    Compute integer shares for a fixed dollar allocation.

    Parameters
    ----------
    price : float
        Latest close / ask price.
    dollar_amount : float
        Target dollar spend.
    portfolio_equity : float, optional
        If provided, the position is capped at MAX_POSITION_FRACTION of equity.

    Returns
    -------
    int
        Number of shares.  Returns 0 if quantity rounds to zero or cap is hit.
    """
    if price <= 0 or dollar_amount <= 0:
        logger.debug("position_sizer.fixed_dollar: invalid inputs (price=%.2f, amount=%.2f).", price, dollar_amount)
        return 0

    # Apply equity cap if portfolio size is known
    if portfolio_equity and portfolio_equity > 0:
        max_dollars = portfolio_equity * MAX_POSITION_FRACTION
        dollar_amount = min(dollar_amount, max_dollars)

    qty = int(math.floor(dollar_amount / price))
    if qty <= 0:
        logger.debug(
            "position_sizer.fixed_dollar: computed qty=0 (price=%.2f, amount=%.2f).",
            price, dollar_amount,
        )
        return 0

    logger.debug(
        "position_sizer.fixed_dollar: qty=%d (price=%.2f, amount=%.2f).",
        qty, price, dollar_amount,
    )
    return qty


def percent_of_equity(
    price: float,
    equity: float,
    fraction: float = DEFAULT_EQUITY_FRACTION,
) -> int:
    """
    Compute integer shares as a fraction of current portfolio equity.

    Parameters
    ----------
    price : float
        Latest close / ask price.
    equity : float
        Total portfolio equity (cash + market value of positions).
    fraction : float
        Portfolio fraction to allocate (e.g. 0.05 = 5%).

    Returns
    -------
    int
        Number of shares.  Returns 0 if quantity rounds to zero.
    """
    if price <= 0 or equity <= 0 or fraction <= 0:
        logger.debug(
            "position_sizer.pct_equity: invalid inputs (price=%.2f, equity=%.2f, frac=%.3f).",
            price, equity, fraction,
        )
        return 0

    fraction = min(fraction, MAX_POSITION_FRACTION)
    dollar_amount = equity * fraction
    qty = int(math.floor(dollar_amount / price))

    if qty <= 0:
        logger.debug(
            "position_sizer.pct_equity: computed qty=0 (price=%.2f, equity=%.2f, frac=%.3f).",
            price, equity, fraction,
        )
        return 0

    logger.debug(
        "position_sizer.pct_equity: qty=%d (price=%.2f, equity=%.2f, frac=%.3f, $=%.2f).",
        qty, price, equity, fraction, dollar_amount,
    )
    return qty
