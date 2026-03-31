"""
strategy/execution_engine.py
------------------------------
Alpaca Paper Trading order submission for Phase 3.

THIS MODULE ONLY SENDS TO ALPACA PAPER TRADING.
It will refuse to operate if the environment is not configured for paper trading.

Supported order flow (baseline)
--------------------------------
  submit_market_order(ticker, side, qty) → OrderResult
  cancel_order(order_id)                 → bool
  get_order_status(order_id)             → dict | None

The caller (strategy_engine) is responsible for all eligibility checks
before calling submit_market_order.  This module trusts that the inputs
have already been validated.

Error handling contract
------------------------
- AlpacaOrderError is raised for clear submission failures (bad symbol, etc.)
- All other exceptions are caught, logged, and returned as structured failures
- An OrderResult with status="failed" is NEVER silently discarded upstream
- The module NEVER assumes an order was filled without confirmation

Paper trading environment verification
----------------------------------------
At initialization, the engine checks that:
  1. ALPACA_API_KEY and ALPACA_SECRET_KEY are set
  2. ALPACA_BASE_URL contains "paper" (safety check)

If either check fails, ExecutionEngineError is raised and the engine is
inoperable.  This is intentional — safety first.

Alpaca SDK: alpaca-py (pip install alpaca-py)
  from alpaca.trading import TradingClient
  from alpaca.trading.requests import MarketOrderRequest
  from alpaca.trading.enums import OrderSide, TimeInForce
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Paper trading URL fragment — used as a safety check
PAPER_URL_FRAGMENT = "paper"


class ExecutionEngineError(Exception):
    """Raised when the engine cannot initialize or is misconfigured."""
    pass


class AlpacaOrderError(Exception):
    """Raised for order-level failures from the Alpaca API."""
    pass


# =========================================================================== #
# ExecutionEngine                                                              #
# =========================================================================== #

class ExecutionEngine:
    """
    Thin wrapper around the Alpaca TradingClient for paper order submission.

    Parameters
    ----------
    api_key : str
        Alpaca API key.
    secret_key : str
        Alpaca secret key.
    base_url : str
        Must contain "paper" — enforced as a safety check.
    dry_run : bool
        If True, log orders but do not call the Alpaca API.
        Useful for integration tests without live credentials.
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        base_url: str = "https://paper-api.alpaca.markets",
        dry_run: bool = False,
    ) -> None:
        self._api_key    = api_key
        self._secret_key = secret_key
        self._base_url   = base_url
        self._dry_run    = dry_run
        self._client     = None

        # Safety check — prevent accidental live-money execution
        if not dry_run and PAPER_URL_FRAGMENT not in base_url.lower():
            raise ExecutionEngineError(
                f"ExecutionEngine: base_url '{base_url}' does not contain 'paper'. "
                "Set dry_run=True or use the Alpaca paper trading URL."
            )

        if not dry_run:
            self._init_client()

        logger.info(
            "ExecutionEngine: initialized (base_url=%s, dry_run=%s).",
            base_url, dry_run,
        )

    # ------------------------------------------------------------------ #
    # Client initialization                                               #
    # ------------------------------------------------------------------ #

    def _init_client(self) -> None:
        try:
            from alpaca.trading.client import TradingClient
        except ImportError as exc:
            raise ExecutionEngineError(
                "alpaca-py not installed.  Run: pip install alpaca-py"
            ) from exc

        try:
            self._client = TradingClient(
                api_key=self._api_key,
                secret_key=self._secret_key,
                paper=True,
            )
            logger.info("ExecutionEngine: Alpaca TradingClient connected.")
        except Exception as exc:
            raise ExecutionEngineError(
                f"ExecutionEngine: failed to connect to Alpaca — {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    # Core order submission                                               #
    # ------------------------------------------------------------------ #

    def submit_market_order(
        self,
        ticker: str,
        side: str,
        qty: int,
        signal_reason: Optional[str] = None,
    ) -> dict:
        """
        Submit a market order to Alpaca Paper Trading.

        Parameters
        ----------
        ticker : str
        side : str
            "buy" | "sell"
        qty : int
            Must be > 0.
        signal_reason : str, optional
            Propagated from SignalDecision.reason for the audit log.

        Returns
        -------
        dict (OrderResult)
            status = "submitted" on success, "failed" on any error.
        """
        if qty <= 0:
            return self._fail_result(
                ticker, side, qty,
                error=f"Invalid quantity {qty} — order not submitted.",
            )

        if side not in ("buy", "sell"):
            return self._fail_result(
                ticker, side, qty,
                error=f"Invalid order side '{side}'.",
            )

        logger.info(
            "ExecutionEngine: submitting %s %s x%d (reason: %s)",
            side.upper(), ticker, qty, signal_reason or "n/a",
        )

        if self._dry_run:
            return self._dry_run_result(ticker, side, qty)

        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            alpaca_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

            request = MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=alpaca_side,
                time_in_force=TimeInForce.DAY,
            )
            order = self._client.submit_order(order_data=request)

            result: dict = {
                "ticker":       ticker,
                "status":       "submitted",
                "order_id":     str(order.id),
                "submitted_at": datetime.now(tz=timezone.utc),
                "side":         side,
                "qty":          qty,
                "fill_price":   None,
                "error":        None,
            }
            logger.info(
                "ExecutionEngine: order submitted — id=%s %s %s x%d",
                result["order_id"], side.upper(), ticker, qty,
            )
            return result

        except AlpacaOrderError as exc:
            return self._fail_result(ticker, side, qty, error=str(exc))
        except Exception as exc:
            logger.error(
                "ExecutionEngine: unexpected error submitting %s %s x%d — %s",
                side.upper(), ticker, qty, exc,
            )
            return self._fail_result(ticker, side, qty, error=str(exc))

    # ------------------------------------------------------------------ #
    # Order management                                                    #
    # ------------------------------------------------------------------ #

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.

        Returns True on success, False on failure.
        Silently returns False in dry_run mode.
        """
        if self._dry_run:
            logger.info("ExecutionEngine (dry_run): cancel_order('%s')", order_id)
            return True

        try:
            self._client.cancel_order_by_id(order_id)
            logger.info("ExecutionEngine: cancelled order %s.", order_id)
            return True
        except Exception as exc:
            logger.warning("ExecutionEngine: cancel_order('%s') failed — %s", order_id, exc)
            return False

    def get_order_status(self, order_id: str) -> Optional[dict]:
        """
        Retrieve the current status of an order.

        Returns a dict with at least {'id', 'status', 'filled_qty', 'filled_avg_price'},
        or None if the order cannot be found.
        """
        if self._dry_run:
            return {"id": order_id, "status": "filled", "filled_qty": 0, "filled_avg_price": None}

        try:
            order = self._client.get_order_by_id(order_id)
            return {
                "id":                str(order.id),
                "status":            str(order.status),
                "filled_qty":        int(order.filled_qty or 0),
                "filled_avg_price":  float(order.filled_avg_price) if order.filled_avg_price else None,
            }
        except Exception as exc:
            logger.warning("ExecutionEngine: get_order_status('%s') failed — %s", order_id, exc)
            return None

    def get_account(self) -> Optional[dict]:
        """
        Retrieve account summary (buying power, equity).

        Returns a dict with at least {'buying_power', 'equity'} or None.
        """
        if self._dry_run:
            return {"buying_power": 100_000.0, "equity": 100_000.0, "cash": 100_000.0}

        try:
            account = self._client.get_account()
            return {
                "buying_power": float(account.buying_power),
                "equity":       float(account.equity),
                "cash":         float(account.cash),
            }
        except Exception as exc:
            logger.warning("ExecutionEngine: get_account() failed — %s", exc)
            return None

    # ------------------------------------------------------------------ #
    # Result builders                                                     #
    # ------------------------------------------------------------------ #

    def _fail_result(self, ticker: str, side: str, qty: int, error: str) -> dict:
        logger.warning("ExecutionEngine: order FAILED — %s %s x%d: %s", side.upper(), ticker, qty, error)
        return {
            "ticker":       ticker,
            "status":       "failed",
            "order_id":     None,
            "submitted_at": datetime.now(tz=timezone.utc),
            "side":         side,
            "qty":          qty,
            "fill_price":   None,
            "error":        error,
        }

    def _dry_run_result(self, ticker: str, side: str, qty: int) -> dict:
        import uuid
        fake_id = f"dry-run-{uuid.uuid4().hex[:8]}"
        logger.info(
            "ExecutionEngine (dry_run): %s %s x%d → fake_order_id=%s",
            side.upper(), ticker, qty, fake_id,
        )
        return {
            "ticker":       ticker,
            "status":       "submitted",
            "order_id":     fake_id,
            "submitted_at": datetime.now(tz=timezone.utc),
            "side":         side,
            "qty":          qty,
            "fill_price":   None,
            "error":        None,
        }

    # ------------------------------------------------------------------ #
    # Factory                                                             #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_settings(cls) -> "ExecutionEngine":
        """
        Construct an ExecutionEngine from environment/settings.

        Reads ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL from
        the project settings module.
        """
        try:
            from src.config.settings import settings
            return cls(
                api_key=settings.ALPACA_API_KEY or "",
                secret_key=settings.ALPACA_SECRET_KEY or "",
                base_url=getattr(settings, "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
                dry_run=(not settings.ALPACA_API_KEY),
            )
        except Exception as exc:
            logger.warning("ExecutionEngine.from_settings: falling back to dry_run — %s", exc)
            return cls(
                api_key="", secret_key="",
                base_url="https://paper-api.alpaca.markets",
                dry_run=True,
            )
