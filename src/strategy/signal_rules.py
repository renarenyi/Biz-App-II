"""
strategy/signal_rules.py
-------------------------
Deterministic trading rules for the Phase 3 Strategy Engine.

This module contains ONLY signal logic — no API calls, no state mutations.
The same rules are used by both:
  - live paper execution (strategy_engine.py)
  - historical backtesting (Phase 4)

This separation is critical.  Do NOT import execution_engine or order_monitor
from this module.

Rules implemented
-----------------
  evaluate_entry()  → SignalDecision (BUY or NO_ACTION)
  evaluate_exit()   → SignalDecision (EXIT or HOLD)
  evaluate_hold()   → SignalDecision (HOLD)

Rule parameters
---------------
All thresholds are explicit keyword arguments with documented defaults.
This makes the rules testable in isolation and easy to sweep in backtests.

Entry rule (all conditions must be True):
  1. close > sma_50                              (trend filter)
  2. sentiment == "POSITIVE"                     (sentiment gate)
  3. conviction_score > CONVICTION_THRESHOLD     (signal strength gate)
  4. sentiment not stale                         (freshness gate)
  5. market is open                              (session gate)
  6. no existing open long position for ticker   (duplicate guard — checked by caller)

Exit rule (any condition triggers EXIT):
  A. close <= stop_loss_price                    (stop-loss hit)
  B. conviction_score < NEG_CONVICTION_THRESHOLD AND sentiment == "NEGATIVE"
                                                  (sentiment reversal)
  C. close < sma_50 AND trend_exit_enabled        (trend failure)
  D. [Phase 4 extension: max holding period]

No-action rule:
  - Missing required market or sentiment inputs
  - Stale sentiment
  - Market closed and no position to monitor
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.strategy.schemas import (
    InputMarketSnapshot,
    InputSentimentSnapshot,
    PositionState,
    SignalDecision,
    VALID_SIGNALS,
    make_no_action,
)

logger = logging.getLogger(__name__)

# =========================================================================== #
# Default thresholds (all overridable at call time)                           #
# =========================================================================== #

DEFAULT_CONVICTION_THRESHOLD:     float = 7.0   # min conviction to enter
DEFAULT_NEG_CONVICTION_THRESHOLD: float = 7.5   # conviction to force exit on neg sentiment
DEFAULT_SENTIMENT_MAX_AGE_HOURS:  float = 24.0  # sentiment expires after this many hours


# =========================================================================== #
# Entry rule                                                                   #
# =========================================================================== #

def evaluate_entry(
    market: dict,
    sentiment: dict,
    conviction_threshold: float = DEFAULT_CONVICTION_THRESHOLD,
    sentiment_max_age_hours: float = DEFAULT_SENTIMENT_MAX_AGE_HOURS,
    reference_time: Optional[datetime] = None,
) -> dict:
    """
    Evaluate whether entry conditions are met for a long position.

    Does NOT check for duplicate open positions — that is eligibility's job.
    Does NOT check buying power — that is eligibility's job.
    This function is purely rule logic.

    Parameters
    ----------
    market : dict (InputMarketSnapshot)
    sentiment : dict (InputSentimentSnapshot)
    conviction_threshold : float
    sentiment_max_age_hours : float
    reference_time : datetime, optional
        Anchor for staleness checks.  Defaults to now (UTC).

    Returns
    -------
    dict (SignalDecision)
        signal = "BUY" if all conditions pass, else "NO_ACTION"
    """
    ref = reference_time or datetime.now(tz=timezone.utc)
    ticker = market.get("ticker", "UNKNOWN")
    ts = ref

    # ── Guard: required market fields ────────────────────────────────── #
    close = market.get("close")
    sma_50 = market.get("sma_50")
    is_open = market.get("is_market_open", False)

    if close is None or close <= 0:
        return make_no_action(ticker, "Missing or invalid close price.", ts)

    if not is_open:
        return make_no_action(ticker, "Market is closed.", ts)

    if sma_50 is None:
        return make_no_action(ticker, "SMA-50 unavailable — cannot evaluate trend filter.", ts)

    # ── Guard: required sentiment fields ─────────────────────────────── #
    if not _sentiment_is_valid(sentiment):
        return make_no_action(ticker, "Sentiment snapshot missing required fields.", ts)

    sent_label    = sentiment.get("sentiment", "NEUTRAL")
    conviction    = float(sentiment.get("conviction_score", 0))
    generated_at  = sentiment.get("generated_at")

    # ── Rule 4: Freshness ─────────────────────────────────────────────── #
    if not _sentiment_is_fresh(generated_at, sentiment_max_age_hours, ref):
        return make_no_action(
            ticker,
            f"Sentiment signal is stale (older than {sentiment_max_age_hours:.0f}h).",
            ts,
        )

    # ── Evaluate each condition ───────────────────────────────────────── #
    trend_pass     = bool(close > sma_50)
    sentiment_pass = (sent_label == "POSITIVE")
    conviction_pass = (conviction > conviction_threshold)

    # RSI overbought filter (optional — skipped if not present)
    rsi = market.get("rsi_14")
    rsi_pass = True  # default: pass if no RSI data
    if rsi is not None:
        rsi_pass = (float(rsi) <= 70.0)

    # ADX trend quality filter (optional — skipped if not present)
    adx = market.get("adx_14")
    adx_pass = True  # default: pass if no ADX data
    if adx is not None:
        adx_pass = (float(adx) >= 20.0)  # ADX < 20 = choppy market, avoid

    # Rolling sentiment context (optional — skipped if not present)
    sent_rolling = sentiment.get("sentiment_rolling")
    rolling_pass = True  # default: pass if no rolling data
    if sent_rolling is not None:
        rolling_pass = (float(sent_rolling) > 0.0)  # recent sentiment trend must be positive

    technical_pass = trend_pass and rsi_pass and adx_pass
    all_pass = technical_pass and sentiment_pass and conviction_pass and rolling_pass

    # ── Build reasons ─────────────────────────────────────────────────── #
    if all_pass:
        rsi_part = f" RSI={rsi:.1f};" if rsi is not None else ""
        adx_part = f" ADX={adx:.1f};" if adx is not None else ""
        roll_part = f" roll_sent={sent_rolling:.2f};" if sent_rolling is not None else ""
        reason = (
            f"All entry conditions met: "
            f"close {close:.2f} > SMA-50 {sma_50:.2f};{rsi_part}{adx_part}{roll_part} "
            f"sentiment={sent_label}; "
            f"conviction={conviction:.1f} > threshold {conviction_threshold:.0f}."
        )
        signal = "BUY"
    else:
        failed = []
        if not trend_pass:
            failed.append(f"price ({close:.2f}) ≤ SMA-50 ({sma_50:.2f})")
        if not rsi_pass:
            failed.append(f"RSI ({rsi:.1f}) > 70 — overbought")
        if not adx_pass:
            failed.append(f"ADX ({adx:.1f}) < 20 — choppy market")
        if not sentiment_pass:
            failed.append(f"sentiment is {sent_label} (need POSITIVE)")
        if not conviction_pass:
            failed.append(f"conviction {conviction:.1f} ≤ threshold {conviction_threshold:.0f}")
        if not rolling_pass:
            failed.append(f"rolling sentiment ({sent_rolling:.2f}) ≤ 0 — recent context negative")
        reason = "Entry blocked: " + "; ".join(failed) + "."
        signal = "NO_ACTION"

    return {
        "ticker":           ticker,
        "timestamp":        ts,
        "signal":           signal,
        "technical_pass":   technical_pass,
        "sentiment_pass":   sentiment_pass and conviction_pass,
        "eligibility_pass": None,   # set downstream by eligibility layer
        "reason":           reason,
        "price":            close,
        "sma_50":           sma_50,
        "sentiment":        sent_label,
        "conviction_score": conviction,
    }


# =========================================================================== #
# Exit rule                                                                    #
# =========================================================================== #

def evaluate_exit(
    market: dict,
    sentiment: dict,
    position: dict,
    neg_conviction_threshold: float = DEFAULT_NEG_CONVICTION_THRESHOLD,
    trend_exit_enabled: bool = True,
    sentiment_max_age_hours: float = DEFAULT_SENTIMENT_MAX_AGE_HOURS,
    reference_time: Optional[datetime] = None,
) -> dict:
    """
    Evaluate whether an open position should be exited.

    Parameters
    ----------
    market : dict (InputMarketSnapshot)
    sentiment : dict (InputSentimentSnapshot)
    position : dict (PositionState)
        Must contain: entry_price, stop_loss_price.
    neg_conviction_threshold : float
        Conviction above which a NEGATIVE sentiment forces exit.
    trend_exit_enabled : bool
        Whether a break below SMA-50 triggers exit.
    sentiment_max_age_hours : float
    reference_time : datetime, optional

    Returns
    -------
    dict (SignalDecision)
        signal = "EXIT" if any exit condition triggers, else "HOLD"
    """
    ref = reference_time or datetime.now(tz=timezone.utc)
    ticker = market.get("ticker", "UNKNOWN")
    ts = ref

    close     = market.get("close")
    sma_50    = market.get("sma_50")
    stop_loss = position.get("stop_loss_price")

    if close is None or close <= 0:
        return _make_hold(ticker, ts, "Close price missing — hold conservatively.", market, sentiment)

    # ── Condition A: Stop-loss hit ────────────────────────────────────── #
    if stop_loss is not None and close <= stop_loss:
        return _make_exit(
            ticker, ts,
            f"Stop-loss hit: close {close:.2f} ≤ stop {stop_loss:.2f}.",
            market, sentiment, stop_hit=True,
        )

    # ── Condition B: Sentiment reversal ──────────────────────────────── #
    if _sentiment_is_valid(sentiment) and _sentiment_is_fresh(
        sentiment.get("generated_at"), sentiment_max_age_hours, ref
    ):
        sent_label = sentiment.get("sentiment", "NEUTRAL")
        conviction = float(sentiment.get("conviction_score", 0))
        if sent_label == "NEGATIVE" and conviction >= neg_conviction_threshold:
            return _make_exit(
                ticker, ts,
                f"Sentiment reversed: {sent_label} conviction {conviction:.1f} ≥ threshold {neg_conviction_threshold:.0f}.",
                market, sentiment,
            )

    # ── Condition C: Trend failure ────────────────────────────────────── #
    if trend_exit_enabled and sma_50 is not None and close < sma_50:
        return _make_exit(
            ticker, ts,
            f"Trend filter failed: close {close:.2f} < SMA-50 {sma_50:.2f}.",
            market, sentiment,
        )

    # ── No exit condition triggered ───────────────────────────────────── #
    return _make_hold(
        ticker, ts,
        "No exit conditions triggered — maintaining position.",
        market, sentiment,
    )


# =========================================================================== #
# Helpers                                                                      #
# =========================================================================== #

def _make_exit(
    ticker: str,
    ts: datetime,
    reason: str,
    market: dict,
    sentiment: dict,
    stop_hit: bool = False,
) -> dict:
    return {
        "ticker":           ticker,
        "timestamp":        ts,
        "signal":           "EXIT",
        "technical_pass":   False,
        "sentiment_pass":   False,
        "eligibility_pass": True,
        "reason":           reason,
        "price":            market.get("close"),
        "sma_50":           market.get("sma_50"),
        "sentiment":        sentiment.get("sentiment"),
        "conviction_score": sentiment.get("conviction_score"),
    }


def _make_hold(
    ticker: str,
    ts: datetime,
    reason: str,
    market: dict,
    sentiment: dict,
) -> dict:
    return {
        "ticker":           ticker,
        "timestamp":        ts,
        "signal":           "HOLD",
        "technical_pass":   True,
        "sentiment_pass":   True,
        "eligibility_pass": True,
        "reason":           reason,
        "price":            market.get("close"),
        "sma_50":           market.get("sma_50"),
        "sentiment":        sentiment.get("sentiment"),
        "conviction_score": sentiment.get("conviction_score"),
    }


def _sentiment_is_valid(sentiment: dict) -> bool:
    """Return True if sentiment dict has the minimum required fields."""
    required = {"sentiment", "conviction_score", "generated_at"}
    return required.issubset(sentiment.keys())


def _sentiment_is_fresh(
    generated_at: Optional[datetime],
    max_age_hours: float,
    ref: datetime,
) -> bool:
    """Return True if sentiment was generated within max_age_hours of ref."""
    if generated_at is None:
        return False
    try:
        gen = generated_at
        if hasattr(gen, "tzinfo") and gen.tzinfo is None:
            gen = gen.replace(tzinfo=timezone.utc)
        age_hours = (ref - gen).total_seconds() / 3600.0
        return age_hours <= max_age_hours
    except Exception:
        return False
