"""
backtest/schemas.py
--------------------
Typed data contracts for Phase 4 backtesting and validation framework.

All dicts flowing through the backtester conform to one of these schemas.
Use validate_* helpers at module boundaries; internal functions may trust
already-validated data.

Time convention
---------------
All timestamps must be timezone-aware UTC datetimes or ISO-8601 strings.
Never use naive datetimes.

Decision convention
-------------------
Signals are computed AFTER market close on day T using that day's closing
price and any news available before close.  Orders are FILLED at the OPEN
of day T+1.  This eliminates same-bar look-ahead bias for daily backtests.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


# =========================================================================== #
# Input schemas                                                                #
# =========================================================================== #

class HistoricalMarketRow(dict):
    """
    One OHLCV bar with derived indicators.

    Required keys
    -------------
    timestamp : datetime (UTC-aware) or ISO-8601 str
    ticker    : str
    open      : float
    high      : float
    low       : float
    close     : float
    volume    : int | float

    Optional keys
    -------------
    sma_50           : float | None
    returns          : float | None  (1-period log return)
    rolling_vol_20d  : float | None  (annualised 20-day realised vol)
    is_market_open   : bool           (defaults True)
    """


class HistoricalSentimentSnapshot(dict):
    """
    Ticker-level sentiment at a specific historical timestamp.

    Required keys
    -------------
    timestamp        : datetime (UTC-aware) or ISO-8601 str
    ticker           : str
    sentiment        : "POSITIVE" | "NEGATIVE" | "NEUTRAL"
    conviction_score : float  (0.0 – 10.0)

    Optional keys
    -------------
    reasoning            : str | None
    source_count         : int  (default 0)
    analysis_window_hours: float  (default 24)
    generated_at         : datetime | None
    """


# =========================================================================== #
# Backtest configuration                                                       #
# =========================================================================== #

class BacktestConfig(dict):
    """
    Runtime parameters for one backtest run.

    Required keys
    -------------
    tickers          : list[str]
    start_date       : datetime (UTC-aware) | str (YYYY-MM-DD)
    end_date         : datetime (UTC-aware) | str (YYYY-MM-DD)
    initial_capital  : float

    Optional keys — strategy parameters
    ------------------------------------
    conviction_threshold       : float  (default 7.0)
    sentiment_max_age_hours    : float  (default 24.0)
    stop_loss_pct              : float  (default 0.02)
    take_profit_pct            : float  (default 0.04)
    take_profit_enabled        : bool   (default False)
    equity_fraction            : float  (default 0.05)
    max_concurrent_positions   : int    (default 5)
    trend_exit_enabled         : bool   (default True)
    neg_conviction_threshold   : float  (default 6.0)

    Optional keys — execution / cost assumptions
    ---------------------------------------------
    slippage_pct               : float  (default 0.0, fraction of price)
    commission_per_trade       : float  (default 0.0, flat dollar)
    fill_model                 : str    ("next_open" | "same_close")
                                         default "next_open"

    Optional keys — misc
    --------------------
    strategy_name              : str    (default "sentiment_ma50_long_only")
    benchmark_ticker           : str    (default "SPY")
    """


# =========================================================================== #
# Trade records                                                                #
# =========================================================================== #

class TradeRecord(dict):
    """
    Complete record of one round-trip trade.

    Keys
    ----
    trade_id    : int
    ticker      : str
    side        : "long"
    entry_time  : datetime (UTC)
    entry_price : float
    exit_time   : datetime (UTC) | None   (None if still open at end)
    exit_price  : float | None
    qty         : int
    stop_loss_price : float
    exit_reason : str | None  ("stop_loss"|"sentiment_reversal"|"trend_exit"|"end_of_period")
    pnl         : float | None
    return_pct  : float | None
    holding_days: float | None
    """


# =========================================================================== #
# Portfolio snapshot                                                           #
# =========================================================================== #

class PortfolioSnapshot(dict):
    """
    Mark-to-market portfolio state at one timestamp.

    Keys
    ----
    timestamp      : datetime (UTC)
    cash           : float
    equity         : float   (cash + mark-to-market position value)
    gross_exposure : float
    net_exposure   : float
    open_positions : int
    """


# =========================================================================== #
# Backtest / benchmark results                                                 #
# =========================================================================== #

class BacktestResult(dict):
    """
    Aggregate performance summary for one backtest run.

    Required keys (all computed by metrics.py)
    -------------------------------------------
    strategy_name         : str
    start_date            : str  (YYYY-MM-DD)
    end_date              : str  (YYYY-MM-DD)
    initial_capital       : float
    final_equity          : float
    total_return          : float  (fraction, e.g. 0.124 = 12.4%)
    annualised_return     : float | None
    max_drawdown          : float  (negative, e.g. -0.061)
    sharpe_ratio          : float | None
    win_rate              : float  (fraction)
    trade_count           : int
    avg_win               : float | None  (avg PnL of winning trades)
    avg_loss              : float | None  (avg PnL of losing trades)
    profit_factor         : float | None  (gross_profit / gross_loss)
    avg_holding_days      : float | None
    exposure_pct          : float | None  (% of days with an open position)
    benchmark_total_return: float | None
    benchmark_sharpe      : float | None

    Attached sub-objects
    --------------------
    trades     : list[TradeRecord]
    equity_curve : list[PortfolioSnapshot]
    config     : BacktestConfig
    """


class BenchmarkResult(dict):
    """
    SPY (or other ticker) buy-and-hold over the backtest window.

    Keys
    ----
    ticker           : str
    start_date       : str
    end_date         : str
    initial_capital  : float
    final_equity     : float
    total_return     : float
    annualised_return: float | None
    max_drawdown     : float
    sharpe_ratio     : float | None
    equity_curve     : list[dict]   [{timestamp, equity}]
    """


# =========================================================================== #
# Validators                                                                   #
# =========================================================================== #

def validate_market_row(row: dict) -> bool:
    """Return True only if row has all required numeric fields."""
    required = ("timestamp", "ticker", "open", "high", "low", "close", "volume")
    for key in required:
        if key not in row or row[key] is None:
            return False
    try:
        float(row["close"])
        float(row["open"])
    except (TypeError, ValueError):
        return False
    return True


def validate_sentiment_snapshot(snap: dict) -> bool:
    """Return True only if snap has required typed fields."""
    if not snap:
        return False
    if snap.get("sentiment") not in ("POSITIVE", "NEGATIVE", "NEUTRAL"):
        return False
    try:
        score = float(snap.get("conviction_score", -1))
        if not (0.0 <= score <= 10.0):
            return False
    except (TypeError, ValueError):
        return False
    return True


def validate_backtest_config(cfg: dict) -> tuple[bool, str]:
    """
    Return (True, "") if config is valid, else (False, reason).
    """
    if not cfg.get("tickers"):
        return False, "tickers must be a non-empty list"
    if cfg.get("initial_capital", 0) <= 0:
        return False, "initial_capital must be positive"
    if "start_date" not in cfg or "end_date" not in cfg:
        return False, "start_date and end_date are required"
    return True, ""


# =========================================================================== #
# Factory helpers                                                              #
# =========================================================================== #

def default_config(**overrides) -> BacktestConfig:
    """
    Return a BacktestConfig with sensible defaults.

    Usage
    -----
    cfg = default_config(
        tickers=["TSLA"],
        start_date="2025-03-17",
        end_date="2026-03-17",
        initial_capital=100_000.0,
    )
    """
    cfg: BacktestConfig = {
        # Required — caller must supply
        "tickers":          [],
        "start_date":       None,
        "end_date":         None,
        "initial_capital":  100_000.0,

        # Strategy parameters (mirror Phase 3 defaults)
        "conviction_threshold":     7.0,
        "sentiment_max_age_hours":  24.0,
        "stop_loss_pct":            0.02,
        "take_profit_pct":          0.04,
        "take_profit_enabled":      False,
        "equity_fraction":          0.05,
        "max_concurrent_positions": 5,
        "trend_exit_enabled":       True,
        "neg_conviction_threshold": 6.0,

        # Execution / cost assumptions
        "slippage_pct":          0.0,
        "commission_per_trade":  0.0,
        "fill_model":            "next_open",

        # Misc
        "strategy_name":    "sentiment_ma50_long_only",
        "benchmark_ticker": "SPY",
    }
    cfg.update(overrides)
    return cfg


def make_trade_record(
    trade_id: int,
    ticker: str,
    entry_time: datetime,
    entry_price: float,
    qty: int,
    stop_loss_price: float,
) -> TradeRecord:
    """Return an open (incomplete) TradeRecord."""
    return {
        "trade_id":        trade_id,
        "ticker":          ticker,
        "side":            "long",
        "entry_time":      entry_time,
        "entry_price":     entry_price,
        "exit_time":       None,
        "exit_price":      None,
        "qty":             qty,
        "stop_loss_price": stop_loss_price,
        "exit_reason":     None,
        "pnl":             None,
        "return_pct":      None,
        "holding_days":    None,
    }


def close_trade_record(
    record: TradeRecord,
    exit_time: datetime,
    exit_price: float,
    exit_reason: str,
) -> TradeRecord:
    """Fill in exit fields and compute PnL."""
    entry = record["entry_price"]
    qty   = record["qty"]
    pnl   = (exit_price - entry) * qty
    ret   = (exit_price - entry) / entry if entry else 0.0
    days  = None
    if isinstance(record["entry_time"], datetime) and isinstance(exit_time, datetime):
        days = (exit_time - record["entry_time"]).total_seconds() / 86_400
    closed = dict(record)
    closed.update({
        "exit_time":    exit_time,
        "exit_price":   exit_price,
        "exit_reason":  exit_reason,
        "pnl":          round(pnl, 4),
        "return_pct":   round(ret, 6),
        "holding_days": round(days, 2) if days is not None else None,
    })
    return closed
