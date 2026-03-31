"""
backtest/backtester.py
------------------------
Top-level orchestration layer for Phase 4 backtesting.

Execution flow
--------------
For each ticker in the config:

  1. Align market bars and sentiment snapshots (DataAligner).
  2. For each bar T (oldest → newest):
       a. If there is an open position, check stop-loss / TP first.
          If triggered → close at current bar (intra-bar stop hit).
       b. If there is still an open position, run exit signal rule.
          If EXIT → simulate sell fill at next bar open.
       c. If no open position, run entry signal rule.
          If BUY and eligibility passes → simulate buy fill at next bar open.
       d. Record portfolio snapshot (mark-to-market at bar close).
  3. At the end of the window, force-close any remaining open positions
     at the last bar's close price.
  4. Compute all metrics.
  5. Build benchmark (SPY buy-and-hold).
  6. Return BacktestResult.

Time alignment guarantee
------------------------
Signal for bar T is generated using bar T's data.
Fill for any signal generated at bar T occurs at bar T+1's open.
Sentiment used must have timestamp ≤ bar T's close timestamp.

These guarantees are enforced by DataAligner and not re-checked here;
the audit_no_lookahead() function in data_alignment.py can be called
post-hoc.

Usage
-----
    from src.backtest.backtester import Backtester
    from src.backtest.schemas import default_config

    cfg = default_config(
        tickers=["TSLA"],
        start_date="2025-03-17",
        end_date="2026-03-17",
        initial_capital=100_000.0,
    )
    bt = Backtester(cfg)
    result = bt.run(market_rows, sentiment_rows, spy_rows)
    from src.backtest.report_generator import print_report
    print_report(result)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from src.backtest.schemas import (
    BacktestConfig,
    BacktestResult,
    TradeRecord,
    default_config,
    validate_backtest_config,
)
from src.backtest.data_alignment import DataAligner
from src.backtest.strategy_adapter import StrategyAdapter
from src.backtest.execution_simulator import ExecutionSimulator
from src.backtest.portfolio_tracker import PortfolioTracker
from src.backtest.benchmark import BenchmarkBuilder
from src.backtest.metrics import compute_metrics
from src.strategy.risk_manager import trailing_stop_update

logger = logging.getLogger(__name__)


class Backtester:
    """
    Orchestrates a single-pass backtest over historical data.

    Parameters
    ----------
    config : BacktestConfig
        All strategy and execution parameters.  Use `default_config()`
        to create one with sensible defaults.
    """

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        cfg = config or default_config()
        ok, reason = validate_backtest_config(cfg)
        if not ok:
            raise ValueError(f"Invalid BacktestConfig: {reason}")
        self._config = cfg

    # ------------------------------------------------------------------ #
    # Public run method                                                   #
    # ------------------------------------------------------------------ #

    def run(
        self,
        market_rows:        list[dict],
        sentiment_rows:     list[dict],
        benchmark_rows:     Optional[list[dict]] = None,
        reference_time:     Optional[datetime]   = None,
    ) -> BacktestResult:
        """
        Run the full backtest and return a BacktestResult.

        Parameters
        ----------
        market_rows     : list[HistoricalMarketRow]  — all tickers combined
        sentiment_rows  : list[HistoricalSentimentSnapshot] — all tickers combined
        benchmark_rows  : list[HistoricalMarketRow] for SPY (optional)
        reference_time  : unused in daily mode; reserved for intraday extension

        Returns
        -------
        BacktestResult dict

        Never raises — returns a partial result on unexpected error.
        """
        logger.info(
            "Backtester.run: %d market rows, %d sentiment rows, tickers=%s",
            len(market_rows), len(sentiment_rows), self._config.get("tickers"),
        )

        # ── Modules ──────────────────────────────────────────────────── #
        aligner   = DataAligner(market_rows, sentiment_rows)
        adapter   = StrategyAdapter(self._config)
        simulator = ExecutionSimulator(
            fill_model   = self._config.get("fill_model", "next_open"),
            slippage_pct = self._config.get("slippage_pct", 0.0),
            commission   = self._config.get("commission_per_trade", 0.0),
        )
        tracker = PortfolioTracker(
            initial_capital          = self._config.get("initial_capital", 100_000.0),
            max_concurrent_positions = self._config.get("max_concurrent_positions", 5),
        )

        tickers    = self._config.get("tickers") or aligner.tickers()
        all_trades: list[TradeRecord] = []

        # ── Per-ticker loop ───────────────────────────────────────────── #
        for ticker in tickers:
            bars = list(aligner.iterate(ticker, lookback_bars=1))
            n    = len(bars)

            for step_idx, aligned in enumerate(bars):
                bar       = aligned["bar"]
                sentiment = aligned["sentiment"]
                bar_ts    = bar["timestamp"]

                # The "next bar" for fill purposes (T+1)
                next_bar: Optional[dict] = None
                if step_idx + 1 < n:
                    next_bar = bars[step_idx + 1]["bar"]

                prices_now = {ticker: float(bar.get("close") or 0)}
                trailing_pct = self._config.get("stop_loss_pct", 0.02)

                # ── (a-pre) Trailing stop update ──────────────────── #
                open_trade = tracker.get_position(ticker)
                if open_trade is not None:
                    bar_high = float(bar.get("high") or 0)
                    new_stop = trailing_stop_update(open_trade, bar_high, trailing_pct)
                    if new_stop is not None:
                        open_trade["stop_loss_price"] = new_stop

                # ── (a) Check stops intra-bar ─────────────────────── #
                if open_trade is not None:
                    closed = simulator.check_stops(open_trade, bar)
                    if closed is not None:
                        tracker.exit_position(ticker, closed)
                        all_trades.append(closed)
                        open_trade = None   # position cleared

                # ── (b) Signal-driven exit ────────────────────────── #
                open_trade = tracker.get_position(ticker)
                if open_trade is not None:
                    exit_dec = adapter.exit_signal(bar, sentiment, open_trade, bar_ts)
                    if exit_dec.get("signal") == "EXIT":
                        closed = simulator.simulate_exit(
                            open_trade, bar, next_bar,
                            exit_reason=_exit_reason(exit_dec),
                            reference_time=_fill_time(next_bar, bar),
                        )
                        tracker.exit_position(ticker, closed)
                        all_trades.append(closed)

                # ── (c) Entry ─────────────────────────────────────── #
                open_trade = tracker.get_position(ticker)
                if open_trade is None:
                    entry_dec = adapter.entry_signal(bar, sentiment, bar_ts)
                    if entry_dec.get("signal") == "BUY":
                        price    = float(bar.get("close") or 0)
                        equity   = tracker.current_equity(prices_now)
                        qty      = adapter.compute_qty(price, equity)

                        if qty > 0:
                            stop_price = adapter.compute_stop(price)
                            tp_price   = adapter.compute_take_profit(price)

                            new_trade = simulator.simulate_entry(
                                ticker          = ticker,
                                signal_bar      = bar,
                                next_bar        = next_bar,
                                qty             = qty,
                                stop_loss_price = stop_price,
                                take_profit_price = tp_price,
                                reference_time  = _fill_time(next_bar, bar),
                            )
                            if new_trade is not None:
                                opened = tracker.enter_position(ticker, new_trade)
                                if not opened:
                                    logger.debug(
                                        "Backtester: entry for %s blocked by portfolio limits", ticker
                                    )

                # ── (d) Snapshot ──────────────────────────────────── #
                tracker.record_snapshot(bar_ts, prices_now)

        # ── End-of-period: force-close remaining positions ─────────── #
        last_prices: dict[str, float] = {}
        for ticker in tickers:
            ticker_bars = list(aligner.iterate(ticker))
            if ticker_bars:
                last_bar = ticker_bars[-1]["bar"]
                last_prices[ticker] = float(last_bar.get("close") or 0)

        last_ts = datetime.now(tz=timezone.utc)
        eop_closed = tracker.force_close_all(last_prices, last_ts)
        all_trades.extend(eop_closed)

        # ── Benchmark ──────────────────────────────────────────────── #
        benchmark_result = None
        if benchmark_rows:
            bench_ticker = self._config.get("benchmark_ticker", "SPY")
            benchmark_result = BenchmarkBuilder.build(
                benchmark_rows,
                initial_capital = self._config.get("initial_capital", 100_000.0),
                ticker          = bench_ticker,
            )

        # ── Metrics ────────────────────────────────────────────────── #
        perf = compute_metrics(
            trades          = all_trades,
            equity_curve    = tracker.equity_curve(),
            initial_capital = self._config.get("initial_capital", 100_000.0),
            benchmark       = benchmark_result,
        )

        # ── Build result ───────────────────────────────────────────── #
        curve = tracker.equity_curve()
        start_date = _fmt_date(curve[0]["timestamp"]) if curve else str(self._config.get("start_date"))
        end_date   = _fmt_date(curve[-1]["timestamp"]) if curve else str(self._config.get("end_date"))

        result: BacktestResult = {
            "strategy_name":   self._config.get("strategy_name", "sentiment_ma50_long_only"),
            "start_date":      start_date,
            "end_date":        end_date,
            "initial_capital": self._config.get("initial_capital", 100_000.0),
            "config":          dict(self._config),
            "trades":          all_trades,
            "equity_curve":    curve,
            "benchmark":       benchmark_result,
            **perf,
        }

        logger.info(
            "Backtester.run complete: trades=%d total_return=%.2f%% sharpe=%s max_dd=%.2f%%",
            len(all_trades),
            (result.get("total_return") or 0) * 100,
            result.get("sharpe_ratio"),
            (result.get("max_drawdown") or 0) * 100,
        )
        return result

    # ------------------------------------------------------------------ #
    # Parameter sweep helper                                              #
    # ------------------------------------------------------------------ #

    def sweep(
        self,
        market_rows:    list[dict],
        sentiment_rows: list[dict],
        param_grid:     list[dict],
        benchmark_rows: Optional[list[dict]] = None,
    ) -> list[BacktestResult]:
        """
        Run multiple backtest configurations from a parameter grid.

        Parameters
        ----------
        param_grid : list of dicts, each dict is a set of config overrides.
            Example: [{"stop_loss_pct": 0.01}, {"stop_loss_pct": 0.03}]

        Returns
        -------
        list of BacktestResult, one per config override
        """
        results = []
        base_config = dict(self._config)
        for overrides in param_grid:
            cfg = {**base_config, **overrides}
            cfg["strategy_name"] = base_config.get("strategy_name", "sweep") + \
                "_" + "_".join(f"{k}={v}" for k, v in overrides.items())
            bt = Backtester(cfg)
            results.append(bt.run(market_rows, sentiment_rows, benchmark_rows))
        return results


# =========================================================================== #
# Helpers                                                                      #
# =========================================================================== #

def _exit_reason(decision: dict) -> str:
    reason = (decision.get("reason") or "").lower()
    if "stop" in reason:
        return "stop_loss"
    if "sentiment" in reason:
        return "sentiment_reversal"
    if "trend" in reason:
        return "trend_exit"
    return "signal_exit"


def _fill_time(next_bar: Optional[dict], signal_bar: dict) -> datetime:
    """Return fill timestamp: next bar open time, or signal bar time as fallback."""
    if next_bar is not None:
        ts = next_bar.get("timestamp")
        if ts is not None:
            return ts
    ts = signal_bar.get("timestamp")
    return ts if ts is not None else datetime.now(tz=timezone.utc)


def _fmt_date(dt) -> str:
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%d")
    if isinstance(dt, str):
        return dt[:10]
    return str(dt) if dt else ""
