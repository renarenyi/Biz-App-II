"""
src/backtest/__init__.py
-------------------------
Public API for the Phase 4 backtesting and validation framework.

Quick-start
-----------
    from src.backtest import Backtester, default_config, print_report

    cfg = default_config(
        tickers=["TSLA"],
        start_date="2025-03-17",
        end_date="2026-03-17",
        initial_capital=100_000.0,
    )

    bt = Backtester(cfg)
    result = bt.run(market_rows, sentiment_rows, spy_rows)
    print_report(result)
"""

from src.backtest.backtester import Backtester
from src.backtest.schemas import (
    BacktestConfig,
    BacktestResult,
    BenchmarkResult,
    TradeRecord,
    PortfolioSnapshot,
    HistoricalMarketRow,
    HistoricalSentimentSnapshot,
    default_config,
    make_trade_record,
    close_trade_record,
    validate_market_row,
    validate_sentiment_snapshot,
    validate_backtest_config,
)
from src.backtest.data_alignment import DataAligner, audit_no_lookahead
from src.backtest.strategy_adapter import StrategyAdapter
from src.backtest.execution_simulator import ExecutionSimulator
from src.backtest.portfolio_tracker import PortfolioTracker
from src.backtest.benchmark import BenchmarkBuilder
from src.backtest.metrics import (
    compute_metrics,
    max_drawdown,
    sharpe_ratio,
    downside_deviation,
    profit_factor,
    win_rate,
    trade_summary,
)
from src.backtest.report_generator import (
    summary_table,
    trade_log_table,
    equity_curve_table,
    comparison_table,
    written_interpretation,
    print_report,
    to_csv_string,
)

__all__ = [
    "Backtester",
    "BacktestConfig",
    "BacktestResult",
    "BenchmarkResult",
    "TradeRecord",
    "PortfolioSnapshot",
    "HistoricalMarketRow",
    "HistoricalSentimentSnapshot",
    "default_config",
    "make_trade_record",
    "close_trade_record",
    "validate_market_row",
    "validate_sentiment_snapshot",
    "validate_backtest_config",
    "DataAligner",
    "audit_no_lookahead",
    "StrategyAdapter",
    "ExecutionSimulator",
    "PortfolioTracker",
    "BenchmarkBuilder",
    "compute_metrics",
    "max_drawdown",
    "sharpe_ratio",
    "downside_deviation",
    "profit_factor",
    "win_rate",
    "trade_summary",
    "summary_table",
    "trade_log_table",
    "equity_curve_table",
    "comparison_table",
    "written_interpretation",
    "print_report",
    "to_csv_string",
]
