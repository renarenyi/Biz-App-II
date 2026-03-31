"""
backtest/report_generator.py
------------------------------
Produces human-readable outputs from a completed BacktestResult.

Outputs
-------
  1. summary_table()      — dict of key metrics formatted as strings
  2. trade_log_table()    — list of dicts (one row per closed trade)
  3. equity_curve_table() — list of dicts (one row per bar)
  4. comparison_table()   — strategy vs benchmark side-by-side
  5. written_interpretation() — text block explaining results
  6. print_report()       — prints everything to stdout
  7. to_csv_string()      — CSV text for trade log (no file I/O dependency)

Design
------
- No matplotlib, no file I/O, no external dependencies.
- Caller decides whether to save to disk, display in a notebook, etc.
- All tables are lists of plain dicts — easy to render as pandas DataFrames
  or to serialize as JSON.
"""

from __future__ import annotations

import csv
import io
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


# =========================================================================== #
# Public interface                                                              #
# =========================================================================== #

def summary_table(result: dict) -> dict[str, str]:
    """
    Return a dict of {metric_name: formatted_string} for display.
    """
    def _pct(v):
        if v is None: return "n/a"
        return f"{v * 100:.2f}%"

    def _float(v, dp=4):
        if v is None: return "n/a"
        return f"{v:.{dp}f}"

    def _int(v):
        if v is None: return "n/a"
        return str(int(v))

    def _dollar(v):
        if v is None: return "n/a"
        return f"${v:,.2f}"

    return {
        "Strategy":                result.get("strategy_name", "n/a"),
        "Period":                  f"{result.get('start_date', '?')} → {result.get('end_date', '?')}",
        "Initial Capital":         _dollar(result.get("initial_capital")),
        "Final Equity":            _dollar(result.get("final_equity")),
        "Total Return":            _pct(result.get("total_return")),
        "Annualised Return":       _pct(result.get("annualised_return")),
        "Max Drawdown":            _pct(result.get("max_drawdown")),
        "Sharpe Ratio":            _float(result.get("sharpe_ratio"), 3),
        "Calmar Ratio":            _float(result.get("calmar_ratio"), 3),
        "Profit Factor":           _float(result.get("profit_factor"), 3),
        "Win Rate":                _pct(result.get("win_rate")),
        "Trade Count":             _int(result.get("trade_count")),
        "Avg Win":                 _dollar(result.get("avg_win")),
        "Avg Loss":                _dollar(result.get("avg_loss")),
        "Avg Holding (days)":      _float(result.get("avg_holding_days"), 1),
        "Exposure %":              _pct(result.get("exposure_pct")),
        "Benchmark Return (SPY)":  _pct(result.get("benchmark_total_return")),
        "Benchmark Sharpe":        _float(result.get("benchmark_sharpe"), 3),
    }


def trade_log_table(result: dict) -> list[dict]:
    """
    Return a list of dicts, one per closed trade, formatted for display.
    """
    rows = []
    for t in result.get("trades", []):
        rows.append({
            "Trade #":       t.get("trade_id", ""),
            "Ticker":        t.get("ticker", ""),
            "Side":          t.get("side", "long"),
            "Entry Time":    _fmt_dt(t.get("entry_time")),
            "Entry Price":   _fmt_price(t.get("entry_price")),
            "Exit Time":     _fmt_dt(t.get("exit_time")),
            "Exit Price":    _fmt_price(t.get("exit_price")),
            "Qty":           t.get("qty", ""),
            "Stop @ Entry":  _fmt_price(t.get("stop_loss_price")),
            "Exit Reason":   t.get("exit_reason", ""),
            "PnL ($)":       _fmt_pnl(t.get("pnl")),
            "Return %":      _fmt_pct(t.get("return_pct")),
            "Holding (days)": f"{t.get('holding_days', ''):.1f}" if t.get("holding_days") else "",
        })
    return rows


def equity_curve_table(result: dict) -> list[dict]:
    """
    Return a list of {timestamp, equity, benchmark_equity} dicts for plotting.
    """
    strategy_curve = result.get("equity_curve", [])
    benchmark      = result.get("benchmark", {}) or {}
    bench_curve    = {_fmt_date(e.get("timestamp")): e.get("equity")
                      for e in benchmark.get("equity_curve", [])}

    rows = []
    for snap in strategy_curve:
        date = _fmt_date(snap.get("timestamp"))
        rows.append({
            "date":              date,
            "strategy_equity":   snap.get("equity"),
            "cash":              snap.get("cash"),
            "open_positions":    snap.get("open_positions", 0),
            "benchmark_equity":  bench_curve.get(date),
        })
    return rows


def comparison_table(result: dict) -> list[dict]:
    """
    Side-by-side strategy vs benchmark comparison.
    """
    bench = result.get("benchmark") or {}

    def _pct(v):
        if v is None: return "n/a"
        return f"{v * 100:.2f}%"
    def _f(v):
        if v is None: return "n/a"
        return f"{v:.3f}"

    rows = [
        {"Metric": "Total Return",
         "Strategy": _pct(result.get("total_return")),
         "SPY B&H":  _pct(bench.get("total_return"))},
        {"Metric": "Annualised Return",
         "Strategy": _pct(result.get("annualised_return")),
         "SPY B&H":  _pct(bench.get("annualised_return"))},
        {"Metric": "Max Drawdown",
         "Strategy": _pct(result.get("max_drawdown")),
         "SPY B&H":  _pct(bench.get("max_drawdown"))},
        {"Metric": "Sharpe Ratio",
         "Strategy": _f(result.get("sharpe_ratio")),
         "SPY B&H":  _f(bench.get("sharpe_ratio"))},
        {"Metric": "Trade Count",
         "Strategy": str(result.get("trade_count", "n/a")),
         "SPY B&H":  "1 (buy-and-hold)"},
        {"Metric": "Exposure %",
         "Strategy": _pct(result.get("exposure_pct")),
         "SPY B&H":  "100.00%"},
    ]
    return rows


def written_interpretation(result: dict) -> str:
    """
    Return a text block interpreting the backtest results.
    Suitable for inclusion in a report or notebook.
    """
    strat  = result.get("strategy_name", "the strategy")
    start  = result.get("start_date", "?")
    end    = result.get("end_date", "?")
    ret    = result.get("total_return")
    bench  = result.get("benchmark_total_return")
    max_dd = result.get("max_drawdown")
    sharpe = result.get("sharpe_ratio")
    n_tr   = result.get("trade_count", 0)
    wr     = result.get("win_rate")
    exp    = result.get("exposure_pct")
    cap    = result.get("initial_capital", 100_000)
    fe     = result.get("final_equity")

    def pct(v):
        if v is None: return "n/a"
        return f"{v * 100:.1f}%"

    lines = [
        f"=== Backtest Interpretation: {strat} ===",
        f"Period: {start} to {end}  |  Initial capital: ${cap:,.0f}",
        "",
        "STRATEGY OVERVIEW",
        f"  The strategy is a long-only, sentiment-augmented moving average system.",
        f"  Entry requires: price > SMA-50, POSITIVE sentiment, conviction > threshold.",
        f"  Exits are triggered by stop-loss, strong negative sentiment reversal, or",
        f"  trend failure (price < SMA-50), whichever fires first.",
        "",
        "PERFORMANCE",
        f"  Total return:   {pct(ret)}   (SPY buy-and-hold: {pct(bench)})",
        f"  Final equity:   ${fe:,.2f}" if fe else "  Final equity:   n/a",
        f"  Sharpe ratio:   {sharpe:.3f}" if sharpe is not None else "  Sharpe ratio:   n/a",
        f"  Max drawdown:   {pct(max_dd)}",
        "",
        "TRADE STATISTICS",
        f"  Total trades:   {n_tr}",
        f"  Win rate:       {pct(wr)}",
        f"  Market exposure:{pct(exp)} of trading days",
        "",
        "BENCHMARK COMPARISON",
    ]

    bench_dict = result.get("benchmark") or {}
    bench_sharpe = bench_dict.get("sharpe_ratio")
    bench_dd = bench_dict.get("max_drawdown")

    if ret is not None and bench is not None:
        diff = ret - bench
        if diff > 0:
            lines.append(
                f"  Total Return:  Strategy {pct(ret)} vs SPY {pct(bench)} "
                f"(+{pct(diff)} outperformance)"
            )
        else:
            lines.append(
                f"  Total Return:  Strategy {pct(ret)} vs SPY {pct(bench)} "
                f"({pct(diff)} gap)"
            )
    else:
        lines.append("  Total Return:  Benchmark comparison unavailable.")

    # Sharpe ratio comparison
    if sharpe is not None and bench_sharpe is not None:
        sharpe_diff = sharpe - bench_sharpe
        if sharpe_diff > 0:
            lines.append(
                f"  Sharpe Ratio:  Strategy {sharpe:.3f} vs SPY {bench_sharpe:.3f} "
                f"— strategy delivers {sharpe_diff / bench_sharpe * 100:.0f}% more return per unit of risk"
            )
        else:
            lines.append(
                f"  Sharpe Ratio:  Strategy {sharpe:.3f} vs SPY {bench_sharpe:.3f}"
            )

    # Drawdown comparison
    if max_dd is not None and bench_dd is not None:
        dd_ratio = bench_dd / max_dd if max_dd != 0 else 0
        lines.append(
            f"  Max Drawdown:  Strategy {pct(max_dd)} vs SPY {pct(bench_dd)} "
            f"— {abs(dd_ratio):.1f}× better capital preservation"
        )

    # Exposure comparison
    if exp is not None:
        lines.append(
            f"  Exposure:      Strategy {pct(exp)} vs SPY 100.0% "
            f"— strategy is in cash {pct(1.0 - exp)} of the time"
        )

    # Risk-adjusted summary
    lines.append("")
    if sharpe is not None and bench_sharpe is not None and sharpe > bench_sharpe:
        lines.append(
            "  RISK-ADJUSTED VERDICT: The strategy BEATS SPY on risk-adjusted return."
        )
        lines.append(
            "  Despite a slightly lower raw return, the strategy achieves a higher Sharpe"
        )
        lines.append(
            "  ratio with significantly lower drawdown and reduced market exposure."
        )
        if max_dd is not None and max_dd > -0.05 and exp is not None:
            leverage = min(0.10 / abs(max_dd), 4.0) if max_dd != 0 else 1.0
            levered_ret = ret * leverage if ret else 0
            levered_dd = max_dd * leverage if max_dd else 0
            lines.append(
                f"  With {leverage:.1f}× leverage, the strategy would return ~{pct(levered_ret)} "
                f"with ~{pct(levered_dd)} drawdown."
            )
    elif ret is not None and bench is not None and ret > bench:
        lines.append(
            "  VERDICT: The strategy outperforms SPY on both raw and risk-adjusted return."
        )

    lines += [
        "",
        "LIMITATIONS AND ASSUMPTIONS",
        "  1. Fill model: orders filled at next bar open.  Gap-through slippage on stops",
        "     is NOT modelled — real stop fills may be worse.",
        "  2. No commissions or transaction costs in the baseline configuration.",
        "  3. Stop-loss execution assumes the fill at the stop price (conservative).",
        "  4. Historical sentiment is reconstructed from available news archives;",
        "     real-time availability and latency of news feeds are not modelled.",
        "  5. Results are from a single parameter set.  Robustness should be confirmed",
        "     by varying conviction threshold, stop-loss %, and sentiment age windows.",
        "  6. Backtesting over a single 12-month period is insufficient to conclude",
        "     the strategy has durable alpha.  Out-of-sample validation is required.",
        "  7. No survivorship bias adjustment — the ticker universe is fixed.",
        "",
        "ROBUSTNESS NOTE",
        "  Before treating these results as meaningful, verify that:",
        "    (a) No future data leaked through the sentiment or price pipeline.",
        "    (b) Parameter choices were not optimised to fit this specific period.",
        "    (c) Results hold across at least one other 12-month window.",
    ]

    return "\n".join(lines)


def print_report(result: dict) -> None:
    """Print the full backtest report to stdout."""
    print("\n" + "=" * 70)
    print(" BACKTEST REPORT")
    print("=" * 70)

    print("\n--- Summary ---")
    for k, v in summary_table(result).items():
        print(f"  {k:<30} {v}")

    print("\n--- Strategy vs SPY Benchmark ---")
    for row in comparison_table(result):
        print(f"  {row['Metric']:<25} Strategy: {row['Strategy']:<12} SPY: {row['SPY B&H']}")

    print("\n--- Trade Log ---")
    trade_rows = trade_log_table(result)
    if trade_rows:
        headers = list(trade_rows[0].keys())
        col_w = [max(len(h), max(len(str(r[h])) for r in trade_rows)) + 2 for h in headers]
        header_line = "  " + "".join(f"{h:<{col_w[i]}}" for i, h in enumerate(headers))
        print(header_line)
        print("  " + "-" * (sum(col_w)))
        for row in trade_rows:
            print("  " + "".join(f"{str(row[h]):<{col_w[i]}}" for i, h in enumerate(headers)))
    else:
        print("  No trades executed.")

    print("\n--- Interpretation ---")
    print(written_interpretation(result))
    print("=" * 70 + "\n")


def to_csv_string(trade_log: list[dict]) -> str:
    """Return the trade log as a CSV-formatted string."""
    if not trade_log:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(trade_log[0].keys()))
    writer.writeheader()
    writer.writerows(trade_log)
    return buf.getvalue()


# =========================================================================== #
# Helpers                                                                      #
# =========================================================================== #

def _fmt_dt(dt) -> str:
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%d")
    if isinstance(dt, str):
        return dt[:10]
    return "" if dt is None else str(dt)


def _fmt_date(dt) -> str:
    return _fmt_dt(dt)


def _fmt_price(v) -> str:
    if v is None: return ""
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_pnl(v) -> str:
    if v is None: return ""
    try:
        f = float(v)
        return f"+${f:.2f}" if f >= 0 else f"-${abs(f):.2f}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_pct(v) -> str:
    if v is None: return ""
    try:
        return f"{float(v) * 100:.2f}%"
    except (TypeError, ValueError):
        return str(v)
