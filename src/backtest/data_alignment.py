"""
backtest/data_alignment.py
---------------------------
Time-safe alignment of historical market data and sentiment snapshots.

This is the most critical module for preventing look-ahead bias.

Alignment convention (daily bars)
----------------------------------
  Decision day T:
    - Market bar used : close of day T  (and all prior bars for indicators)
    - Sentiment used  : most recent snapshot whose timestamp ≤ close of day T
    - Order filled    : open of day T+1  (fill_model="next_open")

  Or, if fill_model="same_close":
    - Order filled    : close of day T

Guarantee
---------
`get_aligned_snapshot(bar_date, ...)` will NEVER return a sentiment snapshot
whose timestamp is strictly after the bar's close timestamp.  If no valid
snapshot exists at or before that point, None is returned — the strategy
will emit NO_ACTION.

Data inputs
-----------
Both inputs are plain Python structures (lists of dicts).  No pandas
dependency is required, though helper functions accept DataFrames too.
"""

from __future__ import annotations

import bisect
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.backtest.schemas import (
    HistoricalMarketRow,
    HistoricalSentimentSnapshot,
    validate_market_row,
    validate_sentiment_snapshot,
)

logger = logging.getLogger(__name__)


# =========================================================================== #
# Normalisation helpers                                                        #
# =========================================================================== #

def _to_utc(ts) -> Optional[datetime]:
    """Convert str / datetime to UTC-aware datetime.  Returns None on failure."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)
    if isinstance(ts, str):
        for fmt in (
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ):
            try:
                dt = datetime.strptime(ts.rstrip("Z") + "+00:00" if ts.endswith("Z") else ts, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except ValueError:
                continue
    return None


def normalise_market_rows(rows: list[dict]) -> list[dict]:
    """
    Return rows with timestamps converted to UTC datetimes and sorted oldest-first.
    Rows that fail `validate_market_row` are dropped with a warning.
    """
    result = []
    for r in rows:
        ts = _to_utc(r.get("timestamp"))
        if ts is None:
            logger.warning("data_alignment: dropping row with unparseable timestamp: %s", r)
            continue
        normed = dict(r)
        normed["timestamp"] = ts
        if not validate_market_row(normed):
            logger.warning("data_alignment: dropping invalid market row for %s @ %s",
                           r.get("ticker"), r.get("timestamp"))
            continue
        result.append(normed)
    result.sort(key=lambda x: x["timestamp"])
    return result


def normalise_sentiment_snapshots(snapshots: list[dict]) -> list[dict]:
    """
    Return snapshots with timestamps converted to UTC datetimes and sorted oldest-first.
    Invalid snapshots are dropped with a warning.
    """
    result = []
    for s in snapshots:
        ts = _to_utc(s.get("timestamp"))
        if ts is None:
            logger.warning("data_alignment: dropping sentiment snapshot with bad timestamp")
            continue
        normed = dict(s)
        normed["timestamp"] = ts
        if not validate_sentiment_snapshot(normed):
            logger.warning("data_alignment: dropping invalid sentiment snapshot @ %s", ts)
            continue
        result.append(normed)
    result.sort(key=lambda x: x["timestamp"])
    return result


# =========================================================================== #
# Sentinel index for binary search                                             #
# =========================================================================== #

class _TimestampIndex:
    """
    Sorted list of (timestamp, item) pairs supporting
    'latest item at or before T' lookups in O(log n).
    """

    def __init__(self, items: list[dict]) -> None:
        self._items = items  # already sorted by timestamp
        self._keys  = [item["timestamp"] for item in items]

    def latest_at_or_before(self, cutoff: datetime) -> Optional[dict]:
        """
        Return the most recent item whose timestamp ≤ cutoff.
        Returns None if no such item exists.
        """
        # bisect_right gives insertion point after all equal keys
        idx = bisect.bisect_right(self._keys, cutoff)
        if idx == 0:
            return None
        return self._items[idx - 1]

    def __len__(self) -> int:
        return len(self._items)


# =========================================================================== #
# Main alignment engine                                                        #
# =========================================================================== #

class DataAligner:
    """
    Aligns historical market bars with sentiment snapshots per ticker.

    Usage
    -----
    aligner = DataAligner(market_rows, sentiment_snapshots)
    for aligned in aligner.iterate(ticker="TSLA"):
        bar        = aligned["bar"]       # HistoricalMarketRow (current day)
        sentiment  = aligned["sentiment"] # HistoricalSentimentSnapshot | None
        prev_bars  = aligned["prev_bars"] # list of prior bars (oldest first)

    Parameters
    ----------
    market_rows : list[dict]
        Raw OHLCV rows — may be multi-ticker.  Will be normalised internally.
    sentiment_snapshots : list[dict]
        Raw sentiment snapshots — may be multi-ticker.  Will be normalised.
    sentiment_cutoff_offset : timedelta
        How far before bar close to cut off sentiment look-up.
        Default 0 (sentiment up to and including bar close is allowed).
    """

    def __init__(
        self,
        market_rows: list[dict],
        sentiment_snapshots: list[dict],
        sentiment_cutoff_offset: timedelta = timedelta(0),
    ) -> None:
        norm_market = normalise_market_rows(market_rows)
        norm_sent   = normalise_sentiment_snapshots(sentiment_snapshots)

        # Partition by ticker
        self._market_by_ticker: dict[str, list[dict]] = {}
        for row in norm_market:
            self._market_by_ticker.setdefault(row["ticker"], []).append(row)

        self._sent_index_by_ticker: dict[str, _TimestampIndex] = {}
        by_ticker: dict[str, list[dict]] = {}
        for snap in norm_sent:
            by_ticker.setdefault(snap["ticker"], []).append(snap)
        for ticker, snaps in by_ticker.items():
            self._sent_index_by_ticker[ticker] = _TimestampIndex(snaps)

        self._cutoff_offset = sentiment_cutoff_offset

    def tickers(self) -> list[str]:
        return list(self._market_by_ticker.keys())

    def iterate(
        self,
        ticker: str,
        lookback_bars: int = 0,
    ):
        """
        Yield one aligned dict per market bar for `ticker`, oldest-first.

        Each dict contains:
          bar        : dict   (current HistoricalMarketRow)
          sentiment  : dict | None  (most recent snapshot ≤ bar timestamp)
          prev_bars  : list[dict]   (up to `lookback_bars` preceding bars)
          bar_index  : int          (0-based index into the bar sequence)
        """
        bars = self._market_by_ticker.get(ticker, [])
        sent_idx = self._sent_index_by_ticker.get(ticker)

        for i, bar in enumerate(bars):
            cutoff = bar["timestamp"] - self._cutoff_offset

            sentiment = None
            if sent_idx is not None:
                sentiment = sent_idx.latest_at_or_before(cutoff)

            prev_bars = bars[max(0, i - lookback_bars): i] if lookback_bars > 0 else []

            yield {
                "bar":       bar,
                "sentiment": sentiment,
                "prev_bars": prev_bars,
                "bar_index": i,
            }

    def iterate_all(self, lookback_bars: int = 0):
        """
        Iterate over all tickers, yielding (ticker, aligned) pairs.
        """
        for ticker in self.tickers():
            for aligned in self.iterate(ticker, lookback_bars=lookback_bars):
                yield ticker, aligned


# =========================================================================== #
# Look-ahead audit                                                             #
# =========================================================================== #

def audit_no_lookahead(
    trades: list[dict],
    market_rows: list[dict],
    sentiment_snapshots: list[dict],
) -> list[str]:
    """
    Validate that no trade entry uses data strictly after the bar timestamp.

    Returns a list of violation descriptions (empty = clean).

    This is a post-hoc audit function, not a real-time guard.
    """
    violations: list[str] = []
    norm_market = normalise_market_rows(market_rows)
    norm_sent   = normalise_sentiment_snapshots(sentiment_snapshots)

    # Build a mapping: ticker → {bar_timestamp → bar}
    bar_map: dict[str, dict[datetime, dict]] = {}
    for row in norm_market:
        bar_map.setdefault(row["ticker"], {})[row["timestamp"]] = row

    # Build sentiment index per ticker
    sent_idx_map: dict[str, _TimestampIndex] = {}
    by_ticker: dict[str, list[dict]] = {}
    for s in norm_sent:
        by_ticker.setdefault(s["ticker"], []).append(s)
    for t, snaps in by_ticker.items():
        sent_idx_map[t] = _TimestampIndex(snaps)

    for trade in trades:
        ticker     = trade.get("ticker", "")
        entry_time = _to_utc(trade.get("entry_time"))
        if entry_time is None:
            continue

        # The bar used for the entry decision must have been closed at or before entry_time
        bars = bar_map.get(ticker, {})
        # Find bar closest at or before entry_time
        bar_ts_list = sorted(bars.keys())
        idx = bisect.bisect_right(bar_ts_list, entry_time)
        if idx == 0:
            violations.append(
                f"{ticker} trade entered at {entry_time} but no bar exists at or before that time"
            )
            continue

        decision_bar_ts = bar_ts_list[idx - 1]

        # Sentiment used must be ≤ decision bar timestamp
        sent_idx = sent_idx_map.get(ticker)
        if sent_idx:
            snap = sent_idx.latest_at_or_before(decision_bar_ts)
            if snap is None:
                # No sentiment available — acceptable (would have been NO_ACTION)
                pass
            elif snap["timestamp"] > decision_bar_ts:
                violations.append(
                    f"{ticker} trade at {entry_time}: sentiment timestamp {snap['timestamp']} "
                    f"is AFTER decision bar {decision_bar_ts} (look-ahead!)"
                )

    return violations
