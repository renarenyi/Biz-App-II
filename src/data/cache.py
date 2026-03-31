"""
cache.py
--------
Two-tier caching for the Trading Agent data layer.

Tier 1: TTLCache  — fast in-memory dict with per-entry TTL
Tier 2: DiskCache — optional file-based persistence using Parquet (OHLCV)
                    and JSON-lines (news)

DataCache combines both tiers.

Design goals:
  - Reduce unnecessary API calls during development and backtesting
  - Support frozen historical snapshots for reproducible backtests
  - Never silently return stale data — always log cache hits / misses
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# =========================================================================== #
# Tier 1 — In-memory TTL cache                                                #
# =========================================================================== #

class TTLCache:
    """
    Thread-unsafe, in-process, TTL-based key/value cache.

    Parameters
    ----------
    default_ttl : float
        Default time-to-live in seconds.
    """

    def __init__(self, default_ttl: float = 60.0) -> None:
        self._store: dict[str, tuple[Any, float]] = {}  # key -> (value, expiry_time)
        self.default_ttl = default_ttl

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def get(self, key: str) -> Optional[Any]:
        """Return cached value or None if missing / expired."""
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expiry = entry
        if time.monotonic() > expiry:
            logger.debug("Cache EXPIRED: %s", key)
            del self._store[key]
            return None
        logger.debug("Cache HIT: %s", key)
        return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Store a value with a TTL (falls back to default_ttl)."""
        expiry = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
        self._store[key] = (value, expiry)
        logger.debug("Cache SET: %s (TTL=%.0fs)", key, ttl or self.default_ttl)

    def invalidate(self, key: str) -> None:
        """Remove a single entry."""
        self._store.pop(key, None)

    def clear(self) -> None:
        """Wipe the entire in-memory cache."""
        self._store.clear()
        logger.debug("In-memory cache cleared.")

    def size(self) -> int:
        return len(self._store)

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _evict_expired(self) -> None:
        """Purge all expired entries. Call periodically if memory is a concern."""
        now = time.monotonic()
        expired = [k for k, (_, expiry) in self._store.items() if now > expiry]
        for k in expired:
            del self._store[k]
        if expired:
            logger.debug("Evicted %d expired cache entries.", len(expired))


# =========================================================================== #
# Tier 2 — File-based disk cache                                              #
# =========================================================================== #

class DiskCache:
    """
    Persistent file cache using:
    - Parquet files for OHLCV DataFrames (efficient columnar storage)
    - JSON-lines files for news article lists

    Each dataset is stored as `{cache_dir}/{key}.parquet` or `{key}.jsonl`.

    Parameters
    ----------
    cache_dir : str | Path
        Directory to store cached files.
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # OHLCV (Parquet)                                                      #
    # ------------------------------------------------------------------ #

    def save_bars(self, key: str, df: pd.DataFrame) -> None:
        """Save a bars DataFrame to parquet."""
        path = self._parquet_path(key)
        df.to_parquet(path, index=False)
        logger.debug("DiskCache WRITE bars: %s → %s", key, path)

    def load_bars(self, key: str) -> Optional[pd.DataFrame]:
        """Load a bars DataFrame from parquet, or return None if not found."""
        path = self._parquet_path(key)
        if not path.exists():
            logger.debug("DiskCache MISS bars: %s", key)
            return None
        try:
            df = pd.read_parquet(path)
            logger.debug("DiskCache HIT bars: %s (%d rows)", key, len(df))
            return df
        except Exception as exc:
            logger.warning("DiskCache READ ERROR bars %s: %s", key, exc)
            return None

    # ------------------------------------------------------------------ #
    # News articles (JSON-lines)                                           #
    # ------------------------------------------------------------------ #

    def save_articles(self, key: str, articles: list[dict]) -> None:
        """Persist articles to a JSON-lines file."""
        path = self._jsonl_path(key)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                for article in articles:
                    # Convert datetime objects to ISO strings for JSON serialisation
                    serialisable = _make_json_serialisable(article)
                    fh.write(json.dumps(serialisable) + "\n")
            logger.debug("DiskCache WRITE news: %s → %s", key, path)
        except Exception as exc:
            logger.warning("DiskCache WRITE ERROR news %s: %s", key, exc)

    def load_articles(self, key: str) -> Optional[list[dict]]:
        """Load articles from a JSON-lines file, or return None if not found."""
        path = self._jsonl_path(key)
        if not path.exists():
            logger.debug("DiskCache MISS news: %s", key)
            return None
        try:
            articles = []
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        articles.append(json.loads(line))
            logger.debug("DiskCache HIT news: %s (%d articles)", key, len(articles))
            return articles
        except Exception as exc:
            logger.warning("DiskCache READ ERROR news %s: %s", key, exc)
            return None

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _parquet_path(self, key: str) -> Path:
        safe_key = _sanitise_key(key)
        return self.cache_dir / f"{safe_key}.parquet"

    def _jsonl_path(self, key: str) -> Path:
        safe_key = _sanitise_key(key)
        return self.cache_dir / f"{safe_key}.jsonl"

    def list_keys(self) -> list[str]:
        """List all cached keys (both parquet and jsonl)."""
        return [p.stem for p in self.cache_dir.iterdir() if p.suffix in (".parquet", ".jsonl")]

    def delete(self, key: str) -> None:
        """Delete all cached files for a key."""
        for path in [self._parquet_path(key), self._jsonl_path(key)]:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass


# =========================================================================== #
# Combined DataCache                                                           #
# =========================================================================== #

class DataCache:
    """
    Three-tier cache combining in-memory TTL and optional disk persistence.

    Usage
    -----
    cache = DataCache(cache_dir="data/cache")
    df = cache.get_bars("AAPL_1Day_2024")
    if df is None:
        df = provider.fetch(...)
        cache.set_bars("AAPL_1Day_2024", df, ttl=3600)

    Parameters
    ----------
    cache_dir : str | Path
        Where disk cache files are stored.
    memory_ttl_bars : float
        In-memory TTL for OHLCV data (seconds).
    memory_ttl_news : float
        In-memory TTL for news data (seconds).
    memory_ttl_quotes : float
        In-memory TTL for live quotes (seconds).
    use_disk : bool
        Whether to persist data to disk. Set to False in pure live mode.
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/cache",
        memory_ttl_bars: float = 3600,
        memory_ttl_news: float = 600,
        memory_ttl_quotes: float = 30,
        use_disk: bool = True,
    ) -> None:
        self._mem = TTLCache(default_ttl=memory_ttl_bars)
        self._disk = DiskCache(cache_dir) if use_disk else None

        self._ttl_bars = memory_ttl_bars
        self._ttl_news = memory_ttl_news
        self._ttl_quotes = memory_ttl_quotes

    # ------------------------------------------------------------------ #
    # OHLCV bars                                                           #
    # ------------------------------------------------------------------ #

    def get_bars(self, key: str) -> Optional[pd.DataFrame]:
        """
        Try in-memory first, then disk.
        Disk hits are promoted back into memory.
        """
        df = self._mem.get(key)
        if df is not None:
            return df

        if self._disk:
            df = self._disk.load_bars(key)
            if df is not None:
                # Promote to memory (use remaining TTL heuristic: half of normal TTL)
                self._mem.set(key, df, ttl=self._ttl_bars / 2)
                return df

        logger.debug("Full cache MISS bars: %s", key)
        return None

    def set_bars(self, key: str, df: pd.DataFrame, ttl: Optional[float] = None) -> None:
        self._mem.set(key, df, ttl=ttl or self._ttl_bars)
        if self._disk:
            self._disk.save_bars(key, df)

    # ------------------------------------------------------------------ #
    # News articles                                                        #
    # ------------------------------------------------------------------ #

    def get_articles(self, key: str) -> Optional[list[dict]]:
        articles = self._mem.get(key)
        if articles is not None:
            return articles

        if self._disk:
            articles = self._disk.load_articles(key)
            if articles is not None:
                self._mem.set(key, articles, ttl=self._ttl_news / 2)
                return articles

        logger.debug("Full cache MISS news: %s", key)
        return None

    def set_articles(self, key: str, articles: list[dict], ttl: Optional[float] = None) -> None:
        self._mem.set(key, articles, ttl=ttl or self._ttl_news)
        if self._disk:
            self._disk.save_articles(key, articles)

    # ------------------------------------------------------------------ #
    # Live quotes                                                          #
    # ------------------------------------------------------------------ #

    def get_quote(self, ticker: str) -> Optional[dict]:
        return self._mem.get(f"quote:{ticker}")

    def set_quote(self, ticker: str, quote: dict, ttl: Optional[float] = None) -> None:
        self._mem.set(f"quote:{ticker}", quote, ttl=ttl or self._ttl_quotes)

    # ------------------------------------------------------------------ #
    # Cache management                                                     #
    # ------------------------------------------------------------------ #

    def clear_memory(self) -> None:
        self._mem.clear()

    def memory_size(self) -> int:
        return self._mem.size()


# =========================================================================== #
# Private helpers                                                              #
# =========================================================================== #

def _sanitise_key(key: str) -> str:
    """Replace characters unsafe for filenames."""
    return key.replace("/", "_").replace(":", "_").replace(" ", "_")


def _make_json_serialisable(obj: Any) -> Any:
    """Recursively convert datetime objects inside dicts/lists to ISO strings."""
    from datetime import datetime
    if isinstance(obj, dict):
        return {k: _make_json_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_serialisable(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj
