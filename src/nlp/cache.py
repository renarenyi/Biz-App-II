"""
nlp/cache.py
------------
Inference result cache for the Phase 2 sentiment layer.

Separate from the Phase 1 DataCache (which caches raw news and OHLCV).
This cache stores *model inference outputs* so articles are never
classified twice across sessions.

Cache key
---------
content_hash from schemas.compute_content_hash() — a 16-char SHA-256
prefix over (ticker | headline | snippet | published_at).

Storage tiers
-------------
Tier 1 — In-memory TTLCache (fast, process-scoped)
Tier 2 — Disk JSON-lines file (persistent, survives restarts)

The inference cache is write-once / read-many:
- Once a result is stored, it is never automatically invalidated.
- If preprocessing rules change materially (e.g., MAX_ANALYSIS_CHARS),
  the cache dir can be wiped by the user and will be rebuilt on next run.

Design rationale
----------------
Financial news feeds republish identical headlines many times.
Without this cache, running the agent over a 24-hour window that
includes 5 Reuters/AP syndication copies of the same event would
trigger 5 model calls.  With the cache, only the first call happens.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default TTL for in-memory entries: 6 hours
# (long enough for a full daily trading session; short enough to refresh
# stale results if the agent runs multiple days)
DEFAULT_MEM_TTL = 6 * 3600


# =========================================================================== #
# In-memory TTL store (identical pattern to Phase 1 TTLCache)                 #
# =========================================================================== #

class _TTLCache:
    """Minimal in-process key/value store with per-entry TTL."""

    def __init__(self, default_ttl: float = DEFAULT_MEM_TTL) -> None:
        self._store: dict[str, tuple[Any, float]] = {}
        self.default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expiry = entry
        if time.monotonic() > expiry:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        expiry = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
        self._store[key] = (value, expiry)

    def size(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()


# =========================================================================== #
# Disk persistence (JSON-lines)                                               #
# =========================================================================== #

class _DiskSentimentCache:
    """
    Persistent JSON-lines store for inference results.

    File layout: one JSON object per line, keyed by content_hash.
    The file acts as an append log; on load it is deduplicated by key.

    Parameters
    ----------
    cache_dir : str | Path
    filename : str
        Name of the cache file within cache_dir.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        filename: str = "sentiment_cache.jsonl",
    ) -> None:
        self._path = Path(cache_dir) / filename
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._store: dict[str, dict] = {}
        self._load()

    # ------------------------------------------------------------------ #
    # Load / save                                                          #
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        """Read the JSONL file into memory (last write wins on key conflict)."""
        if not self._path.exists():
            return
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        key = record.get("content_hash")
                        if key:
                            self._store[key] = record
                    except json.JSONDecodeError:
                        pass
            logger.debug(
                "SentimentDiskCache: loaded %d entries from %s.",
                len(self._store), self._path,
            )
        except Exception as exc:
            logger.warning("SentimentDiskCache: failed to load %s — %s", self._path, exc)

    def _append(self, record: dict) -> None:
        """Append a single record to the JSONL file."""
        try:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(_make_serialisable(record)) + "\n")
        except Exception as exc:
            logger.warning("SentimentDiskCache: write error — %s", exc)

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def get(self, content_hash: str) -> Optional[dict]:
        return self._store.get(content_hash)

    def set(self, content_hash: str, result: dict) -> None:
        result = dict(result)
        result["content_hash"] = content_hash
        self._store[content_hash] = result
        self._append(result)

    def size(self) -> int:
        return len(self._store)

    def clear_memory(self) -> None:
        """Wipe in-memory dict only; disk file is preserved."""
        self._store.clear()


# =========================================================================== #
# SentimentCache — public interface                                            #
# =========================================================================== #

class SentimentCache:
    """
    Two-tier inference cache for ArticleSentimentResult objects.

    Usage
    -----
    cache = SentimentCache(cache_dir="data/cache/nlp")

    result = cache.get("abc123def456")
    if result is None:
        result = provider.classify(item)
        cache.set("abc123def456", result)

    Parameters
    ----------
    cache_dir : str | Path
        Directory where the disk cache file is stored.
        Defaults to "data/cache/nlp" relative to the working directory.
    mem_ttl : float
        In-memory TTL in seconds.
    use_disk : bool
        Set to False to disable persistence (useful in unit tests).
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/cache/nlp",
        mem_ttl: float = DEFAULT_MEM_TTL,
        use_disk: bool = True,
    ) -> None:
        self._mem = _TTLCache(default_ttl=mem_ttl)
        self._disk: Optional[_DiskSentimentCache] = (
            _DiskSentimentCache(cache_dir) if use_disk else None
        )
        logger.info(
            "SentimentCache: initialized (disk=%s, mem_ttl=%.0fs).",
            str(Path(cache_dir) / "sentiment_cache.jsonl") if use_disk else "disabled",
            mem_ttl,
        )

    # ------------------------------------------------------------------ #
    # Core get / set                                                       #
    # ------------------------------------------------------------------ #

    def get(self, content_hash: str) -> Optional[dict]:
        """
        Retrieve a cached ArticleSentimentResult by content hash.

        Checks memory first; promotes disk hits to memory.

        Parameters
        ----------
        content_hash : str
            16-char hash from schemas.compute_content_hash().

        Returns
        -------
        dict | None
            The cached result, or None on miss.
        """
        result = self._mem.get(content_hash)
        if result is not None:
            logger.debug("SentimentCache MEM HIT: %s", content_hash)
            return result

        if self._disk:
            result = self._disk.get(content_hash)
            if result is not None:
                logger.debug("SentimentCache DISK HIT: %s", content_hash)
                self._mem.set(content_hash, result)
                return result

        logger.debug("SentimentCache MISS: %s", content_hash)
        return None

    def set(self, content_hash: str, result: dict) -> None:
        """
        Store an ArticleSentimentResult in both memory and disk.

        Parameters
        ----------
        content_hash : str
        result : dict
            ArticleSentimentResult dict.
        """
        self._mem.set(content_hash, result)
        if self._disk:
            self._disk.set(content_hash, result)
        logger.debug("SentimentCache SET: %s", content_hash)

    # ------------------------------------------------------------------ #
    # Bulk helpers                                                         #
    # ------------------------------------------------------------------ #

    def get_batch(self, hashes: list[str]) -> dict[str, dict]:
        """
        Retrieve multiple results in one call.

        Parameters
        ----------
        hashes : list[str]

        Returns
        -------
        dict[str, dict]
            Mapping of content_hash → result for all cache hits.
            Misses are simply absent from the returned dict.
        """
        return {h: r for h in hashes if (r := self.get(h)) is not None}

    def set_batch(self, results: list[dict]) -> None:
        """
        Store a batch of ArticleSentimentResult dicts.

        Each dict must contain a 'content_hash' field.
        """
        for result in results:
            key = result.get("content_hash")
            if key:
                self.set(key, result)
            else:
                logger.debug("SentimentCache.set_batch: result missing content_hash, skipping.")

    # ------------------------------------------------------------------ #
    # Introspection                                                        #
    # ------------------------------------------------------------------ #

    def memory_size(self) -> int:
        return self._mem.size()

    def disk_size(self) -> int:
        return self._disk.size() if self._disk else 0

    def clear_memory(self) -> None:
        """Wipe in-memory cache only.  Disk entries remain."""
        self._mem.clear()
        logger.debug("SentimentCache: in-memory cache cleared.")


# =========================================================================== #
# Serialization helper                                                         #
# =========================================================================== #

def _make_serialisable(obj: Any) -> Any:
    """Recursively coerce non-JSON-serialisable types to strings."""
    from datetime import datetime
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serialisable(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj
