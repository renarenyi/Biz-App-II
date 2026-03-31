"""
nlp/sentiment_agent.py
-----------------------
Top-level orchestrator for Phase 2 — the LLM Sentiment Agent.

This is the single public entry point that Phase 3 (strategy engine)
and backtesting code should call.

Public interface
----------------
    agent = SentimentAgent()
    result = agent.analyze(ticker="TSLA", articles=articles)

    # result is a TickerSentimentResult dict:
    # {
    #   "ticker": "TSLA",
    #   "sentiment": "NEGATIVE",
    #   "conviction_score": 7.6,
    #   "reasoning": "...",
    #   "source_count": 5,
    #   "unique_event_count": 2,
    #   "analysis_window_hours": 24,
    #   "generated_at": datetime(...),
    #   "provider_used": "finbert",
    #   "article_results": [...]
    # }

Data flow
---------
articles (Phase 1 NewsArticle dicts)
    │
    ▼
preprocessing.prepare_news_items()
    │  normalize, dedup, filter, hash
    ▼
cache lookup (SentimentCache)
    │  split into cache hits vs misses
    ▼
FallbackRouter.classify()   ← only for cache misses
    │  FinBERT → Llama → Mistral
    ▼
cache.set_batch()           ← persist new results
    │
    ▼
aggregation.aggregate_to_ticker()
    │  time-decayed, event-cluster-weighted
    ▼
TickerSentimentResult  →  Phase 3

Design principles
-----------------
- Fail gracefully: if no provider is available, return an empty
  NEUTRAL result rather than raising.  The strategy engine should
  treat a missing signal as "no trade."
- Reproducible: results are cached by content hash, not by time.
  Running the same articles twice produces the same result.
- Auditable: every result carries its source provider, content hash,
  and inferred_at timestamp.
- Phase-isolated: this module imports from src.nlp only.
  It consumes plain dicts from Phase 1 (no Phase 1 class imports at runtime).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from src.nlp.schemas import (
    TickerSentimentResult,
    make_empty_ticker_result,
    validate_ticker_result,
)
from src.nlp.preprocessing import prepare_news_items
from src.nlp.cache import SentimentCache
from src.nlp.fallback_router import FallbackRouter, FallbackExhaustedError
from src.nlp.aggregation import aggregate_to_ticker

logger = logging.getLogger(__name__)


class SentimentAgent:
    """
    Orchestrator for the Phase 2 sentiment pipeline.

    Parameters
    ----------
    router : FallbackRouter, optional
        Provider chain.  Defaults to FallbackRouter.default_chain().
        Pass FallbackRouter.finbert_only() for fast deterministic runs.
    cache : SentimentCache, optional
        Inference result cache.  Defaults to SentimentCache().
    window_hours : int
        Default news analysis window in hours.
    cache_dir : str
        Directory for the disk inference cache (used only when cache is None).
    use_disk_cache : bool
        Whether to persist inference results to disk.

    Examples
    --------
    # Default — uses FinBERT (deterministic, CPU-runnable)
    agent = SentimentAgent()
    result = agent.analyze("TSLA", articles)

    # FinBERT-only, no disk cache (fast for unit tests)
    agent = SentimentAgent(
        router=FallbackRouter.finbert_only(),
        use_disk_cache=False,
    )
    """

    def __init__(
        self,
        router: Optional[FallbackRouter] = None,
        cache: Optional[SentimentCache] = None,
        window_hours: int = 24,
        cache_dir: str = "data/cache/nlp",
        use_disk_cache: bool = True,
    ) -> None:
        self._router = router or FallbackRouter.default_chain()
        self._cache = cache or SentimentCache(
            cache_dir=cache_dir,
            use_disk=use_disk_cache,
        )
        self._window_hours = window_hours

        logger.info(
            "SentimentAgent: initialized (window=%dh, cache_dir=%s, disk=%s).",
            window_hours, cache_dir, use_disk_cache,
        )

    # ------------------------------------------------------------------ #
    # Primary public method                                                #
    # ------------------------------------------------------------------ #

    def analyze(
        self,
        ticker: str,
        articles: list[dict],
        window_hours: Optional[int] = None,
        reference_time: Optional[datetime] = None,
    ) -> dict:
        """
        Run the full Phase 2 pipeline for one ticker.

        Parameters
        ----------
        ticker : str
            Target stock symbol (e.g., "TSLA").
        articles : list[dict]
            Raw NewsArticle dicts from Phase 1 NewsFetcher.
        window_hours : int, optional
            Override the agent's default window_hours.
        reference_time : datetime, optional
            Reference point for recency calculations.  Defaults to now (UTC).

        Returns
        -------
        dict (TickerSentimentResult)
            Always returns a valid TickerSentimentResult — never raises.
            If no articles survive preprocessing or all providers fail,
            returns a neutral NEUTRAL/0 result.
        """
        window = window_hours if window_hours is not None else self._window_hours
        ref = reference_time or datetime.now(tz=timezone.utc)

        logger.info(
            "SentimentAgent.analyze: starting for %s "
            "(%d raw articles, window=%dh, ref=%s)",
            ticker, len(articles), window, ref.isoformat(),
        )

        # ── Step 1: Preprocess ──────────────────────────────────────── #
        news_items = prepare_news_items(
            articles=articles,
            ticker=ticker,
            window_hours=window,
            reference_time=ref,
        )

        if not news_items:
            logger.warning(
                "SentimentAgent: 0 items after preprocessing for %s. "
                "Returning empty signal.", ticker,
            )
            return make_empty_ticker_result(ticker, window)

        # ── Step 2: Cache split ─────────────────────────────────────── #
        hashes = [item.get("content_hash", "") for item in news_items]
        cached_map = self._cache.get_batch(hashes)

        fresh_items = [
            item for item in news_items
            if item.get("content_hash", "") not in cached_map
        ]
        cached_results = list(cached_map.values())

        logger.info(
            "SentimentAgent: %d cached hits, %d items need inference.",
            len(cached_results), len(fresh_items),
        )

        # ── Step 3: Inference for cache misses ──────────────────────── #
        new_results: list[dict] = []
        provider_used: Optional[str] = None

        if fresh_items:
            try:
                new_results = self._router.classify(fresh_items)
                provider_used = self._router.last_used_provider
                # Persist new results
                self._cache.set_batch(new_results)
            except FallbackExhaustedError as exc:
                logger.error(
                    "SentimentAgent: all providers exhausted for %s — %s. "
                    "Will aggregate from cached results only.",
                    ticker, exc,
                )
                # Fall through with whatever cached results we have

        # ── Step 4: Aggregate ───────────────────────────────────────── #
        all_results = cached_results + new_results

        if not all_results:
            logger.warning(
                "SentimentAgent: no results to aggregate for %s. "
                "Returning empty signal.", ticker,
            )
            return make_empty_ticker_result(ticker, window)

        ticker_result = aggregate_to_ticker(
            article_results=all_results,
            ticker=ticker,
            window_hours=window,
            reference_time=ref,
            provider_used=provider_used or (
                cached_results[0].get("provider") if cached_results else None
            ),
        )

        # ── Step 5: Validate and return ─────────────────────────────── #
        if not validate_ticker_result(ticker_result, raise_on_error=False):
            logger.error(
                "SentimentAgent: TickerSentimentResult validation failed for %s. "
                "Returning empty signal.", ticker,
            )
            return make_empty_ticker_result(ticker, window)

        logger.info(
            "SentimentAgent: result for %s — %s / %.1f conviction "
            "(%d articles, %d events, provider=%s)",
            ticker,
            ticker_result["sentiment"],
            ticker_result["conviction_score"],
            ticker_result["source_count"],
            ticker_result["unique_event_count"],
            ticker_result.get("provider_used", "n/a"),
        )

        return ticker_result

    # ------------------------------------------------------------------ #
    # Batch analysis (multiple tickers)                                   #
    # ------------------------------------------------------------------ #

    def analyze_batch(
        self,
        ticker_articles: dict[str, list[dict]],
        window_hours: Optional[int] = None,
        reference_time: Optional[datetime] = None,
    ) -> dict[str, dict]:
        """
        Analyze multiple tickers.

        Parameters
        ----------
        ticker_articles : dict[str, list[dict]]
            Mapping of ticker → list of articles.
        window_hours : int, optional
        reference_time : datetime, optional

        Returns
        -------
        dict[str, dict]
            Mapping of ticker → TickerSentimentResult.
        """
        results: dict[str, dict] = {}
        for ticker, articles in ticker_articles.items():
            results[ticker] = self.analyze(
                ticker=ticker,
                articles=articles,
                window_hours=window_hours,
                reference_time=reference_time,
            )
        return results

    # ------------------------------------------------------------------ #
    # Introspection                                                        #
    # ------------------------------------------------------------------ #

    def cache_stats(self) -> dict:
        """Return cache size statistics."""
        return {
            "memory_entries": self._cache.memory_size(),
            "disk_entries": self._cache.disk_size(),
        }

    @property
    def available_providers(self) -> list[str]:
        """Names of currently available model providers."""
        return self._router.available_providers
