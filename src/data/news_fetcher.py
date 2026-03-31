"""
news_fetcher.py
---------------
Central orchestrator for financial news data.

This is the single entry point downstream sentiment and strategy modules use
to get ticker-linked news. It abstracts away which news provider was used
and handles the entire Alpaca News → RSS fallback chain.

Public interface
----------------
    fetcher = NewsFetcher()
    articles = fetcher.get_recent_news("AAPL", lookback_hours=24)

Each article in the result is:
  - ticker-labelled
  - timestamp-normalized (UTC)
  - deduplicated (URL first, then headline similarity)
  - freshness-filtered (within lookback window)
  - schema-consistent (NewsArticle dict)
  - ready for Phase 2 sentiment analysis without further cleanup
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from src.config.settings import settings
from src.data.cache import DataCache
from src.data.providers.alpaca_news_provider import AlpacaNewsProvider, AlpacaNewsProviderError
from src.data.providers.fmp_news_provider import FMPNewsProvider, FMPNewsProviderError
from src.data.providers.finnhub_news_provider import FinnhubNewsProvider, FinnhubNewsProviderError
from src.data.providers.yahoo_news_provider import YahooNewsProvider, YahooNewsProviderError
from src.data.providers.rss_news_provider import RSSNewsProvider, RSSNewsProviderError
from src.data.schemas import validate_article, make_empty_articles_list
from src.data.utils import (
    get_logger,
    clean_headline,
    deduplicate_by_key,
    deduplicate_by_fingerprint,
    to_utc,
    now_utc,
)

logger = get_logger(__name__)


# =========================================================================== #
# NewsFetcher                                                                  #
# =========================================================================== #

class NewsFetcher:
    """
    Provider-agnostic news fetcher with deduplication and freshness filtering.

    Provider priority:
      1. Alpaca News API (if credentials are configured)
      2. RSS feeds (Yahoo Finance + general financial feeds)

    Deduplication strategy:
      1. By URL (exact match)
      2. By headline fingerprint (exact normalised match)
      3. By headline Jaccard similarity ≥ 0.85 (near-duplicate)

    Parameters
    ----------
    cache : DataCache, optional
    alpaca_news : AlpacaNewsProvider, optional
    rss_news : RSSNewsProvider, optional
    use_cache : bool
    min_articles : int
        Minimum count from primary provider before falling back.
    """

    def __init__(
        self,
        cache: Optional[DataCache] = None,
        alpaca_news: Optional[AlpacaNewsProvider] = None,
        fmp_news: Optional[FMPNewsProvider] = None,
        finnhub_news: Optional[FinnhubNewsProvider] = None,
        yahoo_news: Optional[YahooNewsProvider] = None,
        rss_news: Optional[RSSNewsProvider] = None,
        use_cache: bool = True,
        min_articles: int = None,
    ) -> None:
        self._cache = cache or DataCache(
            cache_dir=settings.CACHE_DIR,
            memory_ttl_news=settings.CACHE_TTL_NEWS,
        )
        self._alpaca_news = alpaca_news or AlpacaNewsProvider()
        self._fmp_news = fmp_news or FMPNewsProvider()
        self._finnhub_news = finnhub_news or FinnhubNewsProvider()
        self._yahoo_news = yahoo_news or YahooNewsProvider()
        self._rss_news = rss_news or RSSNewsProvider()
        self._use_cache = use_cache
        self._min_articles = min_articles if min_articles is not None else settings.MIN_NEWS_ARTICLES

    # ------------------------------------------------------------------ #
    # Primary public methods                                               #
    # ------------------------------------------------------------------ #

    def get_recent_news(
        self,
        ticker: str,
        lookback_hours: int = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        force_refresh: bool = False,
        include_rss: bool = True,
    ) -> list[dict]:
        """
        Fetch, normalize, deduplicate, and filter recent news for a ticker.

        Parameters
        ----------
        ticker : str
        lookback_hours : int, optional
            Ignored if start/end are provided.
        start : datetime, optional
            Explicit start time (UTC-aware).
        end : datetime, optional
            Explicit end time. Defaults to now.
        force_refresh : bool
            Bypass cache.
        include_rss : bool
            Whether to supplement Alpaca results with RSS if volume is low.

        Returns
        -------
        list[dict]
            Normalized, deduplicated, freshness-filtered NewsArticle dicts.
            Sorted newest-first by published_at.
            Empty list if all providers fail (does not raise).
        """
        lookback = lookback_hours if lookback_hours is not None else settings.DEFAULT_NEWS_LOOKBACK_HOURS
        end_dt = end or now_utc()
        start_dt = start or (end_dt - timedelta(hours=lookback))

        cache_key = self._news_cache_key(ticker, start_dt, end_dt)

        # --- Try cache ---
        if self._use_cache and not force_refresh:
            cached = self._cache.get_articles(cache_key)
            if cached is not None:
                logger.info(
                    "NewsFetcher: cache HIT for %s news (%d articles)", ticker, len(cached)
                )
                return cached

        all_articles: list[dict] = []
        sources_tried: list[str] = []

        # --- Primary: Alpaca News ---
        alpaca_articles = self._try_alpaca_news(ticker, start_dt, end_dt)
        if alpaca_articles is not None:
            sources_tried.append("alpaca_news")
            all_articles.extend(alpaca_articles)
            logger.info(
                "NewsFetcher: Alpaca News returned %d articles for %s",
                len(alpaca_articles), ticker,
            )

        # --- Supplement: FMP News (always try for maximum coverage) ---
        fmp_articles = self._try_fmp_news(ticker, start_dt, end_dt)
        if fmp_articles is not None:
            sources_tried.append("fmp")
            all_articles.extend(fmp_articles)
            logger.info(
                "NewsFetcher: FMP returned %d articles for %s",
                len(fmp_articles), ticker,
            )

        # --- Supplement: Finnhub News (always try for maximum coverage) ---
        finnhub_articles = self._try_finnhub_news(ticker, start_dt, end_dt)
        if finnhub_articles is not None:
            sources_tried.append("finnhub")
            all_articles.extend(finnhub_articles)
            logger.info(
                "NewsFetcher: Finnhub returned %d articles for %s",
                len(finnhub_articles), ticker,
            )

        # --- Supplement / fallback: RSS ---
        use_rss = (
            include_rss
            and len(all_articles) < self._min_articles
        )

        if use_rss:
            reason = "primary unavailable" if alpaca_articles is None else f"only {len(alpaca_articles)} articles (min={self._min_articles})"
            logger.info("NewsFetcher: supplementing with RSS for %s (%s).", ticker, reason)
            rss_articles = self._try_rss_news(ticker, start_dt, end_dt)
            if rss_articles is not None:
                sources_tried.append("rss")
                all_articles.extend(rss_articles)
                logger.info(
                    "NewsFetcher: RSS returned %d articles for %s",
                    len(rss_articles), ticker,
                )

        if not all_articles:
            logger.warning(
                "NewsFetcher: no articles found for %s (sources tried: %s)",
                ticker, sources_tried,
            )
            return make_empty_articles_list()

        # --- Normalize, deduplicate, filter ---
        articles = self.normalize_articles(all_articles, ticker)
        articles = self.deduplicate_articles(articles)
        articles = self.filter_stale_articles(articles, start=start_dt, end=end_dt)

        # --- Sort newest first ---
        articles = sorted(
            articles,
            key=lambda a: a.get("published_at") or datetime.min.replace(tzinfo=None),
            reverse=True,
        )

        logger.info(
            "NewsFetcher: returning %d articles for %s "
            "(after dedup/filter, sources=%s)",
            len(articles), ticker, sources_tried,
        )

        # --- Cache ---
        if self._use_cache:
            self._cache.set_articles(cache_key, articles, ttl=settings.CACHE_TTL_NEWS)

        return articles

    # ------------------------------------------------------------------ #
    # Normalization, dedup, filtering — exposed for testing               #
    # ------------------------------------------------------------------ #

    def normalize_articles(
        self,
        articles: list[dict],
        ticker: str,
    ) -> list[dict]:
        """
        Normalize a list of raw article dicts to the NewsArticle schema.

        Ensures:
        - ticker field is set
        - headline is cleaned
        - published_at is UTC-aware datetime
        - invalid articles are dropped with a warning

        Parameters
        ----------
        articles : list[dict]
        ticker : str

        Returns
        -------
        list[dict]
        """
        normalized: list[dict] = []
        for raw in articles:
            try:
                article = dict(raw)

                # Enforce ticker
                article["ticker"] = ticker

                # Clean headline
                article["headline"] = clean_headline(article.get("headline", "") or "")

                # Normalize published_at to UTC datetime
                pub = article.get("published_at")
                if pub is not None:
                    article["published_at"] = to_utc(pub)
                else:
                    logger.debug(
                        "NewsFetcher: article missing published_at: %s",
                        article.get("headline", "")[:60],
                    )

                # Validate
                if not validate_article(article, raise_on_error=False):
                    continue

                normalized.append(article)

            except Exception as exc:
                logger.debug("NewsFetcher: error normalizing article: %s", exc)

        return normalized

    def deduplicate_articles(self, articles: list[dict]) -> list[dict]:
        """
        Remove duplicate articles using a three-pass strategy:

        1. Exact URL deduplication
        2. Exact headline fingerprint deduplication
        3. Near-duplicate headline detection (Jaccard ≥ 0.85)

        Parameters
        ----------
        articles : list[dict]

        Returns
        -------
        list[dict]
        """
        before = len(articles)

        # Pass 1: URL dedup
        articles = deduplicate_by_key(articles, key="url")

        # Pass 2: article_id dedup
        articles = deduplicate_by_key(articles, key="article_id")

        # Pass 3: Near-duplicate headline dedup
        articles = deduplicate_by_fingerprint(articles, threshold=0.85)

        after = len(articles)
        if before != after:
            logger.info(
                "NewsFetcher: deduplicated %d → %d articles (removed %d)",
                before, after, before - after,
            )

        return articles

    def filter_stale_articles(
        self,
        articles: list[dict],
        lookback_hours: int = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> list[dict]:
        """
        Remove articles outside the requested time window.

        Articles with a missing or unparseable published_at are retained
        (we don't drop what we can't evaluate — caller can decide).

        Parameters
        ----------
        articles : list[dict]
        lookback_hours : int, optional
            Used only if start/end are not provided.
        start : datetime, optional
        end : datetime, optional

        Returns
        -------
        list[dict]
        """
        end_dt = end or now_utc()
        if start is not None:
            start_dt = start
        elif lookback_hours is not None:
            start_dt = end_dt - timedelta(hours=lookback_hours)
        else:
            start_dt = end_dt - timedelta(hours=settings.DEFAULT_NEWS_LOOKBACK_HOURS)

        kept: list[dict] = []
        stale_count = 0

        for article in articles:
            pub = article.get("published_at")
            if pub is None:
                kept.append(article)  # can't filter without a date
                continue

            pub_utc = to_utc(pub)
            if pub_utc is None:
                kept.append(article)
                continue

            if start_dt <= pub_utc <= end_dt:
                kept.append(article)
            else:
                stale_count += 1

        if stale_count:
            logger.debug(
                "NewsFetcher: filtered out %d stale articles (window: %s → %s)",
                stale_count, start_dt.isoformat(), end_dt.isoformat(),
            )

        return kept

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _try_alpaca_news(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
    ) -> Optional[list[dict]]:
        """Attempt Alpaca News. Returns None on failure."""
        if not self._alpaca_news.is_healthy():
            logger.debug("NewsFetcher: Alpaca News not healthy, skipping.")
            return None
        try:
            return self._alpaca_news.get_news(ticker, start=start, end=end)
        except AlpacaNewsProviderError as exc:
            logger.warning("NewsFetcher: Alpaca News failed — %s", exc)
            return None

    def _try_fmp_news(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
    ) -> Optional[list[dict]]:
        """Attempt FMP News. Returns None on failure."""
        if not self._fmp_news.is_healthy():
            logger.debug("NewsFetcher: FMP News not healthy, skipping.")
            return None
        try:
            return self._fmp_news.get_news(ticker, start=start, end=end)
        except FMPNewsProviderError as exc:
            logger.warning("NewsFetcher: FMP News failed — %s", exc)
            return None

    def _try_finnhub_news(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
    ) -> Optional[list[dict]]:
        """Attempt Finnhub News. Returns None on failure."""
        if not self._finnhub_news.is_healthy():
            logger.debug("NewsFetcher: Finnhub News not healthy, skipping.")
            return None
        try:
            return self._finnhub_news.get_news(ticker, start=start, end=end)
        except FinnhubNewsProviderError as exc:
            logger.warning("NewsFetcher: Finnhub News failed — %s", exc)
            return None

    def _try_rss_news(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
    ) -> Optional[list[dict]]:
        """Attempt RSS. Returns None on complete failure."""
        try:
            return self._rss_news.get_news(ticker, start=start, end=end)
        except RSSNewsProviderError as exc:
            logger.warning("NewsFetcher: RSS failed — %s", exc)
            return None

    def _try_yahoo_news(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
    ) -> Optional[list[dict]]:
        """Attempt Yahoo/Google News RSS. Returns None on failure."""
        if not self._yahoo_news.is_healthy():
            logger.debug("NewsFetcher: Yahoo/Google News not healthy, skipping.")
            return None
        try:
            return self._yahoo_news.get_news(ticker, start=start, end=end)
        except YahooNewsProviderError as exc:
            logger.warning("NewsFetcher: Yahoo/Google News failed — %s", exc)
            return None

    @staticmethod
    def _news_cache_key(ticker: str, start: datetime, end: datetime) -> str:
        start_str = start.strftime("%Y%m%d%H") if start else "open"
        end_str = end.strftime("%Y%m%d%H") if end else "now"
        return f"news_{ticker}_{start_str}_{end_str}"
