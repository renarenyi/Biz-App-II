"""
rss_news_provider.py
--------------------
Fallback / supplementary news provider using public RSS feeds.

This provider requires zero API keys and provides:
  1. Ticker-specific RSS from Yahoo Finance
  2. General financial market RSS from WSJ Markets and CNBC

All feeds are parsed with feedparser. Malformed or empty feeds are skipped
without crashing the pipeline — consistent with the circuit-breaker mindset.

SDK: feedparser (pip install feedparser)
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Optional
from urllib.parse import quote

from src.config.settings import settings
from src.data.utils import clean_headline, to_utc, now_utc

logger = logging.getLogger(__name__)


# =========================================================================== #
# Default feed catalogue                                                       #
# =========================================================================== #

# Ticker-specific feed URL templates (use {ticker} placeholder)
_TICKER_FEED_TEMPLATES = [
    # Yahoo Finance headline RSS — well-maintained, free, ticker-specific
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
]

# General financial news feeds (not ticker-specific — we filter by mention)
_GENERAL_FEEDS = [
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",                                   # WSJ Markets
    "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069",  # CNBC Markets
    "https://feeds.reuters.com/reuters/businessNews",                                    # Reuters Business
]


# =========================================================================== #
# Custom exception                                                             #
# =========================================================================== #

class RSSNewsProviderError(Exception):
    """Raised when all RSS feeds fail for a ticker."""


# =========================================================================== #
# RSSNewsProvider                                                              #
# =========================================================================== #

class RSSNewsProvider:
    """
    Multi-feed RSS news provider with graceful per-feed error handling.

    Strategy:
    1. Fetch ticker-specific Yahoo Finance RSS.
    2. Optionally fetch general financial feeds and filter for mentions of the ticker.
    3. Merge, normalize, and return all articles.

    Per-feed errors are caught and logged — the provider continues with
    results from healthy feeds. Only raises if zero articles are collected
    from all feeds.

    Parameters
    ----------
    include_general_feeds : bool
        Whether to also query general financial RSS feeds. Increases coverage
        but may add noise (requires ticker mention filtering).
    request_timeout : float
        HTTP timeout per feed in seconds.
    """

    def __init__(
        self,
        include_general_feeds: bool = True,
        request_timeout: float = 10.0,
    ) -> None:
        self.include_general_feeds = include_general_feeds
        self.request_timeout = request_timeout

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def get_news(
        self,
        ticker: str,
        lookback_hours: int = 24,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> list[dict]:
        """
        Fetch news articles from RSS feeds for a ticker.

        Parameters
        ----------
        ticker : str
        lookback_hours : int
        start : datetime, optional
        end : datetime, optional

        Returns
        -------
        list[dict]
            Normalized NewsArticle dicts.

        Raises
        ------
        RSSNewsProviderError
            If no feeds returned any articles.
        """
        try:
            import feedparser
        except ImportError:
            raise RSSNewsProviderError(
                "feedparser is not installed. Run: pip install feedparser"
            )

        end_dt = end or now_utc()
        start_dt = start or (end_dt - timedelta(hours=lookback_hours))

        logger.info(
            "RSSNewsProvider.get_news: ticker=%s, window=%s → %s",
            ticker, start_dt.isoformat(), end_dt.isoformat(),
        )

        all_articles: list[dict] = []
        feed_errors: list[str] = []

        # ---- Ticker-specific feeds ---- #
        for template in _TICKER_FEED_TEMPLATES:
            url = template.format(ticker=quote(ticker, safe=""))
            articles, error = self._fetch_feed(
                url, ticker=ticker, start=start_dt, end=end_dt, require_ticker_mention=False
            )
            if error:
                feed_errors.append(f"{url}: {error}")
            all_articles.extend(articles)

        # ---- General feeds (optional) ---- #
        if self.include_general_feeds:
            for url in _GENERAL_FEEDS:
                articles, error = self._fetch_feed(
                    url, ticker=ticker, start=start_dt, end=end_dt, require_ticker_mention=True
                )
                if error:
                    feed_errors.append(f"{url}: {error}")
                all_articles.extend(articles)

        logger.info(
            "RSSNewsProvider: collected %d articles for %s "
            "(%d feed errors: %s)",
            len(all_articles),
            ticker,
            len(feed_errors),
            feed_errors if feed_errors else "none",
        )

        if not all_articles and feed_errors:
            raise RSSNewsProviderError(
                f"All RSS feeds failed for {ticker}. Errors: {feed_errors}"
            )

        return all_articles

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _fetch_feed(
        self,
        url: str,
        ticker: str,
        start: datetime,
        end: datetime,
        require_ticker_mention: bool = False,
    ) -> tuple[list[dict], Optional[str]]:
        """
        Fetch and parse a single RSS feed.

        Returns
        -------
        tuple[list[dict], Optional[str]]
            (articles, error_message_or_None)
        """
        try:
            import feedparser

            logger.debug("RSSNewsProvider: fetching %s", url)
            t0 = time.monotonic()

            feed = feedparser.parse(url, request_headers={"User-Agent": "TradingAgentBot/1.0"})
            elapsed = time.monotonic() - t0

            if feed.bozo and feed.bozo_exception:
                # bozo = feedparser's flag for malformed feeds
                # Many valid feeds still parse with minor errors — only fail hard on empty
                logger.debug("Feed bozo exception (%s): %s", url, feed.bozo_exception)

            if not feed.entries:
                return [], f"Empty feed (status={getattr(feed, 'status', 'N/A')})"

            logger.debug(
                "RSSNewsProvider: parsed %d entries from %s in %.2fs",
                len(feed.entries), url, elapsed,
            )

            articles = []
            feed_name = getattr(feed.feed, "title", url)

            for entry in feed.entries:
                article = self._normalise_entry(entry, ticker, feed_name)
                if article is None:
                    continue

                # Filter by time window
                pub = article.get("published_at")
                if pub and not (start <= pub <= end):
                    continue

                # Filter by ticker mention if required
                if require_ticker_mention:
                    if not self._ticker_mentioned(article, ticker):
                        continue

                articles.append(article)

            return articles, None

        except Exception as exc:
            logger.warning("RSSNewsProvider: error fetching %s — %s", url, exc)
            return [], str(exc)

    def _normalise_entry(self, entry, ticker: str, feed_name: str) -> Optional[dict]:
        """Convert a feedparser entry to a standard NewsArticle dict."""
        try:
            headline = clean_headline(
                getattr(entry, "title", "") or ""
            )
            if not headline:
                return None

            # Parse publication date
            published_at = self._parse_date(entry)

            summary = getattr(entry, "summary", None) or getattr(entry, "description", None)
            if summary:
                # Strip basic HTML tags from RSS summaries
                summary = re.sub(r"<[^>]+>", "", summary).strip()
                summary = clean_headline(summary)

            url = getattr(entry, "link", None) or getattr(entry, "id", None)
            article_id = getattr(entry, "id", None) or url

            return {
                "ticker": ticker,
                "headline": headline,
                "summary": summary or None,
                "source": feed_name,
                "published_at": published_at,
                "url": url,
                "provider": "rss",
                "author": getattr(entry, "author", None),
                "language": "en",
                "related_tickers": [],
                "article_id": str(article_id) if article_id else None,
                "ingested_at": now_utc(),
            }

        except Exception as exc:
            logger.debug("RSSNewsProvider: failed to normalise entry: %s", exc)
            return None

    def _parse_date(self, entry) -> Optional[datetime]:
        """
        Parse publication date from a feedparser entry.

        Tries multiple date fields in priority order.
        """
        for field in ("published", "updated", "created"):
            raw = getattr(entry, f"{field}_parsed", None)
            if raw is not None:
                try:
                    # feedparser returns a time.struct_time
                    import calendar
                    ts = calendar.timegm(raw)
                    return datetime.fromtimestamp(ts, tz=timezone.utc)
                except Exception:
                    pass

            # Also try the raw string
            raw_str = getattr(entry, field, None)
            if raw_str:
                dt = to_utc(raw_str)
                if dt:
                    return dt

        return None

    def _ticker_mentioned(self, article: dict, ticker: str) -> bool:
        """Check if the ticker appears in the headline or summary."""
        text = (article.get("headline", "") + " " + (article.get("summary") or "")).upper()
        # Match as a word boundary to avoid false positives (e.g. 'SPY' in 'SPYDER')
        pattern = r"\b" + re.escape(ticker.upper()) + r"\b"
        return bool(re.search(pattern, text))
