"""
yahoo_news_provider.py
----------------------
Historical news provider using Google News RSS feeds.

Google News RSS provides structured XML news results with date-range filtering,
making it ideal for backfilling historical news coverage that Alpaca and Finnhub
free tiers do not provide.

Endpoint format:
    GET https://news.google.com/rss/search
    ?q="Apple"+stock+after:2025-04-01+before:2025-05-01
    &hl=en-US&gl=US&ceid=US:en

No API key required. Rate-limited to 1 request/second to be respectful.
"""

from __future__ import annotations

import logging
import time
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Optional
from email.utils import parsedate_to_datetime

import requests

from src.data.utils import clean_headline, now_utc

logger = logging.getLogger(__name__)


# =========================================================================== #
# Custom exception                                                             #
# =========================================================================== #

class YahooNewsProviderError(Exception):
    """Raised when the Yahoo/Google News provider cannot serve a request."""


# =========================================================================== #
# Ticker → search term mapping                                                 #
# =========================================================================== #

_TICKER_SEARCH_TERMS: dict[str, str] = {
    "AAPL":  '"Apple" stock',
    "MSFT":  '"Microsoft" stock',
    "GOOGL": '"Google" OR "Alphabet" stock',
    "GOOG":  '"Google" OR "Alphabet" stock',
    "NVDA":  '"Nvidia" stock',
    "AMZN":  '"Amazon" stock',
    "META":  '"Meta" OR "Facebook" stock',
    "TSLA":  '"Tesla" stock',
    "AMD":   '"AMD" OR "Advanced Micro Devices" stock',
    "INTC":  '"Intel" stock',
    "NFLX":  '"Netflix" stock',
    "CRM":   '"Salesforce" stock',
}


# =========================================================================== #
# YahooNewsProvider                                                            #
# =========================================================================== #

class YahooNewsProvider:
    """
    Adapter that scrapes Google News RSS for historical financial headlines.

    Iterates month-by-month over the requested date range to maximise coverage.
    Each monthly query returns up to ~100 articles.

    Circuit-breaker:
      - MAX_FAILURES consecutive errors trigger a cooldown.
      - During cooldown, requests are rejected immediately.
    """

    BASE_URL: str = "https://news.google.com/rss/search"
    MAX_FAILURES: int = 3
    COOLDOWN_SECONDS: float = 120.0
    REQUEST_DELAY: float = 1.5  # seconds between requests (rate limiting)

    def __init__(self) -> None:
        self._failure_count: int = 0
        self._degraded_until: float = 0.0
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        })

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def is_healthy(self) -> bool:
        if time.monotonic() < self._degraded_until:
            remaining = self._degraded_until - time.monotonic()
            logger.debug("YahooNewsProvider: degraded, %.0fs remaining.", remaining)
            return False
        return True

    def get_news(
        self,
        ticker: str,
        lookback_hours: int = 24,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 500,
    ) -> list[dict]:
        """
        Fetch historical news articles for a ticker via Google News RSS.

        Iterates month-by-month over [start, end] to maximise coverage.

        Parameters
        ----------
        ticker : str
        lookback_hours : int
            Fallback if start/end not provided.
        start : datetime, optional
        end : datetime, optional
        limit : int
            Maximum total articles to return.

        Returns
        -------
        list[dict]
            Normalized NewsArticle dicts.

        Raises
        ------
        YahooNewsProviderError
        """
        if not self.is_healthy():
            raise YahooNewsProviderError(
                "YahooNewsProvider is degraded (circuit breaker open)."
            )

        end_dt = end or now_utc()
        start_dt = start or (end_dt - timedelta(hours=lookback_hours))

        search_term = _TICKER_SEARCH_TERMS.get(
            ticker.upper(),
            f'"{ticker}" stock',
        )

        logger.info(
            "YahooNewsProvider.get_news: ticker=%s, %s → %s, query=%s",
            ticker, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"),
            search_term,
        )

        # Split into monthly chunks for better coverage
        chunks = self._monthly_chunks(start_dt, end_dt)
        all_articles: list[dict] = []

        for chunk_start, chunk_end in chunks:
            if len(all_articles) >= limit:
                break

            try:
                articles = self._fetch_chunk(
                    search_term, ticker, chunk_start, chunk_end
                )
                all_articles.extend(articles)
                logger.debug(
                    "YahooNewsProvider: chunk %s → %s returned %d articles",
                    chunk_start.strftime("%Y-%m-%d"),
                    chunk_end.strftime("%Y-%m-%d"),
                    len(articles),
                )
            except Exception as exc:
                logger.warning(
                    "YahooNewsProvider: chunk %s → %s failed: %s",
                    chunk_start.strftime("%Y-%m-%d"),
                    chunk_end.strftime("%Y-%m-%d"),
                    exc,
                )
                self._record_failure(exc)
                if self._failure_count >= self.MAX_FAILURES:
                    break

            # Rate limiting between chunks
            time.sleep(self.REQUEST_DELAY)

        # Trim to limit
        if len(all_articles) > limit:
            all_articles = all_articles[:limit]

        logger.info(
            "YahooNewsProvider: total %d articles for %s across %d monthly chunks",
            len(all_articles), ticker, len(chunks),
        )

        self._record_success()
        return all_articles

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _monthly_chunks(
        self, start: datetime, end: datetime
    ) -> list[tuple[datetime, datetime]]:
        """Split a date range into monthly chunks."""
        chunks = []
        current = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)

        while current < end:
            # Next month
            if current.month == 12:
                next_month = current.replace(year=current.year + 1, month=1)
            else:
                next_month = current.replace(month=current.month + 1)

            chunk_start = max(current, start)
            chunk_end = min(next_month, end)
            chunks.append((chunk_start, chunk_end))
            current = next_month

        return chunks

    def _fetch_chunk(
        self,
        search_term: str,
        ticker: str,
        start: datetime,
        end: datetime,
    ) -> list[dict]:
        """Fetch one monthly chunk from Google News RSS."""
        # Build query with date range
        query = (
            f"{search_term} "
            f"after:{start.strftime('%Y-%m-%d')} "
            f"before:{end.strftime('%Y-%m-%d')}"
        )

        params = {
            "q": query,
            "hl": "en-US",
            "gl": "US",
            "ceid": "US:en",
        }

        response = self._session.get(
            self.BASE_URL, params=params, timeout=15
        )
        response.raise_for_status()

        # Parse RSS XML
        return self._parse_rss(response.text, ticker)

    def _parse_rss(self, xml_text: str, ticker: str) -> list[dict]:
        """Parse Google News RSS XML into NewsArticle dicts."""
        articles = []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            logger.warning("YahooNewsProvider: XML parse error: %s", exc)
            return []

        # RSS structure: <rss><channel><item>...</item></channel></rss>
        channel = root.find("channel")
        if channel is None:
            return []

        for item in channel.findall("item"):
            try:
                article = self._parse_item(item, ticker)
                if article is not None:
                    articles.append(article)
            except Exception as exc:
                logger.debug(
                    "YahooNewsProvider: failed to parse item: %s", exc
                )

        return articles

    def _parse_item(self, item: ET.Element, ticker: str) -> Optional[dict]:
        """Parse a single RSS <item> into a NewsArticle dict."""
        title_el = item.find("title")
        if title_el is None or not title_el.text:
            return None

        # Google News titles often have " - Source Name" appended
        raw_title = title_el.text.strip()
        source = None

        # Extract source from title (e.g., "Headline here - Reuters")
        if " - " in raw_title:
            parts = raw_title.rsplit(" - ", 1)
            headline_text = parts[0].strip()
            source = parts[1].strip() if len(parts) > 1 else None
        else:
            headline_text = raw_title

        headline = clean_headline(headline_text)
        if not headline:
            return None

        # Parse published date
        pub_el = item.find("pubDate")
        pub_dt = None
        if pub_el is not None and pub_el.text:
            try:
                pub_dt = parsedate_to_datetime(pub_el.text.strip())
                if pub_dt.tzinfo is None:
                    pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                else:
                    pub_dt = pub_dt.astimezone(timezone.utc)
            except Exception:
                pub_dt = None

        # Extract URL
        link_el = item.find("link")
        url = link_el.text.strip() if link_el is not None and link_el.text else None

        # Extract description/snippet
        desc_el = item.find("description")
        description = None
        if desc_el is not None and desc_el.text:
            # Google News descriptions contain HTML — strip tags
            description = re.sub(r"<[^>]+>", " ", desc_el.text)
            description = re.sub(r"\s+", " ", description).strip()
            if len(description) > 500:
                description = description[:500]

        return {
            "ticker": ticker,
            "headline": headline,
            "summary": description,
            "source": source,
            "published_at": pub_dt,
            "url": url,
            "provider": "google_news",
            "author": None,
            "related_tickers": [ticker],
            "article_id": url or headline[:80],
            "ingested_at": now_utc(),
        }

    def _record_success(self) -> None:
        self._failure_count = 0

    def _record_failure(self, exc: Exception) -> None:
        self._failure_count += 1
        logger.warning(
            "YahooNewsProvider: failure #%d — %s", self._failure_count, exc
        )
        if self._failure_count >= self.MAX_FAILURES:
            self._degraded_until = time.monotonic() + self.COOLDOWN_SECONDS
            logger.error(
                "YahooNewsProvider: circuit breaker OPEN for %.0fs. Cause: %s",
                self.COOLDOWN_SECONDS, exc,
            )
