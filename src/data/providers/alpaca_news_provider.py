"""
alpaca_news_provider.py
-----------------------
Primary news provider using the Alpaca News API (alpaca-py SDK).

The Alpaca free tier includes access to news data via the same API key used
for market data. This provider fetches headlines and summaries for a given
ticker within a lookback window.

SDK: alpaca-py
  from alpaca.data.historical.news import NewsClient
  from alpaca.data.requests import NewsRequest

Falls back gracefully when credentials are missing or the API errors.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.config.settings import settings
from src.data.utils import to_utc, clean_headline, now_utc

logger = logging.getLogger(__name__)


# =========================================================================== #
# Custom exception                                                             #
# =========================================================================== #

class AlpacaNewsProviderError(Exception):
    """Raised when the Alpaca News provider cannot serve a request."""


# =========================================================================== #
# AlpacaNewsProvider                                                           #
# =========================================================================== #

class AlpacaNewsProvider:
    """
    Adapter around the alpaca-py NewsClient.

    Fetches news articles for a specific ticker over a given time window
    and normalizes them to the standard NewsArticle schema.

    Circuit-breaker:
      - MAX_FAILURES consecutive errors trigger a cooldown.
      - During cooldown, requests are rejected immediately.

    Parameters
    ----------
    api_key : str, optional
    api_secret : str, optional
    """

    MAX_FAILURES: int = 3
    COOLDOWN_SECONDS: float = float(settings.PROVIDER_COOLDOWN_SECONDS)
    MAX_ARTICLES_PER_REQUEST: int = 50  # Alpaca default page size

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ) -> None:
        self._api_key = api_key or settings.ALPACA_API_KEY
        self._api_secret = api_secret or settings.ALPACA_API_SECRET

        self._client = None
        self._failure_count: int = 0
        self._degraded_until: float = 0.0
        self._available: bool = bool(self._api_key and self._api_secret)

        if not self._available:
            logger.warning(
                "AlpacaNewsProvider: credentials not configured. "
                "This provider will be skipped."
            )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def is_healthy(self) -> bool:
        if not self._available:
            return False
        if time.monotonic() < self._degraded_until:
            remaining = self._degraded_until - time.monotonic()
            logger.debug("AlpacaNewsProvider: degraded, %.0fs remaining.", remaining)
            return False
        return True

    def get_news(
        self,
        ticker: str,
        lookback_hours: int = 24,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 50,
    ) -> list[dict]:
        """
        Fetch recent news articles for a ticker.

        Automatically paginates through all available results using
        Alpaca's next_page_token mechanism. The Alpaca News API has
        historical data back to 2015, but returns max 50 per request.

        Parameters
        ----------
        ticker : str
        lookback_hours : int
            How many hours back to search. Used only if start/end not provided.
        start : datetime, optional
            Explicit start time (UTC-aware).
        end : datetime, optional
            Explicit end time (UTC-aware). Defaults to now.
        limit : int
            Articles per page (max 50). All pages are fetched automatically.

        Returns
        -------
        list[dict]
            Normalized NewsArticle dicts.

        Raises
        ------
        AlpacaNewsProviderError
        """
        if not self.is_healthy():
            raise AlpacaNewsProviderError(
                "AlpacaNewsProvider is degraded or credentials are missing."
            )

        try:
            from alpaca.data.historical.news import NewsClient
            from alpaca.data.requests import NewsRequest
        except ImportError:
            raise AlpacaNewsProviderError(
                "alpaca-py is not installed. Run: pip install alpaca-py"
            )

        client = self._get_client()

        end_dt = end or now_utc()
        start_dt = start or (end_dt - timedelta(hours=lookback_hours))

        logger.info(
            "AlpacaNewsProvider.get_news: ticker=%s, %s → %s (paginated)",
            ticker, start_dt.isoformat(), end_dt.isoformat(),
        )

        try:
            t0 = time.monotonic()
            all_raw_articles = []
            page_token = None
            page_count = 0
            max_pages = 200  # Safety limit: 200 pages × 50 = 10,000 articles max

            while page_count < max_pages:
                request_kwargs = dict(
                    symbols=ticker,
                    start=start_dt,
                    end=end_dt,
                    limit=min(limit, self.MAX_ARTICLES_PER_REQUEST),
                    sort="DESC",
                )
                if page_token:
                    request_kwargs["page_token"] = page_token

                request = NewsRequest(**request_kwargs)
                news_response = client.get_news(request)

                raw_articles = list(news_response.news) if hasattr(news_response, "news") else list(news_response)
                all_raw_articles.extend(raw_articles)
                page_count += 1

                # Check for next page
                next_token = getattr(news_response, "next_page_token", None)
                if not next_token or len(raw_articles) == 0:
                    break
                page_token = next_token

                # Rate limit protection: Alpaca allows 200 req/min
                # Small delay to avoid hitting rate limits on large fetches
                if page_count % 50 == 0:
                    logger.info(
                        "AlpacaNewsProvider: fetched %d articles across %d pages, continuing...",
                        len(all_raw_articles), page_count,
                    )
                    time.sleep(1.0)

            elapsed = time.monotonic() - t0

            logger.info(
                "AlpacaNewsProvider: received %d articles in %.2fs (%d pages)",
                len(all_raw_articles), elapsed, page_count,
            )

            articles = [self._normalise(a, ticker) for a in all_raw_articles]
            articles = [a for a in articles if a is not None]
            self._record_success()
            return articles

        except AlpacaNewsProviderError:
            raise
        except Exception as exc:
            self._record_failure(exc)
            raise AlpacaNewsProviderError(
                f"AlpacaNewsProvider.get_news failed for {ticker}: {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _get_client(self):
        if self._client is None:
            from alpaca.data.historical.news import NewsClient
            self._client = NewsClient(
                api_key=self._api_key,
                secret_key=self._api_secret,
            )
        return self._client

    def _normalise(self, item, ticker: str) -> Optional[dict]:
        """Convert an alpaca-py NewsItem to a standard NewsArticle dict."""
        try:
            headline = clean_headline(getattr(item, "headline", "") or "")
            if not headline:
                logger.debug("AlpacaNewsProvider: skipping article with empty headline.")
                return None

            return {
                "ticker": ticker,
                "headline": headline,
                "summary": getattr(item, "summary", None),
                "source": getattr(item, "source", None),
                "published_at": to_utc(getattr(item, "created_at", None) or getattr(item, "updated_at", None)),
                "url": getattr(item, "url", None),
                "provider": "alpaca_news",
                "author": getattr(item, "author", None),
                "related_tickers": list(getattr(item, "symbols", []) or []),
                "article_id": str(getattr(item, "id", "") or ""),
                "ingested_at": now_utc(),
            }
        except Exception as exc:
            logger.debug("AlpacaNewsProvider: failed to normalise article: %s", exc)
            return None

    def _record_success(self) -> None:
        self._failure_count = 0

    def _record_failure(self, exc: Exception) -> None:
        self._failure_count += 1
        logger.warning(
            "AlpacaNewsProvider: failure #%d — %s", self._failure_count, exc
        )
        if self._failure_count >= self.MAX_FAILURES:
            self._degraded_until = time.monotonic() + self.COOLDOWN_SECONDS
            logger.error(
                "AlpacaNewsProvider: circuit breaker OPEN for %.0fs. Cause: %s",
                self.COOLDOWN_SECONDS, exc,
            )
