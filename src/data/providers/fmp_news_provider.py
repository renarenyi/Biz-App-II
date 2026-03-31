"""
fmp_news_provider.py
--------------------
Supplemental news provider using the Financial Modeling Prep (FMP) API.

The FMP free tier includes 250 API calls/day and provides historical stock
news with decent coverage. This significantly supplements Alpaca's limited
free-tier news archive.

API endpoint:
    GET https://financialmodelingprep.com/api/v3/stock_news
    ?tickers=AAPL&from=2025-01-01&to=2025-12-31&limit=250&apikey=YOUR_KEY

Register for a free API key at: https://financialmodelingprep.com/register
"""

from __future__ import annotations

import logging
import time
import requests
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.config.settings import settings
from src.data.utils import to_utc, clean_headline, now_utc

logger = logging.getLogger(__name__)


# =========================================================================== #
# Custom exception                                                             #
# =========================================================================== #

class FMPNewsProviderError(Exception):
    """Raised when the FMP News provider cannot serve a request."""


# =========================================================================== #
# FMPNewsProvider                                                              #
# =========================================================================== #

class FMPNewsProvider:
    """
    Adapter around the FMP Stock News REST API.

    Fetches news articles for a specific ticker over a given time window
    and normalizes them to the standard NewsArticle schema.

    Circuit-breaker:
      - MAX_FAILURES consecutive errors trigger a cooldown.
      - During cooldown, requests are rejected immediately.

    Parameters
    ----------
    api_key : str, optional
        FMP API key. Defaults to settings.FMP_API_KEY.
    """

    BASE_URL: str = "https://financialmodelingprep.com/api/v3/stock_news"
    MAX_FAILURES: int = 3
    COOLDOWN_SECONDS: float = float(settings.PROVIDER_COOLDOWN_SECONDS)
    MAX_ARTICLES_PER_REQUEST: int = 250  # FMP page size limit

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or getattr(settings, "FMP_API_KEY", "")
        self._failure_count: int = 0
        self._degraded_until: float = 0.0
        self._available: bool = bool(self._api_key)

        if not self._available:
            logger.debug(
                "FMPNewsProvider: API key not configured. "
                "Set FMP_API_KEY in .env to enable. "
                "Register at https://financialmodelingprep.com/register"
            )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def is_healthy(self) -> bool:
        if not self._available:
            return False
        if time.monotonic() < self._degraded_until:
            remaining = self._degraded_until - time.monotonic()
            logger.debug("FMPNewsProvider: degraded, %.0fs remaining.", remaining)
            return False
        return True

    def get_news(
        self,
        ticker: str,
        lookback_hours: int = 24,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 250,
    ) -> list[dict]:
        """
        Fetch news articles for a ticker from FMP.

        Parameters
        ----------
        ticker : str
        lookback_hours : int
            How many hours back to search. Used only if start/end not provided.
        start : datetime, optional
        end : datetime, optional
        limit : int
            Maximum number of articles to return (max 250).

        Returns
        -------
        list[dict]
            Normalized NewsArticle dicts.

        Raises
        ------
        FMPNewsProviderError
        """
        if not self.is_healthy():
            raise FMPNewsProviderError(
                "FMPNewsProvider is degraded or API key is missing."
            )

        end_dt = end or now_utc()
        start_dt = start or (end_dt - timedelta(hours=lookback_hours))

        logger.info(
            "FMPNewsProvider.get_news: ticker=%s, %s → %s, limit=%d",
            ticker, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"), limit,
        )

        try:
            t0 = time.monotonic()

            # FMP paginates with `page` param (0-indexed, max 100)
            all_articles = []
            for page in range(0, 5):  # Max 5 pages = 1250 articles
                params = {
                    "tickers": ticker,
                    "from": start_dt.strftime("%Y-%m-%d"),
                    "to": end_dt.strftime("%Y-%m-%d"),
                    "limit": min(limit, self.MAX_ARTICLES_PER_REQUEST),
                    "page": page,
                    "apikey": self._api_key,
                }

                response = requests.get(self.BASE_URL, params=params, timeout=15)
                response.raise_for_status()
                data = response.json()

                if not data or not isinstance(data, list):
                    break

                all_articles.extend(data)

                # Stop if we got fewer than limit (no more pages)
                if len(data) < min(limit, self.MAX_ARTICLES_PER_REQUEST):
                    break

            elapsed = time.monotonic() - t0

            logger.info(
                "FMPNewsProvider: received %d articles in %.2fs",
                len(all_articles), elapsed,
            )

            articles = [self._normalise(a, ticker) for a in all_articles]
            articles = [a for a in articles if a is not None]
            self._record_success()
            return articles

        except FMPNewsProviderError:
            raise
        except Exception as exc:
            self._record_failure(exc)
            raise FMPNewsProviderError(
                f"FMPNewsProvider.get_news failed for {ticker}: {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _normalise(self, item: dict, ticker: str) -> Optional[dict]:
        """Convert an FMP news dict to a standard NewsArticle dict."""
        try:
            headline = clean_headline(item.get("title", "") or "")
            if not headline:
                return None

            # Parse published date
            pub_str = item.get("publishedDate", "")
            pub_dt = None
            if pub_str:
                try:
                    pub_dt = datetime.strptime(pub_str, "%Y-%m-%d %H:%M:%S").replace(
                        tzinfo=timezone.utc
                    )
                except (ValueError, TypeError):
                    pub_dt = to_utc(pub_str)

            return {
                "ticker": ticker,
                "headline": headline,
                "summary": item.get("text", None),
                "source": item.get("site", None),
                "published_at": pub_dt,
                "url": item.get("url", None),
                "provider": "fmp",
                "author": None,
                "related_tickers": [ticker],
                "article_id": item.get("url", ""),  # FMP doesn't have unique IDs
                "ingested_at": now_utc(),
            }
        except Exception as exc:
            logger.debug("FMPNewsProvider: failed to normalise article: %s", exc)
            return None

    def _record_success(self) -> None:
        self._failure_count = 0

    def _record_failure(self, exc: Exception) -> None:
        self._failure_count += 1
        logger.warning(
            "FMPNewsProvider: failure #%d — %s", self._failure_count, exc
        )
        if self._failure_count >= self.MAX_FAILURES:
            self._degraded_until = time.monotonic() + self.COOLDOWN_SECONDS
            logger.error(
                "FMPNewsProvider: circuit breaker OPEN for %.0fs. Cause: %s",
                self.COOLDOWN_SECONDS, exc,
            )
