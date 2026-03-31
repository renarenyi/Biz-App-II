"""
finnhub_news_provider.py
------------------------
Supplemental news provider using the Finnhub Company News API.

Finnhub's free tier provides 60 API calls/minute and up to 1 year of
historical company news, significantly improving article coverage for
backtesting beyond what Alpaca and FMP alone provide.

API endpoint:
    GET https://finnhub.io/api/v1/company-news
    ?symbol=AAPL&from=2025-01-01&to=2025-12-31&token=YOUR_KEY

Register for a free API key at: https://finnhub.io/register
"""

from __future__ import annotations

import logging
import time
import requests
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.config.settings import settings
from src.data.utils import clean_headline, now_utc

logger = logging.getLogger(__name__)


# =========================================================================== #
# Custom exception                                                             #
# =========================================================================== #

class FinnhubNewsProviderError(Exception):
    """Raised when the Finnhub News provider cannot serve a request."""


# =========================================================================== #
# FinnhubNewsProvider                                                          #
# =========================================================================== #

class FinnhubNewsProvider:
    """
    Adapter around the Finnhub Company News REST API.

    Fetches news articles for a specific ticker over a given time window
    and normalizes them to the standard NewsArticle schema.

    Circuit-breaker:
      - MAX_FAILURES consecutive errors trigger a cooldown.
      - During cooldown, requests are rejected immediately.

    Parameters
    ----------
    api_key : str, optional
        Finnhub API key. Defaults to settings.FINNHUB_API_KEY.
    """

    BASE_URL: str = "https://finnhub.io/api/v1/company-news"
    MAX_FAILURES: int = 3
    COOLDOWN_SECONDS: float = float(settings.PROVIDER_COOLDOWN_SECONDS)

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or getattr(settings, "FINNHUB_API_KEY", "")
        self._failure_count: int = 0
        self._degraded_until: float = 0.0
        self._available: bool = bool(self._api_key)

        if not self._available:
            logger.debug(
                "FinnhubNewsProvider: API key not configured. "
                "Set FINNHUB_API_KEY in .env to enable. "
                "Register at https://finnhub.io/register"
            )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def is_healthy(self) -> bool:
        if not self._available:
            return False
        if time.monotonic() < self._degraded_until:
            remaining = self._degraded_until - time.monotonic()
            logger.debug("FinnhubNewsProvider: degraded, %.0fs remaining.", remaining)
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
        Fetch news articles for a ticker from Finnhub.

        Parameters
        ----------
        ticker : str
        lookback_hours : int
            How many hours back to search. Used only if start/end not provided.
        start : datetime, optional
        end : datetime, optional
        limit : int
            Maximum number of articles to return.

        Returns
        -------
        list[dict]
            Normalized NewsArticle dicts.

        Raises
        ------
        FinnhubNewsProviderError
        """
        if not self.is_healthy():
            raise FinnhubNewsProviderError(
                "FinnhubNewsProvider is degraded or API key is missing."
            )

        end_dt = end or now_utc()
        start_dt = start or (end_dt - timedelta(hours=lookback_hours))

        logger.info(
            "FinnhubNewsProvider.get_news: ticker=%s, %s → %s, limit=%d",
            ticker, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"), limit,
        )

        try:
            t0 = time.monotonic()

            params = {
                "symbol": ticker,
                "from": start_dt.strftime("%Y-%m-%d"),
                "to": end_dt.strftime("%Y-%m-%d"),
                "token": self._api_key,
            }

            response = requests.get(self.BASE_URL, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            elapsed = time.monotonic() - t0

            if not data or not isinstance(data, list):
                logger.info(
                    "FinnhubNewsProvider: empty response or unexpected format in %.2fs", elapsed
                )
                self._record_success()
                return []

            # Finnhub returns all matching articles, limit client-side
            if len(data) > limit:
                data = data[:limit]

            logger.info(
                "FinnhubNewsProvider: received %d articles in %.2fs",
                len(data), elapsed,
            )

            articles = [self._normalise(a, ticker) for a in data]
            articles = [a for a in articles if a is not None]
            self._record_success()

            logger.info(
                "Finnhub returned %d articles for %s",
                len(articles), ticker,
            )

            return articles

        except FinnhubNewsProviderError:
            raise
        except Exception as exc:
            self._record_failure(exc)
            raise FinnhubNewsProviderError(
                f"FinnhubNewsProvider.get_news failed for {ticker}: {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _normalise(self, item: dict, ticker: str) -> Optional[dict]:
        """Convert a Finnhub news dict to a standard NewsArticle dict."""
        try:
            headline = clean_headline(item.get("headline", "") or "")
            if not headline:
                return None

            # Finnhub returns datetime as Unix timestamp
            pub_ts = item.get("datetime")
            pub_dt = None
            if pub_ts:
                try:
                    pub_dt = datetime.fromtimestamp(int(pub_ts), tz=timezone.utc)
                except (ValueError, TypeError, OSError):
                    pub_dt = None

            return {
                "ticker": ticker,
                "headline": headline,
                "summary": item.get("summary", None),
                "source": item.get("source", None),
                "published_at": pub_dt,
                "url": item.get("url", None),
                "provider": "finnhub",
                "author": None,
                "related_tickers": item.get("related", "").split(",") if item.get("related") else [ticker],
                "article_id": str(item.get("id", item.get("url", ""))),
                "ingested_at": now_utc(),
            }
        except Exception as exc:
            logger.debug("FinnhubNewsProvider: failed to normalise article: %s", exc)
            return None

    def _record_success(self) -> None:
        self._failure_count = 0

    def _record_failure(self, exc: Exception) -> None:
        self._failure_count += 1
        logger.warning(
            "FinnhubNewsProvider: failure #%d — %s", self._failure_count, exc
        )
        if self._failure_count >= self.MAX_FAILURES:
            self._degraded_until = time.monotonic() + self.COOLDOWN_SECONDS
            logger.error(
                "FinnhubNewsProvider: circuit breaker OPEN for %.0fs. Cause: %s",
                self.COOLDOWN_SECONDS, exc,
            )
