"""
settings.py
-----------
Central configuration for the Trading Agent.
All credentials and tuneable parameters are read from environment variables.
Never hardcode secrets here. See .env.example for the full list.

Usage:
    from src.config.settings import settings
    key = settings.ALPACA_API_KEY
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)


class Settings:
    """
    Immutable-at-runtime configuration object.
    Values are read once at import time from environment variables.
    """

    # ------------------------------------------------------------------ #
    # Alpaca Credentials                                                   #
    # ------------------------------------------------------------------ #
    ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
    ALPACA_API_SECRET: str = os.getenv("ALPACA_API_SECRET", os.getenv("ALPACA_SECRET_KEY", ""))
    ALPACA_SECRET_KEY: str = ALPACA_API_SECRET  # Alias for components that look for this name
    # Paper trading endpoint — never point this at the live endpoint
    ALPACA_BASE_URL: str = os.getenv(
        "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
    )
    ALPACA_DATA_URL: str = os.getenv(
        "ALPACA_DATA_URL", "https://data.alpaca.markets"
    )

    # ------------------------------------------------------------------ #
    # FMP (Financial Modeling Prep) Credentials                            #
    # ------------------------------------------------------------------ #
    FMP_API_KEY: str = os.getenv("FMP_API_KEY", "")

    # ------------------------------------------------------------------ #
    # Finnhub Credentials                                                  #
    # ------------------------------------------------------------------ #
    FINNHUB_API_KEY: str = os.getenv("FINNHUB_API_KEY", "")

    # ------------------------------------------------------------------ #
    # Caching                                                              #
    # ------------------------------------------------------------------ #
    CACHE_DIR: str = os.getenv("CACHE_DIR", str(_PROJECT_ROOT / "data" / "cache"))
    # TTL values in seconds
    CACHE_TTL_LATEST_PRICE: int = int(os.getenv("CACHE_TTL_LATEST_PRICE", "30"))
    CACHE_TTL_INTRADAY_BARS: int = int(os.getenv("CACHE_TTL_INTRADAY_BARS", "300"))   # 5 min
    CACHE_TTL_DAILY_BARS: int = int(os.getenv("CACHE_TTL_DAILY_BARS", "3600"))        # 1 hour
    CACHE_TTL_NEWS: int = int(os.getenv("CACHE_TTL_NEWS", "600"))                     # 10 min

    # ------------------------------------------------------------------ #
    # Data defaults                                                        #
    # ------------------------------------------------------------------ #
    DEFAULT_TIMEZONE: str = os.getenv("DEFAULT_TIMEZONE", "America/New_York")
    DEFAULT_TIMEFRAME: str = os.getenv("DEFAULT_TIMEFRAME", "1Day")
    DEFAULT_NEWS_LOOKBACK_HOURS: int = int(
        os.getenv("DEFAULT_NEWS_LOOKBACK_HOURS", "24")
    )
    # Minimum articles to consider a news fetch valid before trying backup
    MIN_NEWS_ARTICLES: int = int(os.getenv("MIN_NEWS_ARTICLES", "3"))

    # ------------------------------------------------------------------ #
    # Circuit-breaker / retry behaviour                                    #
    # ------------------------------------------------------------------ #
    # How long (seconds) to wait before retrying a degraded provider
    PROVIDER_COOLDOWN_SECONDS: int = int(
        os.getenv("PROVIDER_COOLDOWN_SECONDS", "120")
    )
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_BACKOFF_BASE: float = float(os.getenv("RETRY_BACKOFF_BASE", "1.5"))

    # ------------------------------------------------------------------ #
    # RSS news sources                                                     #
    # ------------------------------------------------------------------ #
    # Comma-separated list of RSS URLs. {ticker} will be substituted.
    RSS_FEED_TEMPLATE: str = os.getenv(
        "RSS_FEED_TEMPLATE",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
    )
    # General financial news RSS feeds (not ticker-specific)
    RSS_GENERAL_FEEDS: str = os.getenv(
        "RSS_GENERAL_FEEDS",
        (
            "https://feeds.a.dj.com/rss/RSSMarketsMain.xml,"
            "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069"
        ),
    )

    # ------------------------------------------------------------------ #
    # Logging                                                              #
    # ------------------------------------------------------------------ #
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv(
        "LOG_FORMAT",
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """
        Warn loudly if credentials are missing.
        Does not raise — the system can still use yfinance / RSS fallbacks.
        """
        _logger = logging.getLogger(__name__)
        if not self.ALPACA_API_KEY or not self.ALPACA_API_SECRET:
            _logger.warning(
                "ALPACA_API_KEY or ALPACA_API_SECRET not set. "
                "Alpaca providers will be unavailable; yfinance / RSS fallbacks will be used."
            )

    def __repr__(self) -> str:
        key_preview = self.ALPACA_API_KEY[:4] + "****" if self.ALPACA_API_KEY else "(not set)"
        return (
            f"Settings("
            f"ALPACA_API_KEY={key_preview}, "
            f"CACHE_DIR={self.CACHE_DIR}, "
            f"DEFAULT_TIMEZONE={self.DEFAULT_TIMEZONE}"
            f")"
        )


# Module-level singleton — import this everywhere
settings = Settings()
