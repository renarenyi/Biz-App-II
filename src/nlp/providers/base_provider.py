"""
nlp/providers/base_provider.py
-------------------------------
Abstract base class for all sentiment model providers.

Every provider (FinBERT, Llama, Mistral, or a future OpenAI wrapper)
must implement this interface so the FallbackRouter can treat them
interchangeably.

Contract
--------
- classify_articles(news_items) → list[ArticleSentimentResult]
- is_available() → bool   (health check without inference overhead)
- provider_name property

Providers must:
- Return results in the exact ArticleSentimentResult schema
- Set the 'provider' field on every result they produce
- Set 'content_hash' on every result if the input item has one
- Set 'inferred_at' (UTC) on every result
- Never raise for individual article failures — log and skip
- Raise ProviderUnavailableError if the underlying model is unreachable
- Raise ProviderInferenceError for unexpected model failures
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional

from src.nlp.schemas import (
    ArticleSentimentResult,
    NewsItem,
    VALID_SENTIMENT_LABELS,
    clamp_conviction,
)

logger = logging.getLogger(__name__)


# =========================================================================== #
# Custom exceptions                                                            #
# =========================================================================== #

class ProviderUnavailableError(Exception):
    """
    Raised when a provider cannot be contacted or initialized.

    The FallbackRouter catches this to move to the next provider.
    """
    pass


class ProviderInferenceError(Exception):
    """
    Raised when a provider encounters an unexpected failure during inference
    that is not simply unavailability.

    Callers can decide whether to retry or skip to the next provider.
    """
    pass


# =========================================================================== #
# BaseSentimentProvider                                                        #
# =========================================================================== #

class BaseSentimentProvider(ABC):
    """
    Abstract interface for financial news sentiment providers.

    Subclasses implement:
    - classify_articles(news_items) → list[ArticleSentimentResult]
    - is_available() → bool

    All other methods in this base class are utilities shared across
    all providers.
    """

    # ------------------------------------------------------------------ #
    # Abstract interface                                                   #
    # ------------------------------------------------------------------ #

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable name used in logs and result metadata."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """
        Quick health check.

        Returns True if the provider is ready to run inference.
        Should be cheap (no model call) — e.g., check if model is loaded.
        """
        ...

    @abstractmethod
    def classify_articles(
        self,
        news_items: list[dict],
    ) -> list[dict]:
        """
        Classify a batch of NewsItem dicts.

        Parameters
        ----------
        news_items : list[dict]
            Preprocessed NewsItem dicts from nlp.preprocessing.

        Returns
        -------
        list[dict]
            ArticleSentimentResult dicts — one per input item.
            Items that fail individually are skipped (logged, not raised).

        Raises
        ------
        ProviderUnavailableError
            If the model is not available at call time.
        ProviderInferenceError
            For unexpected batch-level failures.
        """
        ...

    # ------------------------------------------------------------------ #
    # Shared helpers available to all subclasses                          #
    # ------------------------------------------------------------------ #

    def _build_result(
        self,
        item: dict,
        sentiment: str,
        conviction_score: float,
        reasoning: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> dict:
        """
        Construct a validated ArticleSentimentResult dict.

        Parameters
        ----------
        item : dict
            The input NewsItem this result was derived from.
        sentiment : str
            Must be one of VALID_SENTIMENT_LABELS.
        conviction_score : float
            Raw score; will be clamped to [0, 10].
        reasoning : str, optional
        event_type : str, optional

        Returns
        -------
        dict (ArticleSentimentResult)
        """
        # Normalize and validate label
        sentiment = str(sentiment).upper().strip()
        if sentiment not in VALID_SENTIMENT_LABELS:
            logger.debug(
                "%s: invalid sentiment label '%s', defaulting to NEUTRAL.",
                self.provider_name, sentiment,
            )
            sentiment = "NEUTRAL"

        score = clamp_conviction(conviction_score)

        result: dict = {
            "ticker": item.get("ticker", ""),
            "headline": item.get("headline", ""),
            "sentiment": sentiment,
            "conviction_score": score,
            "reasoning": reasoning,
            "event_type": event_type,
            "published_at": item.get("published_at"),
            "source": item.get("source"),
            "provider": self.provider_name,
            "content_hash": item.get("content_hash"),
            "inferred_at": datetime.now(tz=timezone.utc),
            "relevance_score": item.get("relevance_score", 1.0),
        }
        return result

    def _log_inference(self, n_items: int, n_results: int) -> None:
        logger.info(
            "%s: classified %d / %d articles.",
            self.provider_name, n_results, n_items,
        )
