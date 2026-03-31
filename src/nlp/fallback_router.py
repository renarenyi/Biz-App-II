"""
nlp/fallback_router.py
-----------------------
Model routing and fallback chain for the Phase 2 sentiment layer.

Responsibilities
----------------
- Maintain an ordered list of BaseSentimentProvider instances
- Attempt inference in priority order
- On ProviderUnavailableError: log and advance to the next provider
- On ProviderInferenceError: log and advance to the next provider
- Record which provider handled each batch (audit trail)
- Raise FallbackExhaustedError if all providers fail

Fallback order (default)
------------------------
1. FinBERT     — fast, deterministic, finance-tuned baseline
2. Llama 3 8B  — richer reasoning and event classification
3. Mistral 7B  — secondary LLM option

The router is constructed with a default chain in SentimentAgent,
but callers can inject any ordered list of providers for testing or
to swap in a custom provider.

Logging contract
----------------
- Every provider attempt is logged at DEBUG level
- Every provider failure is logged at WARNING level with the exception
- Provider success is logged at INFO level
- The selected provider name is recorded in the returned results under
  the 'provider' field (set by BaseSentimentProvider._build_result)

Design notes
------------
- The router does NOT cache — caching is handled by SentimentAgent
- The router does NOT preprocess — preprocessing is done upstream
- If a provider returns a partial result list (fewer results than
  input items), the router logs the discrepancy but returns what it has
- A provider that raises mid-batch causes a full retry on the next provider
  (to keep the result set internally consistent for the aggregation layer)
"""

from __future__ import annotations

import logging
from typing import Optional

from src.nlp.providers.base_provider import (
    BaseSentimentProvider,
    ProviderUnavailableError,
    ProviderInferenceError,
)

logger = logging.getLogger(__name__)


class FallbackExhaustedError(Exception):
    """
    Raised when all providers in the chain have been exhausted
    without producing a result.

    The SentimentAgent should catch this and return an empty result set
    rather than propagating the exception to the strategy engine.
    """
    pass


class FallbackRouter:
    """
    Ordered provider chain with automatic failover.

    Parameters
    ----------
    providers : list[BaseSentimentProvider]
        Ordered list of providers.  Index 0 is tried first.
    """

    def __init__(self, providers: list[BaseSentimentProvider]) -> None:
        if not providers:
            raise ValueError("FallbackRouter: at least one provider is required.")
        self._providers = providers
        self._last_used: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def last_used_provider(self) -> Optional[str]:
        """Name of the provider that handled the most recent batch."""
        return self._last_used

    @property
    def available_providers(self) -> list[str]:
        """Names of providers that currently report as available."""
        return [p.provider_name for p in self._providers if p.is_available()]

    # ------------------------------------------------------------------ #
    # Core routing                                                         #
    # ------------------------------------------------------------------ #

    def classify(self, news_items: list[dict]) -> list[dict]:
        """
        Classify a batch of NewsItems using the first available provider.

        Tries each provider in order.  On any exception, logs the failure
        and tries the next provider.

        Parameters
        ----------
        news_items : list[dict]
            Preprocessed NewsItem dicts.

        Returns
        -------
        list[dict]
            ArticleSentimentResult dicts from the first successful provider.

        Raises
        ------
        FallbackExhaustedError
            If every provider fails.
        """
        if not news_items:
            return []

        errors: list[str] = []

        for provider in self._providers:
            provider_name = provider.provider_name

            # Cheap availability check before attempting expensive inference
            logger.debug("FallbackRouter: checking availability of '%s'.", provider_name)
            if not provider.is_available():
                msg = f"{provider_name}: is_available() returned False."
                logger.warning("FallbackRouter: skipping provider — %s", msg)
                errors.append(msg)
                continue

            logger.info(
                "FallbackRouter: attempting inference with '%s' on %d items.",
                provider_name, len(news_items),
            )

            try:
                results = provider.classify_articles(news_items)
            except ProviderUnavailableError as exc:
                msg = f"{provider_name}: ProviderUnavailableError — {exc}"
                logger.warning("FallbackRouter: provider failed — %s", msg)
                errors.append(msg)
                continue
            except ProviderInferenceError as exc:
                msg = f"{provider_name}: ProviderInferenceError — {exc}"
                logger.warning("FallbackRouter: provider failed — %s", msg)
                errors.append(msg)
                continue
            except Exception as exc:
                msg = f"{provider_name}: unexpected error — {exc}"
                logger.warning("FallbackRouter: provider failed — %s", msg)
                errors.append(msg)
                continue

            # Sanity check: warn if fewer results than inputs
            if len(results) < len(news_items):
                logger.warning(
                    "FallbackRouter: '%s' returned %d results for %d inputs. "
                    "Some articles may have been skipped.",
                    provider_name, len(results), len(news_items),
                )

            self._last_used = provider_name
            logger.info(
                "FallbackRouter: '%s' succeeded — %d results returned.",
                provider_name, len(results),
            )
            return results

        # All providers failed
        error_summary = " | ".join(errors) if errors else "no providers configured"
        raise FallbackExhaustedError(
            f"FallbackRouter: all providers exhausted. Errors: {error_summary}"
        )

    # ------------------------------------------------------------------ #
    # Factory helpers                                                      #
    # ------------------------------------------------------------------ #

    @classmethod
    def default_chain(
        cls,
        llama_model_path: Optional[str] = None,
        mistral_model_path: Optional[str] = None,
    ) -> "FallbackRouter":
        """
        Build the default provider chain using FinBERT.

        FinBERT is the sole production model — it is deterministic,
        free, and runs on CPU with no API keys required.

        Parameters are kept for backward compatibility but ignored.

        Returns
        -------
        FallbackRouter
        """
        from src.nlp.providers.finbert_provider import FinBERTProvider

        providers: list[BaseSentimentProvider] = [
            FinBERTProvider(),
        ]

        logger.info(
            "FallbackRouter.default_chain: created with providers: %s",
            [p.provider_name for p in providers],
        )
        return cls(providers=providers)

    @classmethod
    def finbert_only(cls) -> "FallbackRouter":
        """
        Convenience factory for FinBERT-only operation.

        Use when:
        - Running in a constrained environment without large model support
        - Rapid backtesting that needs throughput over explanation quality
        - Unit tests that should not attempt real model loading
        """
        from src.nlp.providers.finbert_provider import FinBERTProvider
        return cls(providers=[FinBERTProvider()])
