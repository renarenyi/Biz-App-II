"""
src/nlp/providers
-----------------
Model provider implementations for Phase 2 sentiment analysis.

Available providers:
  FinBERTProvider   — finance-tuned BERT classifier (CPU-runnable, deterministic)

Use FallbackRouter to compose a chain automatically.
"""

from src.nlp.providers.base_provider import (
    BaseSentimentProvider,
    ProviderUnavailableError,
    ProviderInferenceError,
)
from src.nlp.providers.finbert_provider import FinBERTProvider

__all__ = [
    "BaseSentimentProvider",
    "ProviderUnavailableError",
    "ProviderInferenceError",
    "FinBERTProvider",
]
