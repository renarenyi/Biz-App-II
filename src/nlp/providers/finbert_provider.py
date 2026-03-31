"""
nlp/providers/finbert_provider.py
----------------------------------
FinBERT sentiment provider — the fast, deterministic baseline.

Model
-----
ProsusAI/finbert  (HuggingFace Hub)
  - Fine-tuned BERT on financial news (Financial PhraseBank + FiQA)
  - Three-way classifier: positive / negative / neutral
  - 512-token input limit
  - CPU-runnable; GPU optional

Design choices
--------------
- Lazy model load: the transformer is downloaded and cached on first
  classify_articles() call, not at import time.
- Batch processing: articles are classified in configurable batches
  to avoid OOM on long document lists.
- Conviction score mapping: FinBERT's raw softmax probability for the
  dominant class is scaled to [0, 10] for consistency with the schema.
  A probability of 1.0 → score 10.0; 0.5 → score 5.0.
- Deterministic: temperature is not applicable; results are deterministic
  for fixed model weights, which is correct for backtesting.

Label mapping
-------------
FinBERT output     → ArticleSentimentResult label
"positive"         → "POSITIVE"
"negative"         → "NEGATIVE"
"neutral"          → "NEUTRAL"

No event_type is extracted by FinBERT — that field is left None.
The Llama/Mistral providers can supply richer event classification.

Dependencies
------------
transformers >= 4.30
torch >= 2.0  (or tensorflow — transformers auto-detects)
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

_FINBERT_MODEL_NAME = "ProsusAI/finbert"
_LABEL_MAP = {"positive": "POSITIVE", "negative": "NEGATIVE", "neutral": "NEUTRAL"}


class FinBERTProvider(BaseSentimentProvider):
    """
    HuggingFace FinBERT sentiment classifier.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.  Override for local checkpoints.
    batch_size : int
        Articles to classify per inference call.
    device : str | None
        "cpu", "cuda", "mps", or None to auto-detect.
    max_length : int
        Token truncation limit.  FinBERT max is 512.
    """

    def __init__(
        self,
        model_name: str = _FINBERT_MODEL_NAME,
        batch_size: int = 16,
        device: Optional[str] = None,
        max_length: int = 512,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._device = device
        self._max_length = max_length

        # Lazy-loaded components
        self._pipeline = None
        self._load_error: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Abstract interface                                                   #
    # ------------------------------------------------------------------ #

    @property
    def provider_name(self) -> str:
        return "finbert"

    def is_available(self) -> bool:
        """
        Returns True if the transformers pipeline is loaded.

        Attempts a lazy load if not yet initialized.
        """
        if self._pipeline is not None:
            return True
        if self._load_error:
            return False
        try:
            self._load_pipeline()
            return True
        except Exception:
            return False

    def classify_articles(self, news_items: list[dict]) -> list[dict]:
        """
        Classify a list of NewsItem dicts using FinBERT.

        Parameters
        ----------
        news_items : list[dict]

        Returns
        -------
        list[dict]
            ArticleSentimentResult dicts.

        Raises
        ------
        ProviderUnavailableError
            If the pipeline cannot be loaded.
        ProviderInferenceError
            For unexpected batch-level failures.
        """
        if not news_items:
            return []

        if self._pipeline is None:
            try:
                self._load_pipeline()
            except Exception as exc:
                raise ProviderUnavailableError(
                    f"FinBERTProvider: failed to load pipeline — {exc}"
                ) from exc

        results: list[dict] = []
        total = len(news_items)

        for batch_start in range(0, total, self._batch_size):
            batch = news_items[batch_start: batch_start + self._batch_size]
            texts = [
                (item.get("analysis_text") or item.get("headline", ""))[:self._max_length]
                for item in batch
            ]

            try:
                raw_outputs = self._pipeline(
                    texts,
                    truncation=True,
                    max_length=self._max_length,
                    padding=True,
                )
            except Exception as exc:
                raise ProviderInferenceError(
                    f"FinBERTProvider: pipeline inference error — {exc}"
                ) from exc

            for item, output in zip(batch, raw_outputs):
                try:
                    label = _LABEL_MAP.get(output["label"].lower(), "NEUTRAL")
                    # Scale softmax probability → [0, 10] conviction
                    score = round(float(output["score"]) * 10.0, 2)

                    result = self._build_result(
                        item=item,
                        sentiment=label,
                        conviction_score=score,
                        reasoning=(
                            f"FinBERT probability: {output['score']:.3f} for class '{output['label']}'."
                        ),
                        event_type=None,
                    )
                    results.append(result)

                except Exception as exc:
                    logger.debug(
                        "FinBERTProvider: error processing result for '%s' — %s",
                        item.get("headline", "")[:60], exc,
                    )

        self._log_inference(total, len(results))
        return results

    # ------------------------------------------------------------------ #
    # Lazy model loading                                                   #
    # ------------------------------------------------------------------ #

    def _load_pipeline(self) -> None:
        """
        Download (or load from HuggingFace cache) and initialize the pipeline.

        Called at most once per process lifetime.
        """
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError as exc:
            self._load_error = "transformers not installed"
            raise ProviderUnavailableError(
                "FinBERTProvider: 'transformers' package is not installed. "
                "Run: pip install transformers torch"
            ) from exc

        logger.info(
            "FinBERTProvider: loading model '%s' (device=%s) …",
            self._model_name, self._device or "auto",
        )

        device_arg = self._device
        if device_arg is None:
            # Auto-detect without requiring torch at module import time
            try:
                import torch
                if torch.cuda.is_available():
                    device_arg = 0   # CUDA device index
                else:
                    device_arg = -1  # CPU
            except ImportError:
                device_arg = -1

        try:
            self._pipeline = hf_pipeline(
                task="text-classification",
                model=self._model_name,
                device=device_arg,
                truncation=True,
                max_length=self._max_length,
            )
            logger.info(
                "FinBERTProvider: model loaded successfully (device=%s).",
                device_arg,
            )
        except Exception as exc:
            self._load_error = str(exc)
            raise ProviderUnavailableError(
                f"FinBERTProvider: model load failed — {exc}"
            ) from exc
