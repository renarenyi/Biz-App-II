"""
src/nlp
-------
Phase 2: LLM Sentiment Agent

Public API:
  from src.nlp.sentiment_agent import SentimentAgent
  from src.nlp.schemas import TickerSentimentResult, ArticleSentimentResult, NewsItem
  from src.nlp.fallback_router import FallbackRouter
"""

from src.nlp.sentiment_agent import SentimentAgent
from src.nlp.schemas import (
    NewsItem,
    ArticleSentimentResult,
    TickerSentimentResult,
    make_empty_ticker_result,
)
from src.nlp.fallback_router import FallbackRouter

__all__ = [
    "SentimentAgent",
    "NewsItem",
    "ArticleSentimentResult",
    "TickerSentimentResult",
    "make_empty_ticker_result",
    "FallbackRouter",
]
