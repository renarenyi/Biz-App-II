"""
nlp/preprocessing.py
--------------------
Text preprocessing pipeline for the Phase 2 sentiment layer.

Responsibilities
----------------
1. Convert raw Phase 1 NewsArticle dicts → typed NewsItem dicts
2. Normalize and clean headline + snippet text
3. Deduplicate at URL, article_id, and headline-fingerprint level
4. Merge headline + snippet into a single 'analysis_text' field
5. Compute content hashes for downstream caching
6. Filter articles outside the requested time window

This module is stateless — every function is a pure transformation.
No model calls happen here.  No Phase 1 imports needed at runtime
(the NewsArticle dict is treated as a plain dict).

Design notes
------------
- Deduplication order: URL → article_id → headline fingerprint → Jaccard
- We reuse the identical Jaccard/fingerprint helpers from Phase 1 utils
  (imported from src.data.utils) so logic stays DRY.
- 'analysis_text' is truncated at MAX_ANALYSIS_CHARS to prevent
  accidentally blowing up model context windows with very long snippets.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from datetime import datetime, timezone, timedelta
from typing import Optional

from src.nlp.schemas import (
    NewsItem,
    compute_content_hash,
    validate_news_item,
    NEWS_ITEM_REQUIRED,
)
from src.data.utils import (
    to_utc,
    headline_fingerprint,
    jaccard_similarity,
    deduplicate_by_key,
)

logger = logging.getLogger(__name__)

# Maximum characters for the combined analysis_text string
# ~512 tokens at typical tokenisation rates — fine for FinBERT (512 limit) and
# sufficient context for local LLMs without risk of runaway prompts.
MAX_ANALYSIS_CHARS = 800
JACCARD_NEAR_DUP_THRESHOLD = 0.85

# NER Keywords for Relevance Scoring
COMPANY_KEYWORDS = {
    "AAPL": {
        "primary": ["apple", "aapl", "tim cook", "iphone", "macbook", "ipad", "vision pro"],
        "competitors": ["microsoft", "google", "samsung", "meta"]
    },
    "MSFT": {
        "primary": ["microsoft", "msft", "satya nadella", "windows", "azure", "xbox", "copilot"],
        "competitors": ["apple", "google", "amazon", "aws"]
    },
    "GOOGL": {
        "primary": ["google", "googl", "alphabet", "sundar pichai", "youtube", "android", "gemini", "pixel"],
        "competitors": ["microsoft", "apple", "meta", "facebook", "openai"]
    },
    "NVDA": {
        "primary": ["nvidia", "nvda", "jensen huang", "gpu", "graphics", "hopper", "blackwell", "chip"],
        "competitors": ["amd", "intel", "tsmc"]
    }
}


# =========================================================================== #
# Public entry point                                                           #
# =========================================================================== #

def prepare_news_items(
    articles: list[dict],
    ticker: str,
    window_hours: int = 24,
    reference_time: Optional[datetime] = None,
) -> list[NewsItem]:
    """
    Full preprocessing pipeline: normalize → deduplicate → filter → hash.

    Parameters
    ----------
    articles : list[dict]
        Raw NewsArticle dicts from Phase 1 NewsFetcher.
    ticker : str
        Target ticker symbol.  Overrides any ticker field already present.
    window_hours : int
        Discard articles older than this many hours before reference_time.
    reference_time : datetime, optional
        Anchor for the staleness filter.  Defaults to now (UTC).

    Returns
    -------
    list[NewsItem]
        Preprocessed, deduplicated, freshness-filtered items, sorted
        newest-first.  Empty list if nothing survives the pipeline.
    """
    ref = reference_time or datetime.now(tz=timezone.utc)
    window_start = ref - timedelta(hours=window_hours)

    if not articles:
        logger.info("preprocessing: no articles received for %s.", ticker)
        return []

    logger.info(
        "preprocessing: starting pipeline for %s — %d raw articles "
        "(window=%dh, ref=%s)",
        ticker, len(articles), window_hours, ref.isoformat(),
    )

    # Step 1 — Coerce to NewsItem, enforce ticker, clean text
    items = _normalize_articles(articles, ticker)
    logger.debug("preprocessing: %d valid items after normalization.", len(items))

    # Step 2 — Remove exact duplicates (URL then article_id)
    items = deduplicate_by_key(items, key="url")
    items = deduplicate_by_key(items, key="article_id")

    # Step 3 — Remove near-duplicate headlines
    items = _deduplicate_by_headline(items, threshold=JACCARD_NEAR_DUP_THRESHOLD)
    logger.debug("preprocessing: %d items after headline dedup.", len(items))

    # Step 4 — Filter stale articles
    items = _filter_window(items, start=window_start, end=ref)
    logger.debug("preprocessing: %d items after window filter.", len(items))

    if not items:
        logger.warning(
            "preprocessing: 0 items remain for %s after full pipeline.", ticker
        )
        return []

    # Step 5 — Build analysis_text, content hash, and relevance score
    items = [_enrich_item(item, ticker) for item in items]

    # Step 6 — Sort newest-first
    items.sort(
        key=lambda x: x.get("published_at") or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )

    logger.info(
        "preprocessing: pipeline complete for %s — %d items ready for inference.",
        ticker, len(items),
    )
    return items


# =========================================================================== #
# Step 1 — Normalize                                                           #
# =========================================================================== #

def _normalize_articles(articles: list[dict], ticker: str) -> list[dict]:
    """
    Convert raw article dicts to cleaned NewsItem dicts.

    Drops articles that fail minimum validation.
    """
    normalized: list[dict] = []
    for raw in articles:
        try:
            item: dict = {}

            # Enforce ticker
            item["ticker"] = ticker

            # Headline — required
            headline = raw.get("headline") or raw.get("title") or ""
            item["headline"] = _clean_text(str(headline))
            if not item["headline"]:
                logger.debug("preprocessing: dropping article with empty headline.")
                continue

            # Snippet — optional; try summary, description, body in that order
            snippet = (
                raw.get("snippet")
                or raw.get("summary")
                or raw.get("description")
                or raw.get("body")
                or ""
            )
            item["snippet"] = _clean_text(str(snippet)) if snippet else None

            # Timestamps — UTC-aware
            pub = raw.get("published_at") or raw.get("published") or raw.get("date")
            item["published_at"] = to_utc(pub)
            if item["published_at"] is None:
                logger.debug(
                    "preprocessing: article missing parseable timestamp: %s",
                    item["headline"][:60],
                )
                # Keep it — cannot filter without a timestamp; Phase 3 can decide
                # to treat None-timestamp items with caution.

            # Optional pass-through fields
            item["source"] = raw.get("source") or raw.get("publisher") or None
            item["url"] = raw.get("url") or raw.get("link") or None
            item["article_id"] = raw.get("article_id") or raw.get("id") or None

            if validate_news_item(item, raise_on_error=False):
                normalized.append(item)

        except Exception as exc:
            logger.debug("preprocessing: error normalizing article — %s", exc)

    return normalized


# =========================================================================== #
# Step 3 — Near-duplicate headline deduplication                              #
# =========================================================================== #

def _deduplicate_by_headline(items: list[dict], threshold: float = 0.85) -> list[dict]:
    """
    Remove near-duplicate items by headline Jaccard similarity.

    Items appearing earlier in the list take precedence.
    """
    kept: list[dict] = []
    kept_headlines: list[str] = []

    for item in items:
        headline = item.get("headline", "")
        is_dup = any(
            jaccard_similarity(headline, h) >= threshold for h in kept_headlines
        )
        if not is_dup:
            kept.append(item)
            kept_headlines.append(headline)

    removed = len(items) - len(kept)
    if removed:
        logger.info(
            "preprocessing: removed %d near-duplicate headlines (threshold=%.2f).",
            removed, threshold,
        )
    return kept


# =========================================================================== #
# Step 4 — Window filter                                                       #
# =========================================================================== #

def _filter_window(
    items: list[dict],
    start: datetime,
    end: datetime,
) -> list[dict]:
    """
    Retain only items whose published_at falls within [start, end].

    Items with None published_at are retained (can't be filtered).
    """
    kept: list[dict] = []
    stale_count = 0

    for item in items:
        pub = item.get("published_at")
        if pub is None:
            kept.append(item)
            continue
        pub_utc = to_utc(pub)
        if pub_utc is None:
            kept.append(item)
            continue
        if start <= pub_utc <= end:
            kept.append(item)
        else:
            stale_count += 1

    if stale_count:
        logger.info(
            "preprocessing: discarded %d articles outside window [%s → %s].",
            stale_count, start.isoformat(), end.isoformat(),
        )
    return kept


# =========================================================================== #
# Step 5 — Enrich: analysis_text + content_hash + relevance_score             #
# =========================================================================== #

def _enrich_item(item: dict, ticker: str) -> dict:
    """
    Add analysis_text, content_hash, and relevance_score to a NewsItem in-place.

    analysis_text = headline + ". " + snippet (truncated to MAX_ANALYSIS_CHARS)
    content_hash  = deterministic SHA-256[:16] of (ticker|headline|snippet|pub_at)
    relevance_score = multiplier based on NER keyword matches
    """
    headline = item.get("headline", "")
    snippet = item.get("snippet") or ""

    if snippet:
        combined = f"{headline}. {snippet}"
    else:
        combined = headline

    item["analysis_text"] = combined[:MAX_ANALYSIS_CHARS]
    item["content_hash"] = compute_content_hash(item)
    item["relevance_score"] = _compute_relevance_score(ticker, headline, item.get("source", ""))
    return item

def _compute_relevance_score(ticker: str, headline: str, source: str) -> float:
    """
    Computes a relevance multiplier (0.3 to 1.0) based on NER and source.
    """
    headline_lower = headline.lower()
    source_lower = str(source).lower()

    # Generic or low-signal source penalty
    is_generic = "google" in source_lower or "rss" in source_lower

    if ticker in COMPANY_KEYWORDS:
        keywords = COMPANY_KEYWORDS[ticker]
        
        # 1. Direct primary mention
        if any(kw in headline_lower for kw in keywords["primary"]):
            return 1.0
        
        # 2. Competitor mention (sector-relevant but not direct)
        if any(kw in headline_lower for kw in keywords["competitors"]):
            return 0.8
    
    # If no specific keyword matches, default to 0.9 for direct APIs, 0.3 for generic
    return 0.3 if is_generic else 0.9

# =========================================================================== #
# Text cleaning helper                                                         #

# =========================================================================== #

def _clean_text(text: str) -> str:
    """
    Normalize unicode, strip HTML remnants, collapse whitespace.

    Does NOT lowercase — preserves original casing for LLM prompts.
    """
    if not text:
        return ""

    # Unicode NFKD normalization
    text = unicodedata.normalize("NFKD", text)

    # Strip crude HTML tags (e.g. from RSS snippets)
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs embedded in text
    text = re.sub(r"https?://\S+", "", text)

    # Collapse repeated punctuation (e.g. "!!!" → "!")
    text = re.sub(r"([!?.]){2,}", r"\1", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
