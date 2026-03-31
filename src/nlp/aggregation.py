"""
nlp/aggregation.py
-------------------
Aggregates article-level sentiment results into a single ticker-level signal.

The aggregation problem
-----------------------
Naive averaging breaks in two common financial news scenarios:

1. Recency bias: a 3-day-old bearish article should not outweigh
   a fresh bullish earnings beat.

2. Syndication flooding: Reuters and AP both publish "TSLA recalls
   vehicles."  The Jaccard deduplication in preprocessing catches
   near-identical copies, but event-level clustering handles the
   case where the same story is written with different phrasing
   across 5–10 articles.

Solution: time-decayed, event-cluster-aware weighted scoring.

Algorithm
---------
1. Assign a recency weight to each article:
     w_time = exp(-λ * hours_since_published)
   where λ (DECAY_LAMBDA) controls how quickly old articles lose weight.
   At λ=0.1: a 24h-old article has ~9% of a fresh article's weight.

2. Group articles into event clusters using headline Jaccard similarity.
   Articles within the same cluster share a cluster weight budget:
     cluster_weight = 1 / cluster_size
   This prevents one event covered by 8 sources from dominating.

3. Each article's effective weight = time_weight × cluster_weight.

4. Sentiment scores are encoded numerically:
     POSITIVE  → +1
     NEUTRAL   →  0
     NEGATIVE  → -1

5. Weighted average sentiment direction is computed.  A +/- threshold
   determines the final label.

6. Conviction score is the weighted average of article conviction_scores.

7. The function returns a TickerSentimentResult dict ready for Phase 3.

Tunable constants
-----------------
DECAY_LAMBDA       : float — recency decay rate (higher = faster decay)
CLUSTER_THRESHOLD  : float — Jaccard similarity for event clustering
SENTIMENT_THRESHOLD: float — |weighted_score| above which we use
                              POSITIVE or NEGATIVE instead of NEUTRAL
MIN_CONVICTION     : float — floor on conviction for NEUTRAL signals

All constants can be overridden at call time for experiments.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Optional

from src.data.utils import jaccard_similarity
from src.nlp.schemas import (
    TickerSentimentResult,
    validate_ticker_result,
    make_empty_ticker_result,
    VALID_SENTIMENT_LABELS,
    clamp_conviction,
)

logger = logging.getLogger(__name__)

# =========================================================================== #
# Tunable constants                                                            #
# =========================================================================== #

DECAY_LAMBDA: float = 0.08        # hours; at 0.08, 12h-old item ≈ 38% weight
CLUSTER_THRESHOLD: float = 0.60   # Jaccard threshold for event clustering
SENTIMENT_THRESHOLD: float = 0.15 # weighted score above which label flips
MIN_CONVICTION: float = 2.0       # floor when signal is too weak to classify


# =========================================================================== #
# Public entry point                                                           #
# =========================================================================== #

def aggregate_to_ticker(
    article_results: list[dict],
    ticker: str,
    window_hours: int = 24,
    reference_time: Optional[datetime] = None,
    provider_used: Optional[str] = None,
    decay_lambda: float = DECAY_LAMBDA,
    cluster_threshold: float = CLUSTER_THRESHOLD,
    sentiment_threshold: float = SENTIMENT_THRESHOLD,
) -> dict:
    """
    Aggregate a list of ArticleSentimentResult dicts into a
    single TickerSentimentResult.

    Parameters
    ----------
    article_results : list[dict]
        Output from any sentiment provider (or mixed providers).
    ticker : str
        Target ticker symbol.
    window_hours : int
        Analysis window size used for metadata and context.
    reference_time : datetime, optional
        Anchor for recency decay calculation.  Defaults to now (UTC).
    provider_used : str, optional
        Name of the provider that generated the results.
    decay_lambda : float
        Controls how quickly older articles lose weight.
    cluster_threshold : float
        Jaccard similarity above which two headlines are treated as
        one event cluster.
    sentiment_threshold : float
        Minimum absolute weighted score to assign POSITIVE/NEGATIVE
        instead of NEUTRAL.

    Returns
    -------
    dict (TickerSentimentResult)
    """
    ref = reference_time or datetime.now(tz=timezone.utc)

    if not article_results:
        logger.info(
            "aggregation: no article results for %s — returning empty signal.", ticker
        )
        result = make_empty_ticker_result(ticker, window_hours)
        result["provider_used"] = provider_used
        result["article_results"] = []
        return result

    logger.info(
        "aggregation: aggregating %d article results for %s.", len(article_results), ticker
    )

    # Step 1 — Filter out invalid results
    valid_results = [r for r in article_results if _is_valid_result(r)]
    if not valid_results:
        logger.warning(
            "aggregation: all %d results for %s failed validation. "
            "Returning empty signal.", len(article_results), ticker
        )
        result = make_empty_ticker_result(ticker, window_hours)
        result["provider_used"] = provider_used
        result["article_results"] = []
        return result

    # Step 2 — Assign recency weights
    weighted = _assign_recency_weights(valid_results, ref, decay_lambda)

    # Step 3 — Assign cluster weights (reduce syndication flooding)
    weighted = _assign_cluster_weights(weighted, cluster_threshold)

    # Step 4 — Compute effective weights and aggregate scores
    total_weight = sum(w["_effective_weight"] for w in weighted)
    if total_weight == 0:
        logger.warning("aggregation: total weight is zero for %s.", ticker)
        result = make_empty_ticker_result(ticker, window_hours)
        result["provider_used"] = provider_used
        return result

    weighted_direction = 0.0
    weighted_conviction = 0.0

    for item in weighted:
        eff_w = item["_effective_weight"]
        direction = _label_to_direction(item.get("sentiment", "NEUTRAL"))
        conviction = clamp_conviction(float(item.get("conviction_score", 0)))

        weighted_direction += eff_w * direction
        weighted_conviction += eff_w * conviction

    avg_direction = weighted_direction / total_weight
    avg_conviction = weighted_conviction / total_weight

    # Step 5 — Map to label
    final_label = _direction_to_label(avg_direction, sentiment_threshold)

    # Step 6 — Apply conviction floor for weak/neutral signals
    final_conviction = round(avg_conviction, 2)
    if final_label == "NEUTRAL":
        final_conviction = min(final_conviction, MIN_CONVICTION)

    # Step 7 — Compose output
    unique_event_count = _count_event_clusters(weighted, cluster_threshold)

    reasoning = _compose_reasoning(
        valid_results, final_label, avg_direction, avg_conviction, unique_event_count
    )

    result: dict = {
        "ticker": ticker,
        "sentiment": final_label,
        "conviction_score": final_conviction,
        "reasoning": reasoning,
        "source_count": len(valid_results),
        "unique_event_count": unique_event_count,
        "analysis_window_hours": window_hours,
        "generated_at": ref,
        "provider_used": provider_used,
        "article_results": article_results,
    }

    logger.info(
        "aggregation: %s → sentiment=%s, conviction=%.1f "
        "(sources=%d, events=%d, avg_direction=%.3f)",
        ticker, final_label, final_conviction,
        len(valid_results), unique_event_count, avg_direction,
    )

    return result


# =========================================================================== #
# Step 2 — Recency weighting                                                  #
# =========================================================================== #

def _assign_recency_weights(
    results: list[dict],
    ref: datetime,
    decay_lambda: float,
) -> list[dict]:
    """
    Attach a '_time_weight' field to each result.

    weight = exp(-λ * hours_since_published)
    If published_at is None, assign weight 0.5 (uncertain freshness).
    """
    enriched = []
    for r in results:
        r = dict(r)
        pub = r.get("published_at") or r.get("inferred_at")
        if pub is None:
            r["_time_weight"] = 0.5
        else:
            if hasattr(pub, "tzinfo") and pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            try:
                hours_old = max((ref - pub).total_seconds() / 3600.0, 0.0)
            except Exception:
                hours_old = 12.0  # safe default
            r["_time_weight"] = math.exp(-decay_lambda * hours_old)
        enriched.append(r)
    return enriched


# =========================================================================== #
# Step 3 — Event cluster weighting                                            #
# =========================================================================== #

def _assign_cluster_weights(results: list[dict], threshold: float) -> list[dict]:
    """
    Group results into event clusters by headline Jaccard similarity.
    Assign '_cluster_weight' = 1 / cluster_size so no event dominates.
    """
    n = len(results)
    cluster_ids = list(range(n))

    # Union-find to assign cluster IDs
    headlines = [r.get("headline", "") for r in results]
    for i in range(n):
        for j in range(i + 1, n):
            if jaccard_similarity(headlines[i], headlines[j]) >= threshold:
                # Merge clusters — set j's cluster ID to i's
                old_id = cluster_ids[j]
                new_id = cluster_ids[i]
                cluster_ids = [new_id if c == old_id else c for c in cluster_ids]

    # Count cluster sizes
    from collections import Counter
    cluster_sizes = Counter(cluster_ids)

    enriched = []
    for idx, r in enumerate(results):
        r = dict(r)
        r["_cluster_id"] = cluster_ids[idx]
        r["_cluster_weight"] = 1.0 / cluster_sizes[cluster_ids[idx]]
        enriched.append(r)

    return enriched


def _attach_effective_weights(results: list[dict]) -> list[dict]:
    """Compute effective weight = time_weight × cluster_weight × relevance_score."""
    for r in results:
        r["_effective_weight"] = (
            r.get("_time_weight", 1.0) *
            r.get("_cluster_weight", 1.0) *
            r.get("relevance_score", 1.0)
        )
    return results


# Attach effective weights inside aggregate_to_ticker — call here:
def _assign_recency_weights(  # noqa: F811 — intentional redefinition with augmentation
    results: list[dict],
    ref: datetime,
    decay_lambda: float,
) -> list[dict]:
    enriched = []
    for r in results:
        r = dict(r)
        pub = r.get("published_at") or r.get("inferred_at")
        if pub is None:
            r["_time_weight"] = 0.5
        else:
            if hasattr(pub, "tzinfo") and pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            try:
                hours_old = max((ref - pub).total_seconds() / 3600.0, 0.0)
            except Exception:
                hours_old = 12.0
            r["_time_weight"] = math.exp(-decay_lambda * hours_old)
        enriched.append(r)
    return enriched


def _assign_cluster_weights(  # noqa: F811 — re-definition; adds effective weight step
    results: list[dict], threshold: float
) -> list[dict]:
    from collections import Counter
    n = len(results)
    cluster_ids = list(range(n))
    headlines = [r.get("headline", "") for r in results]

    for i in range(n):
        for j in range(i + 1, n):
            if jaccard_similarity(headlines[i], headlines[j]) >= threshold:
                old_id = cluster_ids[j]
                new_id = cluster_ids[i]
                cluster_ids = [new_id if c == old_id else c for c in cluster_ids]

    cluster_sizes = Counter(cluster_ids)
    enriched = []
    for idx, r in enumerate(results):
        r = dict(r)
        r["_cluster_id"] = cluster_ids[idx]
        r["_cluster_weight"] = 1.0 / cluster_sizes[cluster_ids[idx]]
        r["_effective_weight"] = (
            r["_time_weight"] *
            r["_cluster_weight"] *
            r.get("relevance_score", 1.0)
        )
        enriched.append(r)
    return enriched


# =========================================================================== #
# Scoring helpers                                                              #
# =========================================================================== #

def _label_to_direction(label: str) -> float:
    """Map sentiment label to numeric direction: +1, 0, -1."""
    mapping = {"POSITIVE": 1.0, "NEUTRAL": 0.0, "NEGATIVE": -1.0}
    return mapping.get(label.upper(), 0.0)


def _direction_to_label(direction: float, threshold: float) -> str:
    """Convert weighted direction to a sentiment label."""
    if direction > threshold:
        return "POSITIVE"
    elif direction < -threshold:
        return "NEGATIVE"
    else:
        return "NEUTRAL"


def _count_event_clusters(results: list[dict], threshold: float) -> int:
    """Count the number of distinct event clusters in the results."""
    cluster_ids = set(r.get("_cluster_id", i) for i, r in enumerate(results))
    return len(cluster_ids)


# =========================================================================== #
# Validation helper                                                            #
# =========================================================================== #

def _is_valid_result(result: dict) -> bool:
    """Accept a result if it has the minimum required fields."""
    sentiment = result.get("sentiment", "")
    if sentiment not in VALID_SENTIMENT_LABELS:
        return False
    score = result.get("conviction_score")
    if score is None:
        return False
    try:
        float(score)
    except (TypeError, ValueError):
        return False
    return True


# =========================================================================== #
# Reasoning composer                                                           #
# =========================================================================== #

def _compose_reasoning(
    results: list[dict],
    final_label: str,
    avg_direction: float,
    avg_conviction: float,
    unique_events: int,
) -> str:
    """
    Compose a concise human-readable reasoning string for the ticker result.

    Draws on the most confident article's reasoning text and summarizes
    the aggregate picture.
    """
    n = len(results)
    top = max(results, key=lambda r: float(r.get("conviction_score", 0)), default=None)
    top_reasoning = top.get("reasoning", "") if top else ""

    sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    for r in results:
        lbl = r.get("sentiment", "NEUTRAL").upper()
        sentiment_counts[lbl] = sentiment_counts.get(lbl, 0) + 1

    summary = (
        f"{n} articles analyzed ({unique_events} distinct events). "
        f"Sentiment breakdown: {sentiment_counts['POSITIVE']} positive, "
        f"{sentiment_counts['NEGATIVE']} negative, {sentiment_counts['NEUTRAL']} neutral. "
        f"Weighted direction: {avg_direction:+.3f}. "
        f"Most confident article: \"{top_reasoning[:120]}\"" if top_reasoning else ""
    )

    return summary.strip()
