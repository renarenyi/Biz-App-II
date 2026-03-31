"""
sentiment.py
------------
This executable script satisfies the project requirement for "The module containing 
the LLM prompt engineering and sentiment logic".

It acts as a wrapper around the core Phase 2 `SentimentAgent` which orchestrates
the sentiment scoring using a FallbackRouter (FinBERT -> Llama 3 -> Mistral).
For the LLM-specific instances (Llama 8B / Mistral), the prompt engineering is implemented
in `src/nlp/providers/llama_provider.py`. The fundamental prompt instructs the model as follows:

-----------------------------------------------------------------------------------
SYSTEM PROMPT:
You are a financial news sentiment classifier.
Your task is to analyze a news headline or snippet about a public company.
Classify the item as POSITIVE, NEGATIVE, or NEUTRAL based on the likely
business impact implied by the text.
Assign a conviction_score from 0 to 10 based on how strongly the news
implies beneficial or adverse business impact (0 = very weak signal, 10 = very strong).
Provide a short reasoning grounded only in the text provided.
Optionally classify the event_type from this list:
  earnings_beat, earnings_miss, guidance_raise, guidance_cut, recall,
  lawsuit, regulation, merger_acquisition, product_launch,
  executive_change, macro, unknown.
Return ONLY valid JSON. No prose. No markdown fences.
Output schema:
{{
  "sentiment": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
  "conviction_score": <integer 0-10>,
  "reasoning": "<brief text explanation>",
  "event_type": "<event label or unknown>"
}}

USER PROMPT:
Analyze the following financial news item.
Ticker: {ticker}
Headline: {headline}
Snippet: {snippet}
Return JSON only.
-----------------------------------------------------------------------------------

Usage:
    python sentiment.py --ticker TSLA --lookback 24
"""

import argparse
import json
import logging
from datetime import datetime

from src.data.news_fetcher import NewsFetcher
from src.nlp.sentiment_agent import SentimentAgent
from src.config.settings import settings

# Configure logging to console
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run the LLM Sentiment Agent on recent news.")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol (e.g. AAPL)")
    parser.add_argument("--lookback", type=int, default=24, help="Hours of news to analyze (default: 24)")
    args = parser.parse_args()

    # Step 1: Fetch recent news for the ticker
    logger.info(f"Fetching news for {args.ticker} over the last {args.lookback} hours...")
    fetcher = NewsFetcher(use_cache=False)
    articles = fetcher.get_recent_news(args.ticker, lookback_hours=args.lookback)

    if not articles:
        logger.warning(f"No news found for {args.ticker} in the last {args.lookback} hours.")
        return

    logger.info(f"Retrieved {len(articles)} articles. Initiating Sentiment Agent...")

    # Step 2: Initialize Sentiment Agent
    # For speed and local execution, the default router relies on FinBERT -> Llama3 fallback
    agent = SentimentAgent(window_hours=args.lookback, use_disk_cache=False)

    # Step 3: Analyze articles
    result = agent.analyze(ticker=args.ticker, articles=articles)

    # Step 4: Display Output
    print("\n" + "="*60)
    print(f" LLM SENTIMENT ANALYSIS RESULT FOR {args.ticker}")
    print("="*60)
    print(json.dumps({
        "Ticker": result["ticker"],
        "Sentiment (Aggregated)": result["sentiment"],
        "Conviction Score (0-10)": result["conviction_score"],
        "Sources Analyzed": result["source_count"],
        "Unique Events Detected": result["unique_event_count"],
        "Analysis Window": f"{result['analysis_window_hours']} hours",
        "Primary Model Used": result.get("provider_used", "finbert / fallback"),
    }, indent=4))
    
    print("\nSAMPLE ARTICLE REASONINGS:")
    article_results = result.get("article_results", [])
    for idx, art in enumerate(article_results[:3]):
        print(f"  [{idx+1}] Headline: {art['headline']}")
        print(f"      Sentiment: {art['sentiment']} (Score: {art['conviction_score']})")
        print(f"      Reasoning: {art['reasoning']}")
        print("-" * 60)
        
    print("\nProcessing complete.")


if __name__ == "__main__":
    main()
