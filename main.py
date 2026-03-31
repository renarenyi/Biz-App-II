"""
main.py
-------
Executable script to run the LLM-Driven Automated Trading Agent (Phase 1-3).
This script fulfills the final requirement for an executable that runs the bot.

It integrates:
  - Phase 1: MarketDataHandler (Price/Technicals) and NewsFetcher (Headlines) via Alpaca
  - Phase 2: SentimentAgent (LLM Sentiment & Conviction Scoring)
  - Phase 3: StrategyEngine & ExecutionEngine (Signal Rules and Paper Order Execution)

Usage:
    python main.py
"""

import logging
import argparse
from datetime import datetime, timedelta, timezone

from src.data.market_data_handler import MarketDataHandler
from src.data.news_fetcher import NewsFetcher
from src.nlp.sentiment_agent import SentimentAgent
from src.strategy.strategy_engine import StrategyEngine
from src.strategy.execution_engine import ExecutionEngine
from src.config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Define a diversified 10-stock portfolio across multiple sectors
DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "V", "JNJ", "UNH", "AMZN", "WMT", "XOM"]

def main():
    parser = argparse.ArgumentParser(description="Run the Automated Trading Bot.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS, help="List of tickers to evaluate")
    args = parser.parse_args()

    tickers = args.tickers
    logger.info(f"Starting LLM Trading Agent run for {len(tickers)} tickers: {tickers}")

    # =========================================================================
    # 1. Initialize Infrastructure
    # =========================================================================
    logger.info("Initializing Data Handlers...")
    market_handler = MarketDataHandler()
    news_fetcher   = NewsFetcher(use_cache=False)

    logger.info("Initializing LLM Sentiment Agent...")
    sentiment_agent = SentimentAgent(window_hours=24, use_disk_cache=False)

    logger.info("Initializing Strategy & Execution Engine...")
    # ExecutionEngine.from_settings() automatically uses paper trading if Alpaca keys are present in .env
    execution_engine = ExecutionEngine.from_settings()
    
    strategy = StrategyEngine(
        execution_engine=execution_engine,
        conviction_threshold=7.5,  # Increased strictness: only trade highly confident news
        sentiment_max_age_hours=24,
        stop_loss_pct=0.045,       # Widened to 4.5% to avoid 'noise' stop-outs
        equity_fraction=0.05,      # 5% of portfolio per trade
    )

    now = datetime.now(timezone.utc)
    lookback_start = now - timedelta(days=100)  # For 50-day SMA calculation

    # =========================================================================
    # 2. Daily Run Loop
    # =========================================================================
    for ticker in tickers:
        logger.info(f"\n{'='*50}\nEvaluating Ticker: {ticker}\n{'='*50}")

        try:
            # --- Gather Market snapshot ---
            logger.info("Fetching market data...")
            bars = market_handler.get_historical_bars(ticker, start=lookback_start, end=now, timeframe="1Day")
            if bars.empty:
                logger.warning(f"No historical bars for {ticker}. Skipping.")
                continue

            # Add technicals
            bars = market_handler.add_moving_averages(bars, windows=[50])
            latest_bar = bars.iloc[-1]

            # Get latest price quote
            quote = market_handler.get_latest_price(ticker)
            last_price = quote.get("last_price") or latest_bar["close"]

            market_snap = {
                "ticker": ticker,
                "timestamp": now,
                "close": last_price,
                "sma_50": latest_bar.get("sma_50"),
                "is_market_open": True,  # Assumed True for simulation. Alpaca paper execution handles real hours.
            }

            # --- Gather Sentiment snapshot ---
            logger.info("Fetching recent news & running LLM Sentiment inference...")
            articles = news_fetcher.get_recent_news(ticker, lookback_hours=24)
            sentiment_result = sentiment_agent.analyze(ticker, articles)

            sentiment_snap = {
                "ticker": ticker,
                "sentiment": sentiment_result.get("sentiment", "NEUTRAL"),
                "conviction_score": sentiment_result.get("conviction_score", 0.0),
                "generated_at": now,
            }

            # --- Evaluate Signal & Execute ---
            logger.info("Evaluating Strategy Rules...")
            result = strategy.evaluate(
                ticker=ticker,
                market=market_snap,
                sentiment=sentiment_snap,
                reference_time=now
            )

            # --- Output Decision ---
            signal = result.get("signal", "NO_ACTION")
            reason = result.get("reason", "No reason provided")
            logger.info(f"ACTION FOR {ticker}: {signal}")
            logger.info(f"REASON: {reason}")
            
            if result.get("order"):
                logger.info(f"ORDER SUBMITTED -> ID: {result['order'].get('order_id')}, Status: {result['order'].get('status')}")

        except Exception as e:
            logger.error(f"Failed to process {ticker}: {str(e)}", exc_info=True)

    logger.info("\nRun completed successfully. Review logs for paper-trading artifacts.")


if __name__ == "__main__":
    main()
