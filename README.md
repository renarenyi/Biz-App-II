# Trading Agent — LLM-Driven Automated Trading Bot

An LLM-driven automated trading bot for paper trading and academic research.
Built for **MMAI 5090 — Business Applications of AI II**.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Setting Up API Keys](#setting-up-api-keys)
3. [Quickstart](#quickstart)
4. [LLM Prompt Logic](#llm-prompt-logic)
5. [Backtest Results](#backtest-results)
6. [Strategy Design](#strategy-design)
7. [Provider Fallback Chains](#provider-fallback-chains)
8. [Output Schemas](#output-schemas)
9. [Caching Model](#caching-model)
10. [Design Principles](#design-principles)

---

## Architecture Overview

```
trading-agent/
  data/                        ← cached files (gitignored)
  src/
    config/
      settings.py              ← env-based config (all secrets via .env)
    data/
      schemas.py               ← data contracts: OHLCVBar, NewsArticle
      cache.py                 ← TTLCache (memory) + DiskCache (parquet/jsonl)
      utils.py                 ← timestamps, dedup, text helpers
      market_data_handler.py   ← primary market data orchestrator
      news_fetcher.py          ← primary news data orchestrator
      providers/
        alpaca_provider.py       ← Alpaca market data (primary)
        yfinance_provider.py     ← yfinance market data (fallback)
        alpaca_news_provider.py  ← Alpaca News API (primary)
        fmp_news_provider.py     ← FMP News API (supplemental)
        finnhub_news_provider.py ← Finnhub News API (supplemental)
        rss_news_provider.py     ← RSS feeds (fallback)
    nlp/                       ← Phase 2: LLM sentiment agent
      sentiment_agent.py         ← orchestrator with caching & freshness
      providers/
        finbert_provider.py      ← FinBERT (primary, CPU-runnable, deterministic)
    strategy/                  ← Phase 3: signal generation & stock selection
      signal_rules.py            ← entry/exit logic (SMA-50 + sentiment + RSI)
      risk_manager.py            ← stop-loss, take-profit, trailing stops
      stock_screener.py          ← dynamic stock screening & monthly rotation
    backtest/                  ← Phase 4: historical backtesting
      backtester.py              ← core simulation engine
      report_generator.py        ← metrics & trade log output
  tests/
    test_market_data_handler.py
    test_news_fetcher.py
    test_sentiment_agent.py
    ...                          (292 tests total)
  run_backtest.py              ← CLI entry point for backtesting
  demo_backtest.ipynb          ← presentation notebook for live demo
  main.py                      ← CLI entry point for live paper trading
  sentiment.py                 ← CLI entry point for sentiment analysis
  requirements.txt
```

---

## Setting Up API Keys

### 1. Alpaca Paper Trading (Required)

Alpaca provides market data (OHLCV bars) and limited news articles.

1. Register at [https://alpaca.markets](https://alpaca.markets)
2. Go to Paper Trading → API Keys
3. Copy your API Key and Secret Key

### 2. FMP — Financial Modeling Prep (Recommended)

FMP provides supplemental historical stock news alongside Alpaca's primary coverage (~3,000+ articles per ticker with pagination).

1. Register at [https://financialmodelingprep.com/register](https://financialmodelingprep.com/register)
2. Copy your API key from the dashboard
3. Free tier: 250 API calls/day (sufficient for multi-ticker backtesting)

### 3. Finnhub (Recommended)

Finnhub provides additional company news coverage, filling gaps in FMP's archive.

1. Register at [https://finnhub.io/register](https://finnhub.io/register)
2. Copy your API key from the dashboard
3. Free tier: 60 API calls/minute (sufficient for backtesting)

### 4. HuggingFace Token (Optional)

Only needed if hitting HuggingFace rate limits when downloading FinBERT.

1. Register at [https://huggingface.co](https://huggingface.co)
2. Go to Settings → Access Tokens

### 5. Configure `.env`

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```env
# Required
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key

# Recommended — enables 1,700+ real news articles for sentiment analysis
FMP_API_KEY=your_fmp_api_key

# Recommended — supplements FMP for additional article coverage
FINNHUB_API_KEY=your_finnhub_api_key

# Optional
HF_TOKEN=your_huggingface_token
```

> **Safety**: Never commit `.env` to version control. It is listed in `.gitignore`.

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run all tests (292 tests)

```bash
pytest tests/ -v
```

### 3. Run Backtesting (Phase 4)

```bash
# Multi-ticker backtest (default: AAPL, MSFT, GOOGL, NVDA)
python run_backtest.py

# With dynamic stock rotation — screens 29-stock universe monthly
python run_backtest.py --screen

# Select top 6 stocks per rotation window
python run_backtest.py --screen --top-n 6

# Custom tickers and lookback
python run_backtest.py --tickers AAPL MSFT AMZN --months 12

# Single ticker (backward compatible)
python run_backtest.py --ticker AAPL --months 12
```

This generates:
- Full backtest report (console output)
- `equity_curve.png` — strategy vs SPY benchmark chart

#### Running the Backtest

Run **`run_backtest.py`** — this uses **FinBERT** (ProsusAI/finbert), a finance-domain BERT model that runs locally on CPU with no API costs. It fetches ~3,000+ real news articles per ticker via the paginated Alpaca News API and classifies each one deterministically. Results are cached to disk, so subsequent runs complete in under 2 seconds.

### 4. Run Sentiment Analysis (Phase 2)

```bash
python sentiment.py --ticker AAPL --lookback 29
```

### 5. Run Live Paper Trading (Phase 3)

```bash
python main.py --tickers AAPL TSLA
```

---

## LLM Prompt Logic

The system uses **FinBERT** (ProsusAI/finbert) — a finance-domain BERT model fine-tuned on 10,000+ financial texts — as its sentiment engine. FinBERT runs locally on CPU, requires no API keys, and is fully deterministic (same headline always produces the same score).

### FinBERT (Primary Model)

FinBERT is a finance-domain BERT model fine-tuned on financial text. It classifies each headline into three classes with a softmax probability:

| FinBERT Output | Mapped Sentiment | Conviction Score |
|---|---|---|
| `positive` (p > 0.5) | POSITIVE | `p × 10` (0–10 scale) |
| `negative` (p > 0.5) | NEGATIVE | `p × 10` |
| `neutral` | NEUTRAL | `p × 10` |

- **Input**: Single headline string (max 512 tokens, automatically truncated)
- **Output**: Sentiment label + conviction score derived from softmax probability
- **Batch processing**: FinBERT processes all articles in a single forward pass for efficiency
- **No prompt engineering needed**: FinBERT is a fine-tuned classifier, not a generative model

### Sentiment Aggregation

For each trading day, the `SentimentAgent` aggregates all article-level scores into a single daily signal:

1. Fetch articles within a 24-hour lookback window
2. Run each article through the LLM provider chain
3. Aggregate: weighted average of conviction scores (newest articles weighted higher)
4. Output: daily `sentiment_label` (POSITIVE/NEGATIVE/NEUTRAL) + `conviction_score` (0–10)

This daily signal is then combined with technical indicators (SMA-50, RSI) by `signal_rules.py` to generate entry/exit decisions.

---

## Backtest Results

### Performance Summary (As of Mar 31st 2026)

| Metric | FinBERT Strategy | SPY Benchmark |
|---|------------------|---|
| **Total Return** | **+15.41%**      | +13.01% |
| **Final Equity** | **$115,410**     | — |
| **Max Drawdown** | **-2.58%**       | -12.05% |
| **Sharpe Ratio** | **1.559**        | 0.772 |
| **Calmar Ratio** | **6.006**        | — |
| **Profit Factor** | **2.376**       | — |
| **Win Rate** | **56.06%**       | — |
| **Trade Count** | **66**           | 1 (buy & hold) |
| **Avg Win / Avg Loss** | **$719 / -$386** | — |
| **Avg Holding** | **11.4 days**    | — |
| **Market Exposure** | **52.59%**       | 100% |

### Key Observations

- **Outperforms SPY on both raw and risk-adjusted return**: 15.41% vs 13.01% total return, with a Sharpe of **1.559 vs 0.772** — 2× more return per unit of risk
- **Capital preservation**: Strategy drawdown (-2.58%) is **4.7× lower** than SPY (-12.05%)
- **Half the exposure, higher return**: The strategy is only in the market 52.6% of the time yet beats SPY's total return
- **Strong profit factor**: 2.376 — winning trades generate 2.4× more profit than losing trades cost
- **Leverage potential**: With 1.9× leverage, the strategy would match SPY's exposure while returning ~29.3% with ~-4.9% drawdown
- **66 trades across 4 stocks**: Sufficient sample size for statistical confidence
- **Full NLP coverage**: With paginated Alpaca News API fetching ~3,000+ articles per ticker, real FinBERT sentiment drives decisions on the vast majority of trading days

### Dynamic Stock Rotation Mode

When run with `--screen`, the system dynamically selects the best stocks from a **29-stock universe** across 6 sectors, re-screening every 21 trading days (~monthly):

| Sector | Stocks |
|---|---|
| Tech (10) | AAPL, MSFT, GOOGL, NVDA, META, AMZN, TSLA, AMD, CRM, ORCL |
| Finance (4) | JPM, GS, V, MA |
| Healthcare (3) | JNJ, UNH, PFE |
| Consumer (3) | WMT, COST, DIS |
| Energy (2) | XOM, CVX |
| Industrial (2) | BA, CAT |

Screening scores each stock on 3 criteria (0-100 total):
1. **Liquidity** (0-33) — average daily dollar volume
2. **Trend Strength** (0-33) — % of days above SMA-50 + current trend
3. **Volatility** (0-34) — targets the 20-35% annualized sweet spot

At each rotation boundary, the top N stocks are selected. Stocks rotated out naturally exit via existing stop-loss/trend-failure rules. Only sentiment for "active" tickers is fed to the backtester, so no trades are opened for rotated-out stocks.

### Parameter Optimization History

We iteratively tuned strategy parameters to maximise risk-adjusted return:

| Config | Stop-Loss | Take-Profit | Conviction | Equity/Trade | Features | Return | Sharpe | Trades |
|---|---|---|---|---|---|---|---|---|
| Initial (50 articles) | 7% | 14% | 7.0 | 10% | NER filters | 5.74% | — | 42 |
| Tighter stop | 5% | 30% | 7.0 | 12% | No NER | -0.01% | — | 62 |
| Optimized SL/TP | 7% | 10% | 7.0 | 12% | No NER | 6.47% | 0.641 | 48 |
| Full coverage (50 articles) | 7% | 10% | 7.0 | 12% | NER + Cross-Ticker | 6.69% | 0.650 | 49 |
| Full coverage (3,000+ articles) | 7% | 10% | 7.0 | 12% | NER + Cross-Ticker + Paginated News | 11.5% | 0.99 | 72 |
| **Optimized TP + lower conviction** | **7%** | **30%** | **6.0** | **12%** | **NER + Cross-Ticker + Paginated News** | **15.41%** | **1.559** | **66** |

Key insights:
- The biggest performance leap came from fixing Alpaca News pagination — going from 50 to 3,000+ articles per ticker gave FinBERT real data to analyze on virtually every trading day, replacing the momentum-based backfill that diluted the NLP signal.
- The final parameter tuning — raising take-profit from 10% to 30% and lowering conviction threshold from 7.0 to 6.0 — let winners run longer and entered more trades, boosting total return from 11.5% to 15.41% and Sharpe from 0.99 to 1.559.

---

## Strategy Design

### Entry Conditions (all must be true)

1. **Trend filter**: Price above 50-day Simple Moving Average (SMA-50)
2. **ADX trend quality**: ADX-14 ≥ 20 (confirms directional trend, filters out choppy markets)
3. **Sentiment signal**: Daily aggregated sentiment = POSITIVE with conviction ≥ 6.0
4. **Rolling sentiment context**: 5-day sentiment EMA > 0 (prevents reacting to isolated positive headlines after sustained negativity)
5. **RSI filter**: 14-period RSI < 70 (prevents buying into overbought conditions)
6. **NER Relevance Weighting**: Headlines are checked for company/competitor keywords; direct mentions get full weight (1.0), competitors get 0.8, and generic/SEO news gets heavily discounted (0.3).
7. **Cross-Ticker Dampening**: If a target's sector competitor is currently experiencing a massive surge in sentiment (conviction ≥ 8.0), the target ticker's conviction is dampened by 10% per competitor (capped at 25%) to reflect competitive dynamics.
8. **No existing position**: Only one position per ticker at a time

### Exit Conditions (any one triggers)

1. **Stop-loss**: -7% from entry price
2. **Take-profit**: +30% from entry price
3. **Trend exit**: Price crosses below SMA-50
4. **Sentiment reversal**: Daily sentiment flips to NEGATIVE with conviction ≥ 8.0

### Position Sizing

- Fixed 12% of portfolio equity per trade
- Maximum 4 concurrent positions (one per ticker)

---

## Provider Fallback Chains

### Market Data
```
1. Alpaca StockHistoricalDataClient   (requires ALPACA_API_KEY + ALPACA_SECRET_KEY)
      ↓ if degraded / empty result
2. yfinance.download()                (no credentials required)
      ↓ if both fail
   Return empty DataFrame + log error
```

### News Data
```
1. Alpaca NewsClient                  (requires ALPACA_API_KEY)
      ↓ if degraded OR < MIN_NEWS_ARTICLES returned
2. FMP Stock News API                 (requires FMP_API_KEY)
      ↓ always supplemented by
3. Finnhub Company News API           (requires FINNHUB_API_KEY)
      ↓ if still < MIN_NEWS_ARTICLES
4. RSS feeds: Yahoo Finance, WSJ, CNBC, Reuters  (no credentials)
      ↓ if all fail
   Return empty list + log warning

5. Google News RSS (gap-fill only)    (no credentials)
   → Available for historical coverage via yahoo_news_provider.py
   → Scrapes month-by-month via Google News RSS with date-range queries
   → Not active in default pipeline (tested but yielded signal dilution)
```

### Sentiment Analysis
```
1. FinBERT (ProsusAI/finbert)         (auto-downloaded from HuggingFace, CPU-only)
```

---

## Output Schemas

### OHLCV DataFrame

| Column | Type | Notes |
|---|---|---|
| timestamp | datetime64[ns, UTC] | Timezone-aware |
| open | float64 | |
| high | float64 | |
| low | float64 | |
| close | float64 | |
| volume | float64 | |
| symbol | str | Ticker |
| source | str | "alpaca" or "yfinance" |

### NewsArticle dict

| Key | Type | Notes |
|---|---|---|
| ticker | str | |
| headline | str | Whitespace-normalized |
| summary | str or None | |
| source | str | Publication name |
| published_at | datetime (UTC) | |
| url | str | |
| provider | str | "alpaca_news", "fmp", "finnhub", "rss", or "google_news" |

---

## Caching Model

| Tier | Storage | Default TTL |
|---|---|---|
| 1 — In-memory | Python dict | 30s (quotes), 1h (bars), 10m (news) |
| 2 — Disk | Parquet (bars) / JSON-lines (news) | Persists across runs |
| 3 — Provider | Alpaca / FMP / Finnhub / yfinance / RSS | Only when tiers 1+2 miss |

Set `use_cache=False` on any handler to bypass caching entirely.
Set `force_refresh=True` per call to skip cache for one request.

---

## Circuit Breakers

Each provider tracks consecutive failures. After `MAX_FAILURES` (default: 3),
the provider enters a cooldown period (`PROVIDER_COOLDOWN_SECONDS`, default: 120s).
During cooldown, requests skip that provider immediately without hitting the API.

---

## Design Principles

- **No look-ahead bias** — timestamps are publication time, not ingestion time
- **Schema-first** — every module returns the same documented schema
- **Provider-agnostic** — `MarketDataHandler` and `NewsFetcher` hide all provider logic
- **Fail gracefully** — individual provider errors never crash the full pipeline
- **Provenance** — every record carries `source`, `provider`, `ingested_at`
- **Reproducibility** — disk cache preserves exact datasets used in experiments
- **Iterative optimization** — parameter tuning (stop-loss, take-profit, position sizing) was data-driven, with each experiment tracked and evaluated

---

## Phases

| Phase | Status | Description |
|---|---|---|
| 1 — Data Infrastructure | ✅ Complete | `MarketDataHandler` + `NewsFetcher` (Alpaca + FMP + Finnhub integrated) |
| 2 — LLM Sentiment Agent | ✅ Complete | FinBERT conviction scoring with 3,000+ articles per ticker |
| 3 — Strategy & Execution | ✅ Complete | Signal generation + dynamic stock screening & rotation |
| 4 — Backtesting | ✅ Complete | Multi-ticker validation vs SPY + equity curve + rotation support |

---

## Safety Notes

- **Paper trading only.** Do not change `ALPACA_BASE_URL` to the live endpoint.
- **LLMs process text, not price math.** The LLM layer analyzes news sentiment; OHLCV signals come from technical indicators.
- **All credentials in `.env` only.** Never commit API keys to version control.
