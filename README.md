# Stock-Signals

**Production-grade trading signal system for Brazilian equities (IBOV + SMLL)**

A multi-indicator, value investing fusion system that analyzes 214 stocks, combining technical indicators, volume analysis, news sentiment, and fundamental analysis to generate actionable trading recommendations with automatic Google Sheets synchronization.

---

## Features

### Real-Time Price Data (BrAPI)
- **BrAPI integration** (brapi.dev) - Near real-time Brazilian stock prices
- **No rate limiting** - Fetches 200+ quotes in seconds (vs yfinance timeouts)
- **Batch processing** - Reliable 5-ticker batches with retry logic
- **Historical data** - 3 months of daily candles for technical analysis

### Multi-Indicator Technical Analysis
- **12 technical indicators** across 4 categories:
  - Momentum: RSI, MACD, Stochastic, Williams %R
  - Volatility: ATR, Bollinger Bands, Keltner, SuperTrend
  - Volume: OBV, VWAP, MFI, Volume Momentum
  - Trend: Moving Averages (MA50/MA200), ADX, DI+/DI-
- **Signal Fusion**: Weighted ensemble combining all indicators
- **Two-timeframe trend detection**: MA50 (macro) + MA20 (micro) for robust signals

### Value Investing Integration
- **Graham's Defensive Investor criteria** (7-point checklist)
- **Lynch's GARP** (PEG ratio analysis)
- **Greenblatt's Magic Formula** (Earnings Yield + ROC)
- **Composite scoring**: 50% technical + 50% fundamental
- **20+ fundamental metrics**: P/E, P/B, ROE, ROIC, margins, debt, dividends
- Daily fundamental updates from fundamentus.com.br

### News Sentiment Analysis
- **Translation pipeline**: Portuguese → English (Google Translate)
- **FinBERT sentiment analysis**: State-of-the-art financial NLP
- **Multi-source**: newsdata.io API + Google News RSS fallback
- **Smart caching**: 4h trading hours, 12h overnight
- **Two-pass processing**: Technical screening → News enhancement for top candidates

### Position Sizing
- **Kelly Criterion** for optimal sizing
- **Volatility-adjusted** for risk management
- **Fundamental quality adjustments**:
  - Quality picks: +20% position boost
  - Value picks: Moderate positions (contrarian)
  - Poor fundamentals: -50% reduction
  - Avoid flag: Signal blocked

### Google Sheets Integration
- **Automatic sync** - Daily signals exported to Google Sheets
- **52 columns** - All technical + fundamental metrics
- **Historical logging** - Append-only for backtesting
- **Shared access** - Multi-user collaboration

---

## Architecture

```
stock-signals/
├── production_brapi.py           # Main production runner (BrAPI)
├── run_production.py             # Simplified runner with caching
├── production_simple.py          # Legacy yfinance runner
├── monitor_live.py               # Two-pass live monitor
│
├── src/
│   ├── data/
│   │   └── brapi_client.py       # BrAPI client for Brazilian stocks
│   │
│   ├── signals/
│   │   └── trend_detector_v2.py  # Dual-timeframe trend detection
│   │
│   ├── fundamentals/
│   │   ├── fundamentus_scraper.py  # Scrapes fundamentus.com.br
│   │   ├── scorer.py             # Graham/Lynch/Greenblatt scoring
│   │   └── integration.py        # Combines technical + fundamental
│   │
│   ├── news/
│   │   ├── free_news_client.py   # News + translation + FinBERT
│   │   └── news_cache.py         # TTL cache
│   │
│   ├── indicators/
│   │   ├── momentum.py           # RSI, MACD, Stochastic, etc.
│   │   ├── volatility.py         # ATR, Bollinger, Keltner
│   │   ├── volume.py             # OBV, VWAP, MFI
│   │   ├── trend.py              # MA, ADX, SuperTrend
│   │   └── signal_fusion.py      # Weighted ensemble
│   │
│   ├── alerts/
│   │   └── alert_generator.py   # Format signals for Telegram
│   │
│   └── config.py               # Centralized configuration
│
├── data/
│   ├── validated_tickers.json    # 214 IBOV + SMLL tickers
│   ├── quotes_cache.json         # Cached BrAPI quotes
│   ├── full_results.json         # Latest analysis results
│   └── fundamentals/
│       ├── fundamentals_cache.json   # Daily scraped data
│       └── fundamental_scores.json  # Computed scores
│
├── scripts/
│   ├── update_fundamentals.py    # Daily scraper script
│   ├── sheets_sync.py            # Google Sheets sync
│   └── update_ticker_list.py     # Ticker maintenance
│
└── config/
    └── thresholds.json           # Runtime thresholds
```

---

## Processing Flow

### Production Pipeline (production_brapi.py)

```
┌─────────────────────────────────────────────────────────┐
│              Load 214 Tickers                           │
│         (IBOV + SMLL from JSON)                         │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Fetch Price Data (BrAPI)                   │
│  • Real-time quotes (5-ticker batches)                  │
│  • 3 months historical data (parallel workers)          │
│  • Retry logic for 502 errors                           │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│          Technical Analysis (12 indicators)             │
│  • Trend Detection (MA50/MA20)                          │
│  • Signal Fusion (weighted ensemble)                    │
│  • Volume/Volatility features                           │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│       Fundamental Analysis Integration                  │
│  • Load cached fundamentals (daily update)              │
│  • Graham/Lynch/Greenblatt scoring                      │
│  • Combine: 50% tech + 50% fund                         │
│  • Pass raw metrics (P/E, ROE, ROIC, etc.)              │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              News Sentiment (Top 10)                    │
│  • Fetch for top candidates only                        │
│  • Translate Portuguese → English                       │
│  • FinBERT sentiment analysis                           │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│            Generate Trading Signal                      │
│  • STRONG_BUY: Composite >= 75                          │
│  • BUY: Composite >= 60                                 │
│  • HOLD: Composite 40-60                                │
│  • SELL/STRONG_SELL: Composite < 40                     │
│  • AVOID: Poor fund + poor tech                         │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│           Position Sizing                               │
│  • Kelly Criterion (volatility-adjusted)                │
│  • Quality picks: +20% boost                            │
│  • Value picks: 15-50% position                         │
│  • Poor fundamentals: -50% reduction                    │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Export to Google Sheets                    │
│  • 52 columns per stock                                 │
│  • All technical + fundamental metrics                  │
│  • Append for historical tracking                       │
└─────────────────────────────────────────────────────────┘
```

---

## Signal Types

| Signal | Composite Score | Condition |
|--------|-----------------|-----------|
| STRONG_BUY | ≥ 75 | Quality fundamentals + strong technicals |
| BUY | ≥ 60 | Acceptable fundamentals |
| HOLD | 40-60 | Mixed or weak signals |
| SELL | < 40 | Poor fundamentals + weak technicals |
| STRONG_SELL | < 25 | Very poor overall |
| **AVOID** | Any | Fundamentals < 40 + Technicals < 35 (signal blocked) |

---

## Fundamental Scoring

### Graham's Defensive Investor (7-point checklist)
1. P/E ratio ≤ 15
2. P/B ratio ≤ 1.5
3. P/E × P/B ≤ 22.5 (Graham's formula)
4. Current ratio ≥ 2
5. Debt/Equity ≤ 1
6. Positive earnings
7. Dividend yield ≥ 2% (bonus)

### Lynch's GARP (PEG Analysis)
- PEG = P/E ÷ Growth Rate
- PEG < 0.5: Excellent (100 points)
- PEG < 1.0: Good (80 points)
- PEG < 1.5: Fair (60 points)
- PEG < 2.0: Poor (40 points)
- PEG ≥ 2.0: Very Poor (20 points)

### Greenblatt's Magic Formula
- Earnings Yield = EBIT / Enterprise Value
- Return on Capital (ROCIC proxy)
- Combined ranking: Lower is better

### Raw Metrics Passed to Results
All 20+ fundamental metrics are available in results:
- **Valuation**: P/E, P/B, P/FCF, EV/EBITDA
- **Profitability**: ROE, ROIC, Net Margin, EBIT Margin
- **Financial Health**: Debt/Equity, Current Ratio, Asset Turnover
- **Dividends**: Div Yield, Payout Ratio
- **Growth**: Revenue Growth, Earnings Growth

---

## Position Sizing Logic

### Kelly Criterion
```python
# Win probability
p = winning_days / total_days

# Loss probability
q = 1 - p

# Win/Loss ratio
b = avg_win / avg_loss

# Kelly fraction
kelly = (b × p - q) / b

# Half-Kelly for safety
position = kelly × 0.5 × confidence
```

### Fundamental Adjustments

| Condition | Adjustment | Reason |
|-----------|------------|--------|
| Quality pick (ROE > 20, ROIC > 15) | +20% | Compounders deserve larger position |
| Value pick (P/E < 10, P/B < 1) | 15-50% | Contrarian opportunity |
| Poor fundamentals (< 40) | -50% | Risk reduction |
| Avoid flag | 0. Blocked | No position |

---

## Quick Start

### Installation

```bash
git clone https://github.com/arnonbruno/stock-signals.git
cd stock-signals
pip install -r requirements.txt
```

### Set Environment Variables

```bash
# Required
export NEWSDATA_API_KEY="your_newsdata_io_key"
export BRAPI_API_KEY="your_brapi_key"  # Optional for higher limits

# For Google Sheets sync
export GOOGLE_CREDENTIALS_PATH="/path/to/credentials.json"
```

### Run Full Analysis

```python
from run_production import ProductionRunner

runner = ProductionRunner()
results = runner.run()

# Results include:
# - 214 stocks analyzed
# - Technical + fundamental scores
# - Trading signals
# - Position sizing
```

### Run with BrAPI (Recommended)

```python
from production_brapi import BrAPIProductionRunner

runner = BrAPIProductionRunner(
    use_news=True,
    use_fundamentals=True
)
results = runner.run()

# Outputs:
# - data/full_results.json (all results)
# - data/quotes_cache.json (price cache)
```

### Sync to Google Sheets

```python
from scripts.sheets_sync import SheetsSync

sync = SheetsSync(sheet_id="your_sheet_id")
sync.append_results(results)

# 52 columns per stock:
# - Ticker, Price, Date
# - Signal, Composite Score
# - Tech Score, Fund Score
# - All 20+ fundamental metrics
# - Position size, Risk level
```

### Generate Trading Alerts

```python
from src.alerts.alert_generator import generate_trading_alerts

alert = generate_trading_alerts(results, top_n=5)
print(alert)
```

---

## Configuration

### Thresholds (config/thresholds.json)

```json
{
  "thresholds": {
    "default": {
      "buy_confidence": 0.55,
      "sell_confidence": 0.45,
      "min_score": 0.25,
      "stop_loss": 0.15
    },
    "bull": {
      "buy_confidence": 0.50,
      "sell_confidence": 0.40,
      "min_score": 0.20,
      "stop_loss": 0.15
    },
    "bear": {
      "buy_confidence": 0.65,
      "sell_confidence": 0.35,
      "min_score": 0.30,
      "stop_loss": 0.10
    }
  }
}
```

### Ticker List (data/validated_tickers.json)

- **214 stocks total**
  - 83 IBOV constituents
  - 131 SMLL constituents
- Verified against B3 official index composition
- Stocks outside IBOV/SMLL are not included

### BrAPI Config (data/brapi_config.yaml)

```yaml
api_key: "your_brapi_key"
base_url: "https://brapi.dev/api"
batch_size: 5
retry_attempts: 3
retry_delay: 0.5
```

---

## Daily Jobs

### Update Fundamentals (6:00 AM Mon-Fri)

```bash
python scripts/update_fundamentals.py --force
```

- Scrapes all 214 stocks from fundamentus.com.br
- Takes ~72 seconds
- Updates `data/fundamentals/fundamentals_cache.json`

### Sync to Google Sheets (After Market Open)

```bash
python scripts/sheets_sync.py
```

- Exports latest results to Google Sheets
- Appends to historical log
- 52 columns per stock

### Schedule via Cron

```python
# Add via OpenClaw
cron.add(
    name="Update Fundamentals Daily",
    schedule="0 6 * * 1-5",
    command="python scripts/update_fundamentals.py --force"
)

cron.add(
    name="Sync to Google Sheets",
    schedule="30 10 * * 1-5",
    command="python scripts/sheets_sync.py"
)
```

---

## Output Example

```
🚨 TOP TRADING OPPORTUNITIES
Generated: 10:15:00

1️⃣  PRIO3 - STRONG_BUY 🟢
    Price: R$55.02
    └─ Drivers:
       • Trend: UPTREND (95% confidence)
       • Fundamentals: Grade B
         P/E: 4.8 | P/B: 3.87
         ROE: 38.7% | Div: 0%
         ✅ Low P/E (4.8), High ROE (38.7%)
       • Position: 15% (HIGH conviction)
    └─ Levels:
       Entry: R$55.02 | Stop: R$49.52
       Targets: R$66.02 (2:1) | R$77.02 (3:1)

2️⃣  CURY3 - STRONG_BUY 🟢
    Price: R$41.67
    └─ Drivers:
       • Trend: UPTREND (75% confidence)
       • Fundamentals: Grade A
         P/E: 14.7 | P/B: 9.27
         ROE: 62.9% | Div: 9.4%
         ✅ High ROE (62.9%), Excellent ROIC (35.7%)
       • Position: 15% (MEDIUM conviction)

3️⃣  TGMA3 - STRONG_BUY 🟢
    Price: R$40.48
    └─ Drivers:
       • Trend: UPTREND (80% confidence)
       • Fundamentals: Grade A
         P/E: 9.7 | P/B: 2.71
         ROE: 28.0% | Div: 10.9%
         ✅ Low P/E (9.7), High ROE (28.0%)
       • Position: 15% (MEDIUM conviction)

======================================================================
📊 Summary: 17 STRONG_BUY | 43 BUY | 65 HOLD | 8 SELL | 6 STRONG_SELL
======================================================================
```

---

## Google Sheets Columns (52 total)

| Category | Columns |
|----------|---------|
| **Basic** | Date, Ticker, Price, Signal, Composite Score |
| **Technical** | Tech Score, Trend, Confidence, MA50, MA200 |
| **Fundamental Scores** | Fund Score, Graham Score, Lynch Score, Greenblatt Score |
| **Valuation** | P/E, P/B, P/S, P/FCF, EV/EBITDA, PEG |
| **Profitability** | ROE, ROIC, Net Margin, EBIT Margin, Gross Margin |
| **Financial Health** | Debt/Equity, Current Ratio, Asset Turnover |
| **Dividends** | Div Yield, Payout Ratio |
| **Growth** | Revenue Growth, Earnings Growth, Book Value Growth |
| **Position** | Position Size, Risk Level, Entry, Stop, Target |

---

## News Sources

1. **newsdata.io API** (primary)
   - 200 credits/day free tier
   - Market endpoint for financial news
   - Fallback when exhausted

2. **Google News RSS** (free fallback)
   - No API limits
   - Used when newsdata.io budget exhausted

### Translation Pipeline

Brazilian news is in Portuguese, but FinBERT is trained on English:

```python
from deep_translator import GoogleTranslator

# Translate headline before sentiment analysis
translator = GoogleTranslator(source='pt', target='en')
translated = translator.translate("Petrobras lucro recorde")
# Result: "Petrobras profit record"
```

---

## Performance

| Metric | Value |
|--------|-------|
| Processing time | ~30 sec for 214 stocks (BrAPI) |
| Price fetch time | ~5 sec for 200 quotes |
| API calls per cycle | 10-30 news calls |
| Memory usage | ~500MB (FinBERT model) |
| Parallel workers | 4 (configurable) |
| Google Sheets sync | ~10 sec for 200 rows |

---

## Data Sources

| Data Type | Source | Update Frequency |
|-----------|--------|------------------|
| **Real-time prices** | BrAPI (brapi.dev) | On-demand |
| **Historical data** | BrAPI | On-demand |
| **Fundamentals** | Fundamentus | Daily (6:00 AM) |
| **News** | newsdata.io + Google News | On-demand |

---

## Logs & Monitoring

- Analysis results: `data/full_results.json`
- Price cache: `data/quotes_cache.json`
- API budget: `api_budget.json`
- News cache: `news_sentiment_cache.json`
- Fundamental cache: `data/fundamentals/fundamentals_cache.json`
- System review: `REVIEW.md`

---

## Branches

| Branch | Status | Description |
|--------|--------|-------------|
| `master` | Stable | Production-ready code |
| `fundamental-analysis` | **Active** | Integrated value investing + BrAPI |

---

## License

MIT License - See LICENSE file for details.

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `python -m pytest tests/`
4. Submit a pull request

---

## Support

- Issues: https://github.com/arnonbruno/stock-signals/issues
- Repository: https://github.com/arnonbruno/stock-signals
- Google Sheet: https://docs.google.com/spreadsheets/d/1OGwb43E3_CCxcmuzxHuxTWR_PS6QwD7HkKIdn06sx1E/edit
