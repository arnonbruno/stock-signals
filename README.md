# Stock-Signals

**Production-grade trading signal system for Brazilian equities (IBOV + SMLL)**

A multi-indicator, value investing fusion system that analyzes 214 stocks every 10 minutes during market hours, combining technical indicators, volume analysis, news sentiment, and fundamental analysis to generate actionable trading recommendations.

---

## Features

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

---

## Architecture

```
stock-signals/
├── production_simple.py        # Main analysis engine with fundamentals
├── monitor_live.py             # Two-pass live monitor
├── src/
│   ├── signals/
│   │   └── trend_detector_v2.py   # Dual-timeframe trend detection
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
│   ├── config.py               # Centralized configuration
│   │
│   └── features/
│       └── feature_engineering.py
│
├── data/
│   ├── validated_tickers.json    # 214 IBOV + SMLL tickers
│   └── fundamentals/
│       ├── fundamentals_cache.json   # Daily scraped data
│       └── fundamental_scores.json  # Computed scores
│
├── config/
│   └── thresholds.json            # Runtime thresholds
│
└── scripts/
    ├── update_fundamentals.py     # Daily scraper script
    └── update_ticker_list.py     # Ticker maintenance
```

---

## Processing Flow

### Production Pipeline (production_simple.py)

```
┌─────────────────────────────────────────────────────────┐
│              Load 214 Tickers                        │
│         (IBOV + SMLL from JSON)                       │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Fetch Price Data                         │
│         (yfinance, 120 days history)                   │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│          Technical Analysis                        │
│  • Trend Detection (MA50/MA20)                   │
│  • Signal Fusion (12 indicators)               │
│  • Volume/Volatility features                 │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│       Fundamental Analysis Integration               │
│  • Load cached fundamentals (daily update)          │
│  • Graham/Lynch/Greenblatt scoring                  │
│  • Combine: 50% tech + 50% fund                 │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              News Sentiment                           │
│  • Fetch for top 10 candidates only               │
│  • Translate Portuguese → English               │
│  • FinBERT sentiment analysis                  │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│            Generate Trading Signal                  │
│  • STRONG_BUY: Composite >= 75                 │
│  • BUY: Composite >= 60                        │
│  • HOLD: Composite 40-60                    │
│  • SELL/STRONG_SELL: Composite < 40           │
│  • AVOID: Poor fund + poor tech              │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│           Position Sizing                              │
│  • Kelly Criterion (volatility-adjusted)            │
│  • Quality picks: +20% boost                   │
│  • Value picks: 15-50% position               │
│  • Poor fundamentals: -50% reduction          │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Format Alert                               │
│  • Telegram-ready format                       │
│  • Include P/E, P/B, ROE, Div Yield           │
│  • Strengths, weaknesses, action notes         │
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
export NEWSDATA_API_KEY="your_newsdata_io_key"
```

### Run Analysis (Single Ticker)

```python
from production_simple import SimpleProductionRunner

runner = SimpleProductionRunner(use_news=True, use_fundamentals=True)
result = runner.analyze_ticker('PETR4.SA')
print(result)
```

### Run Full Analysis (All 214 Stocks)

```python
results = runner.run()  # Parallel processing (~2 minutes)
```

### Generate Trading Alerts

```python
from src.alerts.alert_generator import generate_trading_alerts

alert = generate_trading_alerts(results, top_n=5)
print(alert)
```

### Run with Technical Analysis Only

```python
runner = SimpleProductionRunner(use_news=False, use_fundamentals=False)
results = runner.run()
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

---

## Daily Jobs

### Update Fundamentals (6:00 AM Mon-Fri)

```bash
python scripts/update_fundamentals.py --force
```

- Scrapes all 214 stocks from fundamentus.com.br
- Takes ~72 seconds
- Updates `data/fundamentals/fundamentals_cache.json`

### Schedule via Cron

```python
# Add via OpenClaw
cron.add(
    name="Update Fundamentals Daily",
    schedule="0 6 * * 1-5",
    command="python scripts/update_fundamentals.py --force"
)
```

---

## Output Example

```
🚨 TOP TRADING OPPORTUNITIES
Generated: 10:15:00

1️⃣  CURY3 - STRONG_BUY 🟢
    Price: R$41.67
    └─ Drivers:
       • Trend: UPTREND (81% confidence)
       • Fundamentals: Grade A
         P/E: 14.7 | P/B: 9.27
         ROE: 62.9% | Div: 9.4%
         ✅ High ROE (62.9%), Excellent ROIC (35.7%)
       • Position: 15% (MEDIUM conviction)
    └─ Watch for:
       VALUE PLAY: Undervalued with acceptable quality
       QUALITY PLAY: High ROE/ROIC compounder

2️⃣  JHSF3 - STRONG_BUY 🟢
    Price: R$10.09
    └─ Drivers:
       • Trend: UPTREND (81% confidence)
       • Fundamentals: Grade A
         P/E: 5.3 | P/B: 1.10
         ROE: 20.6% | Div: 4.7%
         ✅ Low P/E (5.3), High ROE (20.6%)
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
📊 Summary: 5 BUY signals | 0 SELL signals
======================================================================
```

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
| Processing time | ~2 min for 214 stocks |
| API calls per cycle | 10-30 news calls |
| Memory usage | ~500MB (FinBERT model) |
| Parallel workers | 4 (configurable) |

---

## Logs & Monitoring

- Analysis results: `live_alerts.json`
- API budget: `api_budget.json`
- News cache: `news_sentiment_cache.json`
- Fundamental cache: `data/fundamentals/fundamentals_cache.json`

---

## Branches

| Branch | Status | Description |
|--------|--------|-------------|
| `master` | Stable | Production-ready code |
| `fundamental-analysis` | Testing | Integrated value investing |

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