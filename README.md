# Stock-Signals

**Production-grade trading signal system for Brazilian equities (IBOV + SMLL)**

A multi-indicator signal fusion system that analyzes 150 stocks every 10 minutes during market hours, combining technical indicators, volume analysis, and news sentiment to generate actionable trading recommendations.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Signal Fusion System](#signal-fusion-system)
3. [Indicators Explained](#indicators-explained)
4. [Position Sizing](#position-sizing)
5. [News Sentiment](#news-sentiment)
6. [Threshold Configuration](#threshold-configuration)
7. [Output Format](#output-format)
8. [Quick Start](#quick-start)
9. [Architecture](#architecture)
10. [Deployment](#deployment)

---

## How It Works

The system uses a **two-pass architecture** for efficient processing:

### Pass 1: Technical Screening (~1 minute)
Analyzes all 150 stocks with technical indicators only (no news):
- Fast screening to identify candidates
- Outputs: fusion score, confidence, position sizing
- Top 10 candidates selected for news enhancement

### Pass 2: News Enhancement (~10 minutes)
Fetches and analyzes news for top 10 candidates only:
- News sources: newsdata.io → Google News RSS fallback
- Translation: Portuguese → English via deep_translator
- Sentiment: FinBERT analysis on translated headlines
- Fusion score recalculated with news component

### Processing Flow

```
Price Data (yfinance)
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│              PASS 1: TECHNICAL SCREENING                 │
│                  (~1 min for 150 stocks)                 │
├─────────────────────────────────────────────────────────┤
│  MOMENTUM    │  VOLATILITY  │   VOLUME    │    TREND    │
│  (25% wt)    │   (20% wt)   │   (25% wt)  │   (30% wt)  │
│              │              │             │             │
│  • RSI       │  • ATR       │  • OBV      │  • MA Align │
│  • MACD      │  • Bollinger │  • VWAP     │  • ADX      │
│  • Stochastic│  • Keltner   │  • MFI      │  • SuperTrend│
│  • Williams %R│             │  • Vol Mom  │             │
│  • ROC       │              │             │             │
└─────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│              TOP 10 CANDIDATES SELECTED                   │
│  (based on fusion score + confidence)                    │
└─────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│              PASS 2: NEWS ENHANCEMENT                    │
│                (~10 min for 10 stocks)                   │
├─────────────────────────────────────────────────────────┤
│  NEWS FETCH          │  TRANSLATION    │  SENTIMENT     │
│  • newsdata.io       │  • deep_translator│  • FinBERT   │
│  • Google News RSS   │  • PT → EN      │  • +0.15 wt   │
│    (fallback)        │                 │               │
└─────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│              FUSION RECALCULATION                        │
│  Trend (40%) + Momentum (25%) + Volatility (20%) + News (15%) │
└─────────────────────────────────────────────────────────┘
       │
       ▼
   FINAL SIGNAL
   (BUY / SELL / HOLD)
```

**Total processing time: ~11 minutes** (vs 5+ hours for single-pass with news)

---

## Signal Fusion System

### What Is Fusion Score?

The **fusion score** is a weighted average of signals from 12+ technical indicators, normalized to a range of -1.0 (strong bearish) to +1.0 (strong bullish).

### How It's Calculated

Each indicator produces a **signal** (bullish/bearish/neutral) and **strength** (0.0 to 1.0). These are aggregated by category:

```python
fused_score = (momentum_score × 0.25) +
              (volatility_score × 0.20) +
              (volume_score × 0.25) +
              (trend_score × 0.30)
```

### Category Weights

| Category | Weight | Rationale |
|----------|--------|-----------|
| **Trend** | 30% | Most reliable for direction |
| **Momentum** | 25% | Confirms trend strength |
| **Volume** | 25% | Validates participation |
| **Volatility** | 20% | Adjusts for risk |

### Interpreting Fusion Score

| Score Range | Interpretation |
|-------------|----------------|
| +0.5 to +1.0 | Strong bullish alignment |
| +0.3 to +0.5 | Moderate bullish |
| -0.3 to +0.3 | Mixed/consolidation |
| -0.5 to -0.3 | Moderate bearish |
| -1.0 to -0.5 | Strong bearish alignment |

### Confidence

**Confidence** measures how many indicators agree on the signal direction:

```python
confidence = (agreeing_indicators / total_indicators) × strength_factor
```

High confidence (80%+) means most indicators point in the same direction.

---

## Indicators Explained

### MOMENTUM (25% weight)

**RSI (Relative Strength Index)**
- Measures speed and magnitude of price movements
- Range: 0-100
- Overbought: >70 (potential reversal down)
- Oversold: <30 (potential reversal up)
- Signal: bullish when 30-70 and rising, bearish when falling from >70

**MACD (Moving Average Convergence Divergence)**
- Compares 12-day and 26-day exponential moving averages
- Signal line: 9-day EMA of MACD
- Bullish crossover: MACD crosses above signal line
- Bearish crossover: MACD crosses below signal line

**Stochastic Oscillator**
- Compares closing price to price range over period
- %K line and %D signal line
- Overbought: >80, Oversold: <20

**Williams %R**
- Similar to Stochastic but inverted
- Range: 0 to -100
- Overbought: >-20, Oversold: <-80

**ROC (Rate of Change)**
- Measures percentage price change over period
- Positive: bullish momentum, Negative: bearish momentum

---

### VOLATILITY (20% weight)

**ATR (Average True Range)**
- Measures market volatility by decomposing range
- High ATR = high volatility regime
- Used to adjust position sizing and stop-loss levels

**Bollinger Bands**
- Standard deviation bands around moving average
- Price above upper band: potentially overbought
- Price below lower band: potentially oversold
- Bandwidth indicates volatility regime

**Keltner Channels**
- ATR-based channels around EMA
- Similar to Bollinger but uses ATR instead of std dev
- More stable in trending markets

---

### VOLUME (25% weight)

**OBV (On-Balance Volume)**
- Cumulative volume based on price direction
- Rising OBV + rising price = strong trend
- Divergence = potential reversal

**VWAP (Volume Weighted Average Price)**
- Average price weighted by volume
- Price above VWAP: bullish intraday
- Price below VWAP: bearish intraday

**MFI (Money Flow Index)**
- RSI-like oscillator using volume
- Range: 0-100
- Overbought: >80, Oversold: <20

**Volume Momentum**
- Current volume vs average volume
- >2x average = unusual volume (confirms signal)
- <0.5x average = low participation (weakens signal)

---

### TREND (30% weight)

**Moving Average Alignment**
- MA50 vs MA200 crossover analysis
- Golden Cross: MA50 crosses above MA200 (bullish)
- Death Cross: MA50 crosses below MA200 (bearish)
- Also checks MA20 alignment for short-term trend

**ADX (Average Directional Index)**
- Measures trend strength (not direction)
- ADX >25: strong trend present
- ADX <20: weak/no trend
- Combined with +DI/-DI for direction

**SuperTrend**
- Trend indicator using ATR
- Clear support/resistance levels
- Green = uptrend, Red = downtrend

---

## Position Sizing

### Kelly Criterion

The system uses the **Kelly Criterion** for optimal position sizing:

```python
# Win probability
p = winning_days / total_days

# Loss probability
q = 1 - p

# Average win/loss ratio
b = avg_win / avg_loss

# Kelly fraction
kelly = (b × p - q) / b

# Half-Kelly for safety
position = kelly × 0.5 × confidence
```

### Volatility Adjustment

High volatility stocks get smaller positions:

```python
if volatility_regime == 'high':
    position *= 0.8  # Reduce by 20%
```

### Position Limits

| Limit | Value | Reason |
|-------|-------|--------|
| Minimum | 10% | Meaningful exposure |
| Maximum | 40% | Risk management |
| Default | 20% | Conservative baseline |

---

## News Sentiment

### Translation Layer

Brazilian news headlines are in Portuguese, but FinBERT was trained on English financial text. The system uses **deep_translator** (Google Translate) to translate headlines before analysis:

```python
from deep_translator import GoogleTranslator

def _translate_to_english(self, text: str) -> str:
    translator = GoogleTranslator(source='pt', target='en')
    return translator.translate(text)
```

This ensures accurate sentiment analysis for Portuguese-language financial news.

### FinBERT Analysis

The system uses **FinBERT**, a BERT model trained on English financial text:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
```

Outputs:
- Positive sentiment: +1.0
- Negative sentiment: -1.0
- Neutral: 0.0

### News Sources

1. **newsdata.io API** (primary, 200 credits/day) - Market endpoint for financial news
2. **Google News RSS** (free fallback) - Used when API budget exhausted

### Smart Caching

- **Trading hours**: 4-hour TTL
- **Overnight**: 12-hour TTL
- Sentiment cache stored in `news_sentiment_cache.json`

### Two-Pass Strategy

News is only fetched for **top 10 candidates** after technical screening:
- Reduces API calls from 150 to 10 per cycle
- Total processing: ~11 minutes (vs 5+ hours)
- Budget: ~10-30 API credits/day (well under 200 limit)

---

## Threshold Configuration

Thresholds are stored in `config/thresholds.json` and can be updated without code changes.

### Available Thresholds

| Threshold | Description | Default |
|-----------|-------------|---------|
| `buy_confidence` | Minimum confidence for BUY signal | 55% |
| `sell_confidence` | Minimum confidence for SELL signal | 45% |
| `min_score` | Minimum fusion score for action | 25% |
| `stop_loss` | Default stop-loss percentage | 15% |

### Regime-Specific Thresholds

```json
{
  "thresholds": {
    "default": {"buy_confidence": 0.55, "sell_confidence": 0.45, "min_score": 0.25, "stop_loss": 0.15},
    "bull": {"buy_confidence": 0.50, "sell_confidence": 0.40, "min_score": 0.20, "stop_loss": 0.15},
    "bear": {"buy_confidence": 0.65, "sell_confidence": 0.35, "min_score": 0.30, "stop_loss": 0.10},
    "sideways": {"buy_confidence": 0.55, "sell_confidence": 0.45, "min_score": 0.25, "stop_loss": 0.20}
  }
}
```

### Optimizing Thresholds

Run walk-forward optimization:

```bash
python scripts/optimize_thresholds.py --period 180
```

---

## Output Format

### Alert Structure

```
🚨 TOP TRADING OPPORTUNITIES
Generated: 10:00:00

1️⃣  TICKER - BUY 🟢
    Price: R$XX.XX
    └─ Drivers:
       • Trend: UPTREND (XX% confidence)
       • Fusion: Strong bullish alignment (+0.XX)
       • News: +0.XX sentiment
         ✅ Positive news boost (+X% position)
         📰 Headlines:
            • Article title... (Source)
       • Position: XX% (VERY HIGH conviction)
    └─ Levels:
       Entry: R$XX.XX | Stop: R$XX.XX
       Targets: R$XX.XX (2:1) | R$XX.XX (3:1)
    └─ Watch for:
       ⚠️ Price drops below 50-day MA → Exit
```

### Entry/Exit Levels

For BUY signals:
- **Entry**: Current price
- **Stop Loss**: Entry × (1 - stop_loss%)
- **Target 1**: Entry + 2×Risk (2:1 risk-reward)
- **Target 2**: Entry + 3×Risk (3:1 risk-reward)

### Reversal Signals

The system monitors for conditions that could flip the recommendation:
- Price crossing key moving averages
- RSI reaching extreme levels
- Major news catalysts
- Volume divergences

---

## Quick Start

### Installation

```bash
git clone https://github.com/arnonbruno/stock-signals.git
cd stock-signals
pip install -r requirements.txt
```

### Set Environment

```bash
export NEWSDATA_API_KEY="your_key_here"
```

### Run Analysis

```python
from production_enhanced import EnhancedProductionRunner

runner = EnhancedProductionRunner(
    use_news=True,
    use_fusion=True,
    n_workers=4
)

results = runner.run()
```

### Run Monitor

```bash
python monitor_live.py
```

---

## Architecture

```
stock-signals/
├── monitor_live.py              # Two-pass live monitor (technical + news)
├── production_enhanced.py       # SOTA analysis engine
├── production_simple.py         # Lightweight runner
│
├── src/
│   ├── indicators/
│   │   ├── momentum.py          # RSI, MACD, Stochastic, etc.
│   │   ├── volatility.py        # ATR, Bollinger, Keltner
│   │   ├── volume.py            # OBV, VWAP, MFI
│   │   ├── trend.py             # MA, ADX, SuperTrend
│   │   └── signal_fusion.py     # Weighted ensemble
│   │
│   ├── signals/
│   │   └── trend_detector_v2.py # Dual-timeframe detection
│   │
│   ├── news/
│   │   ├── free_news_client.py  # News fetching + translation + FinBERT
│   │   ├── news_cache.py        # TTL cache management
│   │   ├── api_budget_tracker.py# API budget enforcement
│   │   └── historical_sentiment.py # Backtest sentiment
│   │
│   ├── features/
│   │   └── feature_engineering.py
│   │
│   ├── strategy/
│   │   └── regime_detection.py  # Bull/bear/sideways
│   │
│   ├── risk/
│   │   └── risk_parity.py       # Correlation adjustments
│   │
│   ├── validation/
│   │   ├── threshold_optimizer.py
│   │   └── walk_forward.py
│   │
│   └── config.py                # Threshold management
│
├── config/
│   └── thresholds.json          # Runtime thresholds
│
├── scripts/
│   └── optimize_thresholds.py   # Threshold optimization
│
└── tests/                       # Unit tests
```

### Two-Pass Architecture Details

**`monitor_live.py`** implements the two-pass architecture:

```python
# Configuration
NEWS_CANDIDATES = 10  # Top N stocks for news enhancement

# Pass 1: Technical screening (no news)
runner = EnhancedProductionRunner(use_news=False)
results = runner.run()  # ~1 min for 150 stocks

# Pass 2: News enhancement for top candidates
top_candidates = sorted(results, key=lambda x: x['confidence'], reverse=True)[:NEWS_CANDIDATES]
for candidate in top_candidates:
    news = news_client.fetch_news(candidate['ticker'])
    sentiment = finbert_analyzer.analyze(news, translate=True)
    candidate['fusion_score'] = recalculate_with_news(sentiment)
```

---

## Deployment

### Cron Setup

```bash
# Hourly during market hours (9:00-20:00 Brasilia)
0 9-20 * * 1-5 cd /path/to/stock-signals && python monitor_live.py >> logs/monitor.log 2>&1
```

### Requirements

- Python 3.10+
- 4GB RAM minimum
- Internet connection for data/news

### Monitoring

- **Universe**: 150 stocks (65 IBOV + 85 SMLL)
- **Frequency**: Hourly during market hours
- **Hours**: 9:00-20:00 GMT-3 (market hours + extended)
- **Processing time**: ~11 minutes per cycle
- **Output**: Telegram alerts + JSON files

---

## Metrics Summary

| Metric | Description | Range |
|--------|-------------|-------|
| **fused_score** | Weighted indicator sum | -1.0 to +1.0 |
| **confidence** | Signal agreement rate | 0% to 100% |
| **position_size** | Kelly-adjusted allocation | 10% to 40% |
| **news_sentiment** | FinBERT + lexicon ensemble | -1.0 to +1.0 |
| **volume_momentum** | Current vs average volume | Ratio |
| **volatility_regime** | ATR-based regime | low/medium/high |

---

## Authors

- **Bruno Santos** - Initial work, architecture, testing
- **TARS** - System design, indicator integration, documentation

---

## License

MIT License - See LICENSE file for details.
