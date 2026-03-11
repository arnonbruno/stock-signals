# Code Review: Stock Signals System

## Overview
A thorough review of the `stock-signals` project (main branch / fundamental-analysis integrated version).
The system represents a sophisticated intersection of technical analysis (trend detection, feature engineering) and fundamental analysis (value, quality, momentum), coupled with smart position sizing via the Kelly Criterion.

## Strengths & SOTA Alignment

1. **Dual-Timeframe Trend Detection (`TrendDetectorV2`)**:
   - ✅ **SOTA alignment**: Moving away from single-period linear regressions to a 50-day macro + 20-day micro approach is an industry-standard way to prevent getting whipsawed by bull-market pullbacks.
   - ✅ **Robust Statistics**: Uses `scipy.stats.theilslopes` (Theil-Sen estimator), which is robust to outliers, solving the classic issue where a single gap-up/down distorts the OLS slope.
   - ✅ **Volatility normalization**: Normalizing the slope by ATR-like measure rather than standard deviation is exactly how systematic trend-followers measure trend strength.

2. **Position Sizing (`calculate_kelly_position`)**:
   - ✅ **SOTA alignment**: Implementing the true Kelly Criterion ($f^* = (bp - q) / b$) based on the historical win/loss ratio and average payoff of the asset.
   - ✅ **Risk Management**: Employs the "Half-Kelly" ($0.5 \times f^*$) scaled by signal confidence, which is universally recommended by quants to minimize drawdowns while maintaining geometric growth. Minimum and maximum exposure bounds are properly enforced.

3. **Fundamental Scoring (`scorer.py`)**:
   - ✅ Comprehensive mapping of three major schools of thought:
     - **Graham** (Defensive value: P/E < 15, P/B < 1.5, Current Ratio > 2)
     - **Lynch** (GARP: PEG ratio)
     - **Greenblatt** (Magic Formula: Earnings Yield + ROC)
   - ✅ Graceful handling of missing fundamental data (e.g., redistributing weights if PEG or EV/EBITDA data is missing).

4. **Integration Engine (`integration.py`)**:
   - ✅ Elegant matrix combinations: Quality + Uptrend = Momentum boost; Value + Sideways = Contrarian opportunity.
   - ✅ **Avoid Flags**: Using poor fundamentals to override bullish technicals acts as an excellent "trap" prevention mechanism.

5. **Alert & Reporting System (`alert_generator.py`)**:
   - ✅ Includes sector deduplication (e.g., picking the better of PETR3 vs PETR4) and sector diversification to prevent over-concentration in energy or banks.

## Areas for Improvement & Vulnerabilities

### 1. Feature Engineering (`feature_engineering.py`)
- **Vulnerability**: Currently, `add_sector_momentum` and `add_correlation_features` return **hardcoded placeholders** (`0.5`, `'stable'`).
  - *Impact*: The system expects these features to be real, but they are mocked.
  - *Fix*: Implement actual sector ETF/index comparative returns and rolling window correlations.
- **Volume Momentum Bug Risk**: In `add_volume_features`, `volume_trend` uses `np.polyfit` over the last 10 days of volume. Raw volume is highly volatile; a single block trade 2 days ago will skew the slope. It should be smoothed (e.g., using a 3-day SMA of volume before fitting the slope) or measured via OBV (On-Balance Volume).

### 2. Trend Confidence & Sigmoid Scaling (`production_simple.py`)
- **Issue**: In `analyze_ticker`, news sentiment is applied using a sigmoid: `1 / (1 + np.exp(-5 * news_sentiment))`. 
  - If sentiment is `0.2` (positive), sigmoid = `0.73`. The code does `position_size *= 0.73` — which actually **reduces** the position size by 27% instead of boosting it.
  - *Fix*: The scaling factor should be centered around `1.0`. A correct multiplier would be `1.0 + (sentiment * 0.5)` or a shifted sigmoid like `0.5 + (1 / (1 + np.exp(-5 * news_sentiment)))`.

### 3. Kelly Criterion Data Leakage / Lookback
- **Issue**: `calculate_kelly_position` calculates the win probability and average win/loss using the `Close` price returns of the *entire* provided dataframe (which is often the last 120 days).
  - *Impact*: This measures the stock's overall daily return statistics over the last 120 days, NOT the win rate of the actual trading strategy. Kelly requires the probability of *the strategy's trades* winning. 
  - *Fix*: While calculating historical daily drift is an okay proxy, a more SOTA approach uses the actual backtested hit rate of the `TrendDetectorV2` signals for that specific ticker.

### 4. Data Fetching Resiliency (`fundamentus_scraper.py`)
- **Issue**: The scraper relies heavily on `requests.get` with standard headers. Fundamentus is known to aggressively block scrapers and change their HTML structure.
  - *Impact*: If the `<table>` layout changes, the fundamental leg of the engine will silently fail or return zeros.
  - *Fix*: Add structural validation (e.g., ensuring we got >10 valid metrics) and implement a robust fallback (e.g., fetching from BrAPI if Fundamentus fails, as noted in the lessons learned).

### 5. Config Centralization
- **Issue**: Hardcoded thresholds exist in `production_simple.py` (e.g., `if len(data) < 50: return None`). This should ideally be pulled from `src/config.py`.

## Summary
The architectural foundation of this system is **excellent and highly aligned with state-of-the-art retail quant systems**. The integration of robust regression for trend analysis, multi-factor value scoring, and Kelly-based sizing is top-tier. 

To reach true production-grade resilience, the mathematical logic around the news sentiment multiplier needs an immediate fix, and the mocked features in `feature_engineering.py` should be fully implemented or removed to prevent false confidence in the model's outputs.