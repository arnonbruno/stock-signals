# Threshold Optimization System

## Overview

The threshold optimization system automatically finds optimal trading thresholds for the stock-signals strategy. It uses walk-forward validation to prevent overfitting and supports regime-specific optimization (bull/bear/sideways markets).

## How Thresholds Work

### Threshold Parameters

| Parameter | Description | Default Range |
|-----------|-------------|---------------|
| `buy_confidence` | Minimum confidence required for BUY signals | 0.50 - 0.75 |
| `sell_confidence` | Minimum confidence required for SELL signals | 0.35 - 0.50 |
| `min_score` | Minimum fused score threshold | 0.20 - 0.35 |
| `stop_loss` | Stop loss percentage | 0.10 - 0.20 |

### Regime-Specific Thresholds

The system uses different thresholds depending on market conditions:

- **Bull Market**: Lower confidence requirements, larger positions
  - `buy_confidence`: ~0.50 (more trades)
  - `stop_loss`: ~0.15 (room for volatility)

- **Bear Market**: Higher confidence requirements, tighter stops
  - `buy_confidence`: ~0.65 (only high-quality setups)
  - `stop_loss`: ~0.10 (limit losses)

- **Sideways Market**: Neutral settings
  - `buy_confidence`: ~0.55 (balanced)
  - `stop_loss`: ~0.20 (wider stops for noise)

## Configuration File

Thresholds are stored in `config/thresholds.json`:

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

## Updating Thresholds

### Manual Updates

Edit `config/thresholds.json` directly:

```json
{
  "thresholds": {
    "default": {
      "buy_confidence": 0.60,  // Changed from 0.55
      ...
    }
  }
}
```

The system will automatically load the new thresholds on the next run.

### Automatic Optimization

Run the optimization script:

```bash
# Full optimization with regime detection
python scripts/optimize_thresholds.py

# Skip regime-specific optimization
python scripts/optimize_thresholds.py --no-regime

# Use specific tickers
python scripts/optimize_thresholds.py --tickers PETR4.SA VALE3.SA ITUB4.SA

# Use different time period
python scripts/optimize_thresholds.py --period 365
```

## Using Thresholds in Code

### Get Thresholds for Regime

```python
from src.config import get_thresholds, get_config

# Get thresholds for specific regime
thresholds = get_thresholds('bull')
print(f"Buy confidence: {thresholds.buy_confidence}")

# Or via config
config = get_config()
buy_conf = config.get_buy_confidence('bull')
```

### In Production Runner

```python
from src.config import get_thresholds

# Detect regime
regime_info = regime_detector.detect_regime(market_data)
regime = regime_info['regime']  # 'bull', 'bear', 'sideways'

# Get regime-specific thresholds
thresholds = get_thresholds(regime)

# Use in signal generation
if signal == 'BUY' and confidence >= thresholds.buy_confidence:
    # Generate buy order
    ...
```

### Reload Without Restart

```python
from src.config import reload_config

# Reload config from file
config = reload_config()

# Now using updated thresholds
```

## Optimization Process

### Walk-Forward Validation

The optimizer uses walk-forward validation to prevent overfitting:

1. **Split data** into rolling windows (6 months train, 2 months test)
2. **Grid search** over all parameter combinations
3. **Test** each combination on out-of-sample data
4. **Select** parameters with best average Sharpe ratio

### Optimization Target

The system optimizes for **Sharpe ratio** (risk-adjusted returns):

```
Sharpe Ratio = (Return - Risk-Free Rate) / Volatility
```

Higher Sharpe = Better risk-adjusted performance.

### Parameter Grid

Default grid search space:

```python
BUY_CONFIDENCE = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
SELL_CONFIDENCE = [0.35, 0.40, 0.45, 0.50]
MIN_SCORE = [0.20, 0.25, 0.30, 0.35]
STOP_LOSS = [0.10, 0.15, 0.20]
```

Total combinations: 6 × 4 × 4 × 3 = **288 combinations**

## Recommended Optimization Schedule

| Frequency | When to Run |
|-----------|-------------|
| Monthly | Regular maintenance |
| Regime Change | When market shifts (bull → bear) |
| Poor Performance | When strategy underperforms |
| Quarterly | Comprehensive review |

## Example: Running Optimization

```bash
$ python scripts/optimize_thresholds.py

======================================================================
THRESHOLD OPTIMIZATION
======================================================================
Start time: 2026-02-18 17:15:00
Tickers: ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', ...]
Period: 546 days (~1.5 years)
Regime optimization: ON

======================================================================
FETCHING HISTORICAL DATA
======================================================================
Period: 2024-08-19 to 2026-02-18
Days: 546
Tickers: 10
  Fetching PETR4.SA... ✅ 540 days
  Fetching VALE3.SA... ✅ 538 days
  ...
✅ Successfully fetched 10/10 tickers

======================================================================
REGIME-SPECIFIC OPTIMIZATION
======================================================================

--- Optimizing for BULL regime ---
Walk-forward splits: 3
Testing combination 1/288
...
Testing combination 288/288

OPTIMIZATION COMPLETE - bull
Best thresholds:
  buy_confidence: 0.52
  sell_confidence: 0.40
  min_score: 0.22
  stop_loss: 0.15

Performance:
  Sharpe Ratio: 1.24
  Total Return: 12.5%
  Max Drawdown: 4.2%
  Win Rate: 58%

...

✅ Thresholds saved to config/thresholds.json
```

## Troubleshooting

### "Insufficient data" error

- Need at least 200 days of historical data
- Try increasing `--period` parameter
- Check if tickers have data

### "No regime segments" error

- Market data (IBOV) not available
- Try `--no-regime` flag

### Performance issues

- Reduce number of tickers
- Use `--no-regime` for faster optimization
- Grid search has 288 combinations by default

## Advanced: Custom Parameter Grid

Create a custom optimizer with different parameters:

```python
from src.validation.threshold_optimizer import ThresholdOptimizer

custom_grid = {
    'buy_confidence': [0.45, 0.50, 0.55, 0.60],
    'sell_confidence': [0.40, 0.45, 0.50],
    'min_score': [0.15, 0.20, 0.25],
    'stop_loss': [0.08, 0.10, 0.12, 0.15]
}

optimizer = ThresholdOptimizer(
    train_window_days=126,
    test_window_days=42,
    param_grid=custom_grid
)

result = optimizer.optimize(data, regime='default')
```

## Files Reference

| File | Purpose |
|------|---------|
| `src/validation/threshold_optimizer.py` | Core optimization logic |
| `src/config.py` | Configuration loading |
| `config/thresholds.json` | Threshold values |
| `scripts/optimize_thresholds.py` | Optimization CLI |
| `docs/threshold_optimization.md` | This documentation |