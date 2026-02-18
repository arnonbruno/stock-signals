#!/usr/bin/env python3
"""
Threshold Optimization Script

Standalone script to run threshold optimization using historical data.
Optimizes trading thresholds for different market regimes.

Usage:
    python scripts/optimize_thresholds.py
    python scripts/optimize_thresholds.py --tickers PETR4.SA VALE3.SA ITUB4.SA
    python scripts/optimize_thresholds.py --period 365 --no-regime
    python scripts/optimize_thresholds.py --output custom_thresholds.json

Requirements:
    - yfinance for historical data (free, no API limits)
    - Walk-forward validation with 6-month train, 2-month test windows
    - Optimizes using last 1.5 years of data
"""

import sys
import os
import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yfinance as yf

from src.validation.threshold_optimizer import (
    ThresholdOptimizer,
    save_thresholds_to_config,
    OptimizationResult
)
from src.strategy.regime_detection import MarketRegimeDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Default tickers for optimization (liquid IBOV stocks)
DEFAULT_TICKERS = [
    'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'BBAS3.SA',
    'ABEV3.SA', 'B3SA3.SA', 'SUZB3.SA', 'WEGE3.SA', 'RENT3.SA'
]


def fetch_historical_data(tickers: List[str], 
                          days: int = 546,
                          max_retries: int = 3) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical price data for multiple tickers.
    
    Args:
        tickers: List of ticker symbols
        days: Number of days of history (default 1.5 years)
        max_retries: Maximum retry attempts per ticker
    
    Returns:
        Dict mapping ticker to DataFrame
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    data = {}
    
    logger.info(f"\n{'='*60}")
    logger.info(f"FETCHING HISTORICAL DATA")
    logger.info(f"{'='*60}")
    logger.info(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"Days: {days}")
    logger.info(f"Tickers: {len(tickers)}")
    
    for ticker in tickers:
        logger.info(f"  Fetching {ticker}...")
        
        for attempt in range(max_retries):
            try:
                df = yf.download(
                    ticker,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    progress=False
                )
                
                if not df.empty and len(df) >= 200:
                    data[ticker] = df
                    logger.info(f"      ✅ {len(df)} days")
                    break
                else:
                    logger.info(f"      ⚠️ Insufficient data ({len(df)} days)")
                    break
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)
                else:
                    logger.info(f"      ❌ Error: {e}")
    
    logger.info(f"\n✅ Successfully fetched {len(data)}/{len(tickers)} tickers")
    
    return data


def fetch_market_data(days: int = 546) -> pd.DataFrame:
    """Fetch IBOV index data for regime detection."""
    logger.info(f"  Fetching IBOV market data...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        market_data = yf.download(
            '^BVSP',
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            progress=False
        )
        
        if not market_data.empty:
            logger.info(f"      ✅ {len(market_data)} days")
            return market_data
    except Exception as e:
        logger.info(f"      ❌ Error: {e}")
    
    return None


def combine_ticker_data(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine multiple ticker data into a single DataFrame for optimization.
    Uses average close prices across all tickers.
    """
    if not data:
        return None
    
    # Align all data on dates
    all_dates = None
    for ticker, df in data.items():
        if all_dates is None:
            all_dates = df.index
        else:
            all_dates = all_dates.union(df.index)
    
    all_dates = all_dates.sort_values()
    
    # Average close prices
    combined = pd.DataFrame(index=all_dates)
    closes = []
    
    for ticker, df in data.items():
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        reindexed = df['Close'].reindex(all_dates, method='ffill')
        closes.append(reindexed)
    
    # Stack and average
    close_df = pd.concat(closes, axis=1)
    combined['Close'] = close_df.mean(axis=1)
    
    return combined


def segment_by_regime(data: pd.DataFrame, 
                      market_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Segment price data by market regime.
    
    Returns:
        Dict mapping regime to DataFrame subset
    """
    if market_data is None or len(market_data) < 200:
        return {'default': data}
    
    detector = MarketRegimeDetector()
    
    # Detect regimes
    regimes = {}
    current_regime = []
    current_dates = []
    
    window = 200
    
    for i in range(window, len(market_data)):
        subset = market_data.iloc[i-window:i]
        result = detector.detect_regime(subset)
        regime = result['regime']
        date = market_data.index[i]
        
        if regime not in regimes:
            regimes[regime] = []
        
        regimes[regime].append(date)
    
    # Create data subsets for each regime
    result = {}
    
    for regime, dates in regimes.items():
        if len(dates) >= 50:  # Need at least 50 days
            regime_data = data.loc[data.index.isin(dates)]
            if len(regime_data) >= 50:
                result[regime] = regime_data
    
    # Always include default (all data)
    result['default'] = data
    
    return result


def run_optimization(tickers: List[str],
                     period_days: int = 546,
                     use_regime: bool = True,
                     output_path: str = None) -> Dict:
    """
    Run full threshold optimization.
    
    Args:
        tickers: List of tickers to use for optimization
        period_days: Historical period in days
        use_regime: Whether to optimize by regime
        output_path: Path to save results
    
    Returns:
        Dict with optimization results
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"THRESHOLD OPTIMIZATION")
    logger.info(f"{'='*70}")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Tickers: {tickers}")
    logger.info(f"Period: {period_days} days (~{period_days/365:.1f} years)")
    logger.info(f"Regime optimization: {'ON' if use_regime else 'OFF'}")
    
    # Fetch data
    ticker_data = fetch_historical_data(tickers, period_days)
    
    if not ticker_data:
        logger.error("❌ No data fetched. Cannot run optimization.")
        return None
    
    # Combine ticker data
    combined_data = combine_ticker_data(ticker_data)
    
    if combined_data is None or len(combined_data) < 200:
        logger.error("❌ Insufficient combined data. Cannot run optimization.")
        return None
    
    logger.info(f"\nCombined data: {len(combined_data)} days")
    
    # Fetch market data for regime detection
    market_data = None
    if use_regime:
        market_data = fetch_market_data(period_days)
    
    # Initialize optimizer
    optimizer = ThresholdOptimizer(
        train_window_days=126,  # ~6 months
        test_window_days=42      # ~2 months
    )
    
    results = {}
    
    if use_regime and market_data is not None:
        # Regime-specific optimization
        logger.info(f"\n{'='*60}")
        logger.info(f"REGIME-SPECIFIC OPTIMIZATION")
        logger.info(f"{'='*60}")
        
        regime_segments = segment_by_regime(combined_data, market_data)
        
        for regime in ['bull', 'bear', 'sideways', 'default']:
            if regime in regime_segments:
                logger.info(f"\n--- Optimizing for {regime.upper()} regime ---")
                regime_data = regime_segments[regime]
                result = optimizer.optimize(regime_data, regime=regime)
                results[regime] = result
            else:
                logger.info(f"\n--- Skipping {regime.upper()} (insufficient data) ---")
    else:
        # Default optimization only
        logger.info(f"\n{'='*60}")
        logger.info(f"DEFAULT OPTIMIZATION")
        logger.info(f"{'='*60}")
        
        result = optimizer.optimize(combined_data, regime='default')
        results['default'] = result
    
    # Save results
    config_path = save_thresholds_to_config(results, output_path)
    
    # Print summary
    print_summary(results)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"OPTIMIZATION COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Config saved to: {config_path}")
    
    return results


def print_summary(results: Dict[str, OptimizationResult]):
    """Print optimization results summary."""
    logger.info(f"\n{'='*70}")
    logger.info(f"OPTIMIZATION RESULTS SUMMARY")
    logger.info(f"{'='*70}")
    
    for regime, result in results.items():
        logger.info(f"\n{regime.upper()} Regime:")
        logger.info(f"  Thresholds:")
        logger.info(f"    buy_confidence: {result.best_thresholds.buy_confidence:.2f}")
        logger.info(f"    sell_confidence: {result.best_thresholds.sell_confidence:.2f}")
        logger.info(f"    min_score: {result.best_thresholds.min_score:.2f}")
        logger.info(f"    stop_loss: {result.best_thresholds.stop_loss:.2f}")
        logger.info(f"  Performance:")
        logger.info(f"    Sharpe Ratio: {result.sharpe_ratio:.3f}")
        logger.info(f"    Total Return: {result.total_return:.2%}")
        logger.info(f"    Max Drawdown: {result.max_drawdown:.2%}")
        logger.info(f"    Win Rate: {result.win_rate:.2%}")
        logger.info(f"    Trades: {result.num_trades}")
        logger.info(f"    Validation Windows: {result.validation_windows}")


def main():
    parser = argparse.ArgumentParser(
        description="Optimize trading thresholds using historical data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default settings
    python scripts/optimize_thresholds.py

    # Use specific tickers
    python scripts/optimize_thresholds.py --tickers PETR4.SA VALE3.SA ITUB4.SA

    # Use 1 year of data
    python scripts/optimize_thresholds.py --period 365

    # Skip regime-specific optimization
    python scripts/optimize_thresholds.py --no-regime

    # Save to custom path
    python scripts/optimize_thresholds.py --output my_thresholds.json
        """
    )
    
    parser.add_argument(
        '--tickers',
        nargs='+',
        default=DEFAULT_TICKERS,
        help='Tickers to use for optimization (default: 10 liquid IBOV stocks)'
    )
    
    parser.add_argument(
        '--period',
        type=int,
        default=546,
        help='Historical period in days (default: 546, ~1.5 years)'
    )
    
    parser.add_argument(
        '--no-regime',
        action='store_true',
        help='Skip regime-specific optimization'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for thresholds config (default: config/thresholds.json)'
    )
    
    args = parser.parse_args()
    
    # Run optimization
    results = run_optimization(
        tickers=args.tickers,
        period_days=args.period,
        use_regime=not args.no_regime,
        output_path=args.output
    )
    
    if results:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())