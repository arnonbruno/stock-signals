#!/usr/bin/env python3
"""
Backtest Comparison: Stock-Signals Model vs Passive Strategy

This script compares our signal-based strategy against a passive BOVA11.SA buy-and-hold.

Period: 2024-08-18 to 2025-02-18 (6 months)
Starting Capital: R$ 100,000

Strategy:
- Day 1: Run signal analysis on all stocks, buy all with "BUY" signal
- Hold until end date
- Track daily portfolio value

Passive:
- Buy BOVA11.SA on day 1
- Hold until end date
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Core components
from production_enhanced import EnhancedProductionRunner, IBOV_TICKERS, SMLL_TICKERS
from src.news.historical_sentiment import HistoricalSentimentEngine, SentimentBacktestAdapter

# Plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

# Constants
START_DATE = datetime(2024, 8, 18)
END_DATE = datetime(2025, 2, 18)
STARTING_CAPITAL = 100000  # R$ 100,000

ALL_TICKERS = IBOV_TICKERS + SMLL_TICKERS


def fetch_price_data(tickers: list, start_date: datetime, end_date: datetime) -> dict:
    """
    Fetch price data for all tickers in the period.
    
    Returns dict: ticker -> DataFrame
    """
    print(f"\n📥 Fetching price data for {len(tickers)} tickers...")
    print(f"   Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    price_data = {}
    failed = []
    
    for i, ticker in enumerate(tickers):
        if (i + 1) % 20 == 0:
            print(f"   Progress: {i+1}/{len(tickers)}")
        
        try:
            # Get extra history for indicators (need ~120 days before start)
            data_start = start_date - timedelta(days=180)
            
            data = yf.download(
                ticker,
                start=data_start.strftime("%Y-%m-%d"),
                end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
                progress=False
            )
            
            if not data.empty and len(data) >= 100:
                price_data[ticker] = data
            else:
                failed.append(ticker)
                
        except Exception as e:
            failed.append(ticker)
    
    print(f"\n✅ Successfully fetched: {len(price_data)} tickers")
    if failed:
        print(f"   ⚠️ Failed: {len(failed)} tickers")
    
    return price_data


def run_signal_analysis_on_date(price_data: dict, as_of_date: datetime) -> list:
    """
    Run signal analysis on all tickers as of a specific date.
    
    Uses historical sentiment proxy (no news API).
    
    Returns list of results with BUY signals.
    """
    print(f"\n🔍 Running signal analysis as of {as_of_date.strftime('%Y-%m-%d')}...")
    
    # Initialize runner WITHOUT news (use historical sentiment)
    runner = EnhancedProductionRunner(
        use_news=False,  # No news API
        use_fusion=True,  # Use full signal fusion
        use_regime=True   # Use regime detection
    )
    
    # Fetch market data for regime detection (up to as_of_date)
    runner.fetch_market_data(days=365)
    runner.detect_market_regime()
    
    results = []
    
    for ticker, data in price_data.items():
        try:
            # Slice data to as_of_date for analysis
            if isinstance(data.index, pd.DatetimeIndex):
                data_slice = data.loc[:as_of_date].copy()
            else:
                data_slice = data.copy()
            
            if len(data_slice) < 50:
                continue
            
            # Analyze with our signal system
            result = runner.analyze_ticker(ticker, data=data_slice, as_of_date=as_of_date)
            
            if result:
                results.append(result)
                
        except Exception as e:
            pass  # Skip problematic tickers
    
    # Filter for BUY signals
    buy_signals = [r for r in results if r['signal'] == 'BUY']
    
    print(f"\n   Analysis complete: {len(results)} tickers analyzed")
    print(f"   🟢 BUY signals: {len(buy_signals)}")
    print(f"   🔴 SELL signals: {len([r for r in results if r['signal'] == 'SELL'])}")
    print(f"   ⚪ HOLD signals: {len([r for r in results if r['signal'] == 'HOLD'])}")
    
    return buy_signals, results


def build_portfolio(buy_signals: list, starting_capital: float) -> dict:
    """
    Build equal-weighted portfolio from BUY signals.
    
    Returns dict with:
    - holdings: {ticker: shares}
    - entry_prices: {ticker: price}
    - allocation per stock
    """
    if not buy_signals:
        print("\n⚠️ No BUY signals - cannot build portfolio")
        return None
    
    # Equal weight allocation
    n_stocks = len(buy_signals)
    allocation_per_stock = starting_capital / n_stocks
    
    holdings = {}
    entry_prices = {}
    
    print(f"\n💼 Building Portfolio:")
    print(f"   Total Capital: R$ {starting_capital:,.2f}")
    print(f"   Stocks with BUY signal: {n_stocks}")
    print(f"   Allocation per stock: R$ {allocation_per_stock:,.2f}")
    
    for signal in buy_signals:
        ticker = signal['ticker']
        price = signal['price']
        
        # Calculate shares (fractional allowed)
        shares = allocation_per_stock / price
        
        holdings[ticker] = shares
        entry_prices[ticker] = price
    
    return {
        'holdings': holdings,
        'entry_prices': entry_prices,
        'n_stocks': n_stocks,
        'allocation_per_stock': allocation_per_stock
    }


def track_portfolio_value(portfolio: dict, price_data: dict, 
                          start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Track daily portfolio value over the backtest period.
    """
    print(f"\n📊 Tracking daily portfolio values...")
    
    holdings = portfolio['holdings']
    
    # Get all trading days in the period
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    
    daily_values = []
    last_valid_value = None
    
    for date in date_range:
        date_value = 0.0
        valid_stocks = 0
        
        for ticker, shares in holdings.items():
            if ticker in price_data:
                data = price_data[ticker]
                
                # Find price on this date (or nearest)
                try:
                    if date in data.index:
                        price = float(data.loc[date, 'Close'].iloc[0] if hasattr(data.loc[date, 'Close'], 'iloc') else data.loc[date, 'Close'])
                        if not np.isnan(price) and price > 0:
                            date_value += shares * price
                            valid_stocks += 1
                except:
                    # Try nearest date (forward fill)
                    try:
                        # Get all dates up to current date
                        valid_idx = data.index[data.index <= date]
                        if len(valid_idx) > 0:
                            nearest_date = valid_idx[-1]
                            price = float(data.loc[nearest_date, 'Close'].iloc[0] if hasattr(data.loc[nearest_date, 'Close'], 'iloc') else data.loc[nearest_date, 'Close'])
                            if not np.isnan(price) and price > 0:
                                date_value += shares * price
                                valid_stocks += 1
                    except:
                        pass
        
        # Only add if we have valid data for most stocks (at least 50%)
        if valid_stocks >= len(holdings) * 0.5 and date_value > 0:
            last_valid_value = date_value
            daily_values.append({
                'date': date,
                'value': date_value,
                'valid_stocks': valid_stocks
            })
        elif last_valid_value is not None:
            # Use last valid value for missing days (forward fill)
            daily_values.append({
                'date': date,
                'value': last_valid_value,
                'valid_stocks': valid_stocks
            })
    
    df = pd.DataFrame(daily_values)
    if not df.empty:
        df.set_index('date', inplace=True)
    
    return df


def track_passive_strategy(price_data: dict, start_date: datetime, 
                           end_date: datetime, starting_capital: float) -> pd.DataFrame:
    """
    Track BOVA11.SA buy-and-hold passive strategy.
    """
    print(f"\n📈 Tracking passive strategy (BOVA11.SA)...")
    
    # Fetch BOVA11 data with more buffer days
    try:
        bova_data = yf.download(
            'BOVA11.SA',
            start=(start_date - timedelta(days=30)).strftime("%Y-%m-%d"),
            end=(end_date + timedelta(days=5)).strftime("%Y-%m-%d"),
            progress=False
        )
        
        # Handle multi-index columns from yfinance
        if isinstance(bova_data.columns, pd.MultiIndex):
            bova_data.columns = bova_data.columns.get_level_values(0)
            
    except Exception as e:
        print(f"   ❌ Failed to fetch BOVA11.SA: {e}")
        return None
    
    if bova_data.empty or len(bova_data) < 5:
        print(f"   ❌ No BOVA11.SA data available")
        return None
    
    # Ensure index is timezone-naive for comparison
    if hasattr(bova_data.index, 'tz') and bova_data.index.tz is not None:
        bova_data.index = bova_data.index.tz_localize(None)
    
    # Find entry price on start_date (or nearest trading day after)
    start_date_naive = pd.Timestamp(start_date).tz_localize(None)
    
    try:
        # Try exact date first
        if start_date_naive in bova_data.index:
            close_val = bova_data.loc[start_date_naive, 'Close']
            entry_price = float(close_val.iloc[0] if hasattr(close_val, 'iloc') else close_val)
        else:
            # Find nearest date on or after start_date
            valid_dates = bova_data.index[bova_data.index >= start_date_naive]
            if len(valid_dates) > 0:
                nearest_date = valid_dates[0]
                close_val = bova_data.loc[nearest_date, 'Close']
                entry_price = float(close_val.iloc[0] if hasattr(close_val, 'iloc') else close_val)
                print(f"   Using nearest trading day: {nearest_date.strftime('%Y-%m-%d')}")
            else:
                # Fall back to first available date
                nearest_date = bova_data.index[0]
                close_val = bova_data.loc[nearest_date, 'Close']
                entry_price = float(close_val.iloc[0] if hasattr(close_val, 'iloc') else close_val)
                print(f"   Using first available date: {nearest_date.strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"   ❌ Could not determine BOVA11 entry price: {e}")
        return None
    
    shares = starting_capital / entry_price
    
    print(f"   Entry Price: R$ {entry_price:.2f}")
    print(f"   Shares: {shares:.2f}")
    
    # Track daily values
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    
    daily_values = []
    
    for date in date_range:
        try:
            if date in bova_data.index:
                price = float(bova_data.loc[date, 'Close'].iloc[0] if hasattr(bova_data.loc[date, 'Close'], 'iloc') else bova_data.loc[date, 'Close'])
            else:
                idx = bova_data.index.get_indexer([date], method='nearest')[0]
                price = float(bova_data.iloc[idx]['Close'])
            
            daily_values.append({
                'date': date,
                'value': shares * price
            })
        except:
            pass
    
    df = pd.DataFrame(daily_values)
    if not df.empty:
        df.set_index('date', inplace=True)
    
    return df


def calculate_metrics(values: pd.Series, starting_capital: float) -> dict:
    """
    Calculate performance metrics.
    
    Returns:
    - Total return %
    - Sharpe ratio (annualized)
    - Max drawdown
    - Volatility (annualized)
    """
    if values is None or len(values) == 0:
        return None
    
    # Filter out zero and negative values
    values = values[values > 0].dropna()
    
    if len(values) == 0:
        return None
    
    # Returns
    returns = values.pct_change().dropna()
    
    # Filter out infinite and NaN returns
    returns = returns[np.isfinite(returns)]
    
    if len(returns) == 0:
        return None
    
    # Total return
    final_value = values.iloc[-1]
    total_return = (final_value - starting_capital) / starting_capital * 100
    
    # Sharpe ratio (assuming risk-free rate = 0 for simplicity, or 10% annual for Brazil CDI)
    risk_free_rate = 0.10 / 252  # Daily risk-free rate (CDI ~10% annual)
    
    excess_returns = returns - risk_free_rate
    if len(excess_returns) > 0 and excess_returns.std() > 0:
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    else:
        sharpe = 0
    
    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    # Volatility (annualized)
    volatility = returns.std() * np.sqrt(252) * 100
    
    # Win days / loss days
    win_days = (returns > 0).sum()
    loss_days = (returns < 0).sum()
    
    return {
        'total_return': total_return,
        'final_value': final_value,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'volatility': volatility,
        'win_days': win_days,
        'loss_days': loss_days,
        'total_days': len(returns)
    }


def create_plot(strategy_values: pd.DataFrame, passive_values: pd.DataFrame,
                starting_capital: float, output_path: str):
    """
    Create comparison plot.
    """
    print(f"\n📉 Creating comparison plot...")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Main comparison plot
    ax1 = axes[0]
    
    if strategy_values is not None and not strategy_values.empty:
        ax1.plot(strategy_values.index, strategy_values['value'], 
                label='Stock-Signals Strategy', color='#2ecc71', linewidth=2)
    
    if passive_values is not None and not passive_values.empty:
        ax1.plot(passive_values.index, passive_values['value'], 
                label='Passive (BOVA11.SA)', color='#3498db', linewidth=2)
    
    ax1.axhline(y=starting_capital, color='gray', linestyle='--', alpha=0.5, label='Starting Capital')
    
    ax1.set_title('Backtest Comparison: Stock-Signals vs Passive Strategy\n(6 months: Aug 2024 - Feb 2025)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Portfolio Value (R$)', fontsize=11)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'R$ {x:,.0f}'))
    
    # Format x-axis
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    # Drawdown comparison
    ax2 = axes[1]
    
    if strategy_values is not None and not strategy_values.empty:
        strat_returns = strategy_values['value'].pct_change().dropna()
        strat_cum = (1 + strat_returns).cumprod()
        strat_max = strat_cum.cummax()
        strat_dd = (strat_cum - strat_max) / strat_max * 100
        ax2.fill_between(strat_dd.index, strat_dd, 0, alpha=0.3, color='#2ecc71', label='Strategy Drawdown')
        ax2.plot(strat_dd.index, strat_dd, color='#2ecc71', linewidth=1)
    
    if passive_values is not None and not passive_values.empty:
        pass_returns = passive_values['value'].pct_change().dropna()
        pass_cum = (1 + pass_returns).cumprod()
        pass_max = pass_cum.cummax()
        pass_dd = (pass_cum - pass_max) / pass_max * 100
        ax2.fill_between(pass_dd.index, pass_dd, 0, alpha=0.3, color='#3498db', label='Passive Drawdown')
        ax2.plot(pass_dd.index, pass_dd, color='#3498db', linewidth=1)
    
    ax2.set_title('Drawdown Comparison', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Drawdown (%)', fontsize=11)
    ax2.legend(loc='lower left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Plot saved to: {output_path}")


def save_results(results: dict, output_path: str):
    """
    Save detailed results to JSON file.
    """
    import json
    
    # Convert non-serializable types
    def convert(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d')
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj
    
    results = convert(results)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ Results saved to: {output_path}")


def main():
    """
    Main backtest execution.
    """
    print("=" * 70)
    print("🚀 BACKTEST COMPARISON: Stock-Signals vs Passive Strategy")
    print("=" * 70)
    print(f"\n📅 Period: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
    print(f"💰 Starting Capital: R$ {STARTING_CAPITAL:,.2f}")
    print(f"📊 Universe: {len(ALL_TICKERS)} stocks (IBOV + SMLL)")
    
    # Step 1: Fetch all price data
    price_data = fetch_price_data(ALL_TICKERS, START_DATE, END_DATE)
    
    if len(price_data) < 50:
        print("\n❌ Insufficient price data. Aborting.")
        return
    
    # Step 2: Run signal analysis on start date
    buy_signals, all_results = run_signal_analysis_on_date(price_data, START_DATE)
    
    if not buy_signals:
        print("\n❌ No BUY signals generated. Cannot build portfolio.")
        # Still run passive for comparison
        
        portfolio = None
        strategy_values = None
        strategy_metrics = None
    else:
        # Step 3: Build portfolio
        portfolio = build_portfolio(buy_signals, STARTING_CAPITAL)
        
        # Step 4: Track daily values
        strategy_values = track_portfolio_value(portfolio, price_data, START_DATE, END_DATE)
        
        # Step 5: Calculate strategy metrics
        if strategy_values is not None and not strategy_values.empty:
            strategy_metrics = calculate_metrics(strategy_values['value'], STARTING_CAPITAL)
        else:
            strategy_metrics = None
    
    # Step 6: Track passive strategy
    passive_values = track_passive_strategy(price_data, START_DATE, END_DATE, STARTING_CAPITAL)
    
    if passive_values is not None and not passive_values.empty:
        passive_metrics = calculate_metrics(passive_values['value'], STARTING_CAPITAL)
    else:
        passive_metrics = None
    
    # Step 7: Create output
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(output_dir, '..', 'backtest_results.png')
    results_path = os.path.join(output_dir, '..', 'backtest_results.json')
    
    # Create plot
    create_plot(strategy_values, passive_values, STARTING_CAPITAL, plot_path)
    
    # Prepare results
    results = {
        'backtest_info': {
            'start_date': START_DATE.strftime('%Y-%m-%d'),
            'end_date': END_DATE.strftime('%Y-%m-%d'),
            'starting_capital': STARTING_CAPITAL,
            'total_stocks_analyzed': len(all_results) if all_results else 0,
            'buy_signals_count': len(buy_signals) if buy_signals else 0
        },
        'portfolio': {
            'n_stocks': portfolio['n_stocks'] if portfolio else 0,
            'allocation_per_stock': portfolio['allocation_per_stock'] if portfolio else 0,
            'holdings': portfolio['holdings'] if portfolio else {},
            'entry_prices': portfolio['entry_prices'] if portfolio else {}
        },
        'buy_signals': [
            {
                'ticker': s['ticker'],
                'price': s['price'],
                'confidence': s['confidence'],
                'trend': s['trend'],
                'fused_score': s.get('fused_score', 0)
            }
            for s in buy_signals
        ] if buy_signals else [],
        'strategy_metrics': strategy_metrics,
        'passive_metrics': passive_metrics,
        'daily_values': {
            'strategy': {str(k.date()) if hasattr(k, 'date') else str(k): v for k, v in strategy_values['value'].to_dict().items()} if strategy_values is not None else {},
            'passive': {str(k.date()) if hasattr(k, 'date') else str(k): v for k, v in passive_values['value'].to_dict().items()} if passive_values is not None else {}
        }
    }
    
    # Save results
    save_results(results, results_path)
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 BACKTEST RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Metric':<25} {'Strategy':>20} {'Passive':>20}")
    print("-" * 65)
    
    if strategy_metrics:
        print(f"{'Final Value':<25} R$ {strategy_metrics['final_value']:>15,.2f} ", end="")
    else:
        print(f"{'Final Value':<25} {'N/A':>20} ", end="")
    
    if passive_metrics:
        print(f"R$ {passive_metrics['final_value']:>15,.2f}")
    else:
        print(f"{'N/A':>20}")
    
    if strategy_metrics:
        print(f"{'Total Return':<25} {strategy_metrics['total_return']:>19.2f}% ", end="")
    else:
        print(f"{'Total Return':<25} {'N/A':>20} ", end="")
    
    if passive_metrics:
        print(f"{passive_metrics['total_return']:>19.2f}%")
    else:
        print(f"{'N/A':>20}")
    
    if strategy_metrics:
        print(f"{'Sharpe Ratio':<25} {strategy_metrics['sharpe_ratio']:>19.2f} ", end="")
    else:
        print(f"{'Sharpe Ratio':<25} {'N/A':>20} ", end="")
    
    if passive_metrics:
        print(f"{passive_metrics['sharpe_ratio']:>19.2f}")
    else:
        print(f"{'N/A':>20}")
    
    if strategy_metrics:
        print(f"{'Max Drawdown':<25} {strategy_metrics['max_drawdown']:>18.2f}% ", end="")
    else:
        print(f"{'Max Drawdown':<25} {'N/A':>20} ", end="")
    
    if passive_metrics:
        print(f"{passive_metrics['max_drawdown']:>18.2f}%")
    else:
        print(f"{'N/A':>20}")
    
    if strategy_metrics:
        print(f"{'Volatility (Ann.)':<25} {strategy_metrics['volatility']:>18.2f}% ", end="")
    else:
        print(f"{'Volatility (Ann.)':<25} {'N/A':>20} ", end="")
    
    if passive_metrics:
        print(f"{passive_metrics['volatility']:>18.2f}%")
    else:
        print(f"{'N/A':>20}")
    
    print("\n" + "=" * 70)
    
    if strategy_metrics and passive_metrics:
        diff_return = strategy_metrics['total_return'] - passive_metrics['total_return']
        outperformance = "OUTPERFORMED" if diff_return > 0 else "UNDERPERFORMED"
        
        print(f"\n🎯 VERDICT: Strategy {outperformance} passive by {abs(diff_return):.2f}%")
    
    print("\n📁 Files generated:")
    print(f"   - {plot_path}")
    print(f"   - {results_path}")
    
    return results


if __name__ == "__main__":
    main()
