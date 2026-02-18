#!/usr/bin/env python3
"""
Adaptive Backtest: System with Market Regime Detection

This version detects market conditions and adjusts strategy:
- Bull market: Aggressive (lower confidence, higher positions, less cash)
- Bear market: Defensive (higher confidence, smaller positions, more cash)
- Sideways: Neutral (balanced parameters)

This should perform well in ALL market conditions.
"""

import sys
sys.path.insert(0, '.')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from production_simple import SimpleProductionRunner
from src.strategy.regime_detection import (
    get_regime_detector,
    get_adaptive_params,
    calculate_price_based_sentiment
)
import warnings
warnings.filterwarnings('ignore')

# Test stocks
TEST_STOCKS = [
    'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'BBAS3.SA',
    'ABEV3.SA', 'B3SA3.SA', 'SUZB3.SA', 'WEGE3.SA', 'RENT3.SA',
]

# IBOV index for regime detection (use ticker or get B3 data)
# We'll use the first stock as proxy, or download IBOV separately
IBOV_TICKER = '^BVSP'  # IBOV index

INITIAL_CAPITAL_PER_STOCK = 1000
TOTAL_INITIAL_CAPITAL = INITIAL_CAPITAL_PER_STOCK * len(TEST_STOCKS)

# Backtest period: 3 years
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=3*365)


class AdaptiveActiveStrategy:
    """Active strategy with market regime awareness."""
    
    def __init__(self, tickers: list, initial_capital_per_stock: float):
        self.tickers = tickers
        self.initial_capital_per_stock = initial_capital_per_stock
        self.positions = {}
        self.cash = TOTAL_INITIAL_CAPITAL
        self.runner = SimpleProductionRunner(use_news=False, n_workers=1)
        self.trades = []
        self.portfolio_history = []
        self.regime_history = []
        
        self.regime_detector = get_regime_detector()
        self.adaptive_params = get_adaptive_params()
        
    def run(self, price_data: dict, market_data: pd.DataFrame):
        """Execute adaptive strategy with weekly rebalancing."""
        print("\n" + "="*70)
        print("🤖 ADAPTIVE STRATEGY: Regime-Aware Active Trading")
        print("="*70)
        print(f"Initial Capital: ${TOTAL_INITIAL_CAPITAL:.2f}")
        print(f"Strategy: Trend + Sentiment + Kelly + Regime Detection")
        print()
        
        # Get all trading dates
        all_dates = sorted(price_data[self.tickers[0]].index)
        
        # Rebalance weekly
        rebalance_dates = all_dates[::5]
        
        print(f"  Total trading days: {len(all_dates)}")
        print(f"  Rebalance points: {len(rebalance_dates)}")
        print()
        
        for i, rebalance_date in enumerate(rebalance_dates):
            if i % 10 == 0:
                print(f"  Processing rebalance {i+1}/{len(rebalance_dates)}...")
            
            # 1. DETECT MARKET REGIME (using data up to this date only)
            if rebalance_date in market_data.index:
                market_slice = market_data.loc[:rebalance_date]
            else:
                # Get closest date
                loc = market_data.index.get_loc(rebalance_date, method='nearest')
                market_slice = market_data.iloc[:loc+1]
            
            regime_info = self.regime_detector.detect_regime(market_slice)
            params = self.adaptive_params.get_parameters(
                regime_info['regime'],
                regime_info['strength']
            )
            
            # Log regime every 25 rebalances
            if i % 25 == 0:
                print(f"\n  📅 {rebalance_date.strftime('%Y-%m-%d')}: Regime={regime_info['regime'].upper()} (strength={regime_info['strength']:.2f})")
                print(f"     → confidence_threshold={params['confidence_threshold']:.2f}")
                print(f"     → max_cash_pct={params['max_cash_pct']:.1%}")
                print(f"     → regime_multiplier={params['regime_multiplier']:.2f}")
            
            self.regime_history.append({
                'date': rebalance_date,
                'regime': regime_info['regime'],
                'strength': regime_info['strength'],
                'params': params
            })
            
            # 2. Execute trades based on regime-aware parameters
            for ticker in self.tickers:
                if ticker not in price_data:
                    continue
                
                # Get data up to this date (NO LOOK-AHEAD!)
                data = price_data[ticker].loc[:rebalance_date].copy()
                
                if len(data) < 50:
                    continue
                
                # Analyze
                result = self.runner.analyze_ticker(ticker, data=data)
                
                if result is None:
                    continue
                
                # Inject price-based sentiment
                sentiment = calculate_price_based_sentiment(data)
                result['news_sentiment'] = sentiment
                
                # Get current price
                current_price = data['Close'].iloc[-1]
                if hasattr(current_price, 'item'):
                    current_price = current_price.item()
                elif hasattr(current_price, 'iloc'):
                    current_price = current_price.iloc[0]
                
                signal = result['signal']
                base_position_size = result['position_size']
                confidence = result['confidence']
                
                # Apply regime-aware adjustments
                if signal == "BUY" and ticker not in self.positions:
                    # Adjust position size based on regime
                    adjusted_position = self.adaptive_params.adjust_position_size(
                        base_position_size,
                        regime_info['regime'],
                        regime_info['strength'],
                        params
                    )
                    
                    # Boost with sentiment
                    if abs(sentiment) > 0.1:
                        sigmoid_boost = 1 / (1 + np.exp(-5 * sentiment))
                        adjusted_position *= sigmoid_boost
                    
                    # Calculate position value
                    position_value = min(
                        self.cash * adjusted_position,
                        INITIAL_CAPITAL_PER_STOCK * 2.0  # Allow up to 200%
                    )
                    
                    # Minimum cash constraint (keep some cash for safety)
                    min_cash = self.cash * (1 - params['max_cash_pct'])
                    if position_value > (self.cash - min_cash):
                        position_value = self.cash - min_cash
                    
                    if position_value > 100:
                        shares = position_value / current_price
                        self.positions[ticker] = {
                            'shares': shares,
                            'avg_cost': current_price,
                            'date': rebalance_date,
                            'regime': regime_info['regime']
                        }
                        self.cash -= position_value
                        
                        self.trades.append({
                            'date': rebalance_date,
                            'ticker': ticker,
                            'action': 'BUY',
                            'shares': shares,
                            'price': current_price,
                            'value': position_value,
                            'confidence': confidence,
                            'sentiment': sentiment,
                            'regime': regime_info['regime'],
                            'position_size': adjusted_position
                        })
                
                elif signal == "SELL" and ticker in self.positions:
                    # Check if confidence meets regime-adjusted threshold
                    if confidence >= params['sell_threshold']:
                        pos = self.positions[ticker]
                        shares = pos['shares']
                        sale_value = shares * current_price
                        cost_basis = shares * pos['avg_cost']
                        
                        self.cash += sale_value
                        pnl = sale_value - cost_basis
                        
                        del self.positions[ticker]
                        
                        self.trades.append({
                            'date': rebalance_date,
                            'ticker': ticker,
                            'action': 'SELL',
                            'shares': shares,
                            'price': current_price,
                            'value': sale_value,
                            'pnl': pnl,
                            'sentiment': sentiment,
                            'regime': regime_info['regime'],
                            'confidence': confidence
                        })
            
            # Record portfolio value
            portfolio_value = self.cash
            for ticker, pos in self.positions.items():
                if ticker in price_data and rebalance_date in price_data[ticker].index:
                    price = price_data[ticker].loc[rebalance_date, 'Close']
                    if hasattr(price, 'item'):
                        price = price.item()
                    elif hasattr(price, 'iloc'):
                        price = price.iloc[0]
                    portfolio_value += pos['shares'] * price
            
            self.portfolio_history.append({
                'date': rebalance_date,
                'value': portfolio_value,
                'regime': regime_info['regime']
            })
        
        # Final results
        final_value = self.portfolio_history[-1]['value']
        total_return = (final_value / TOTAL_INITIAL_CAPITAL - 1) * 100
        
        print(f"\n  💰 Final Cash: R${self.cash:.2f} ({self.cash/final_value:.1%})")
        print(f"  💰 Final Value: R${final_value:.2f}")
        print(f"  📈 Return: {total_return:+.2f}%")
        
        # Regime statistics
        regime_counts = {}
        for rh in self.regime_history:
            regime = rh['regime']
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        print(f"\n  📊 Regime Distribution:")
        for regime, count in regime_counts.items():
            pct = count / len(self.regime_history) * 100
            print(f"    {regime.upper()}: {count} periods ({pct:.1f}%)")
        
        # Trade statistics
        buys = [t for t in self.trades if t['action'] == 'BUY']
        sells = [t for t in self.trades if t['action'] == 'SELL']
        
        print(f"\n  📊 Trade Statistics:")
        print(f"    Total Trades: {len(self.trades)}")
        print(f"    Buys: {len(buys)}")
        print(f"    Sells: {len(sells)}")
        
        if sells:
            winning_trades = [t for t in sells if t.get('pnl', 0) > 0]
            total_pnl = sum(t.get('pnl', 0) for t in sells)
            win_rate = len(winning_trades) / len(sells) * 100
            
            print(f"    Win Rate: {win_rate:.1f}%")
            print(f"    Realized P&L: R${total_pnl:.2f}")
        
        return self.portfolio_history


def download_price_data(tickers: list, start_date: datetime, end_date: datetime) -> dict:
    """Download price data for all tickers."""
    print("\n" + "="*70)
    print("📥 DOWNLOADING PRICE DATA (3 YEARS)")
    print("="*70)
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print()
    
    price_data = {}
    
    for ticker in tickers:
        print(f"  {ticker}...", end=" ", flush=True)
        
        try:
            data = yf.download(
                ticker,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                progress=False
            )
            
            if not data.empty and len(data) >= 50:
                price_data[ticker] = data
                print(f"✅ {len(data)} days")
            else:
                print(f"❌ Insufficient data")
        
        except Exception as e:
            print(f"❌ {e}")
    
    print(f"\n✅ Downloaded {len(price_data)}/{len(tickers)} stocks")
    
    return price_data


def calculate_metrics(portfolio_history: list, initial_capital: float) -> dict:
    """Calculate risk-adjusted performance metrics."""
    values = [p['value'] for p in portfolio_history]
    returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
    
    # Total return
    total_return = (values[-1] / initial_capital - 1) * 100
    
    # Volatility (annualized)
    if len(returns) > 1:
        volatility = np.std(returns) * np.sqrt(252) * 100
    else:
        volatility = 0
    
    # Sharpe ratio (assuming 5% risk-free rate)
    if volatility > 0:
        sharpe = (total_return - 5) / volatility
    else:
        sharpe = 0
    
    # Max drawdown
    peak = values[0]
    max_dd = 0
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd
    
    max_drawdown = max_dd * 100
    
    return {
        'total_return': total_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown
    }


def main():
    print("\n" + "="*70)
    print("🚀 ADAPTIVE BACKTEST: Regime-Aware Strategy (3 Years)")
    print("="*70)
    print(f"Test Period: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
    print(f"Initial Capital: ${TOTAL_INITIAL_CAPITAL:.2f}")
    print(f"Stocks: {len(TEST_STOCKS)}")
    print()
    print("🎯 Adaptive Features:")
    print("   • Bull market: Aggressive (lower confidence, higher positions)")
    print("   • Bear market: Defensive (higher confidence, smaller positions)")
    print("   • Sideways: Neutral (balanced parameters)")
    print()
    
    # Download stock data
    stock_data = download_price_data(TEST_STOCKS, START_DATE, END_DATE)
    
    if len(stock_data) < 5:
        print("\n❌ Insufficient data")
        return
    
    # Download market index (IBOV) for regime detection
    print(f"\n📥 Downloading IBOV index for regime detection...")
    try:
        market_data = yf.download(
            IBOV_TICKER,
            start=START_DATE.strftime("%Y-%m-%d"),
            end=END_DATE.strftime("%Y-%m-%d"),
            progress=False
        )
        
        if market_data.empty:
            print(f"❌ Failed to download IBOV, using first stock as proxy")
            market_data = stock_data[TEST_STOCKS[0]]
        else:
            print(f"✅ IBOV: {len(market_data)} days")
    except Exception as e:
        print(f"❌ IBOV error: {e}, using first stock as proxy")
        market_data = stock_data[TEST_STOCKS[0]]
    
    # Run adaptive strategy
    adaptive = AdaptiveActiveStrategy(list(stock_data.keys()), INITIAL_CAPITAL_PER_STOCK)
    adaptive_history = adaptive.run(stock_data, market_data)
    
    # Calculate metrics
    adaptive_metrics = calculate_metrics(adaptive_history, TOTAL_INITIAL_CAPITAL)
    
    # Results
    print("\n" + "="*70)
    print("📊 ADAPTIVE STRATEGY RESULTS")
    print("="*70)
    print()
    
    print(f"  Total Return:     {adaptive_metrics['total_return']:+14.2f}%")
    print(f"  Sharpe Ratio:     {adaptive_metrics['sharpe_ratio']:>14.2f}")
    print(f"  Max Drawdown:     {adaptive_metrics['max_drawdown']:>13.2f}%")
    print(f"  Volatility:       {adaptive_metrics['volatility']:>13.2f}%")
    
    print()
    print("="*70)
    
    # Save results
    with open('adaptive_backtest_results.txt', 'w') as f:
        f.write("ADAPTIVE REGIME-AWARE STRATEGY RESULTS\n")
        f.write("="*70 + "\n\n")
        for k, v in adaptive_metrics.items():
            f.write(f"{k}: {v:.2f}\n")
    
    print("\n✅ Results saved to adaptive_backtest_results.txt")


if __name__ == "__main__":
    main()