#!/usr/bin/env python3
"""
Bear Market Backtest — COVID-era (2020) crash + recovery.

Same logic as fair_backtest.py but focused on a bear market
where the active strategy should show its risk-management edge.
"""

import sys
sys.path.insert(0, '.')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from production_simple import SimpleProductionRunner
import warnings
warnings.filterwarnings('ignore')

# Same 10 major IBOV components
TEST_STOCKS = [
    'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'BBAS3.SA',
    'ABEV3.SA', 'B3SA3.SA', 'SUZB3.SA', 'WEGE3.SA', 'RENT3.SA',
]

INITIAL_CAPITAL_PER_STOCK = 1000
TOTAL_INITIAL_CAPITAL = INITIAL_CAPITAL_PER_STOCK * len(TEST_STOCKS)

# COVID bear market: Jan 2020 (pre-crash) to Dec 2020 (post-recovery)
START_DATE = datetime(2020, 1, 2)
END_DATE   = datetime(2020, 12, 31)


def calculate_price_based_sentiment(data: pd.DataFrame, lookback: int = 10) -> float:
    if len(data) < lookback:
        return 0.0
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    recent = data.tail(lookback)
    returns = recent['Close'].pct_change().dropna()
    if len(returns) == 0:
        momentum_sentiment = 0.0
    else:
        momentum_sentiment = np.clip(returns.mean() * 50, -1, 1)
    vol_sentiment = 0.0
    if len(recent) >= 2:
        recent_returns = recent['Close'].pct_change().dropna()
        if len(recent_returns) > 0 and recent_returns.std() > 0:
            vol_ratio = recent_returns.std() / max(recent_returns.std(), 0.001)
            vol_sentiment = np.clip((0.5 - vol_ratio) * 2, -1, 1)
    return np.clip(momentum_sentiment * 0.7 + vol_sentiment * 0.3, -1, 1)


class PassiveStrategy:
    def __init__(self, tickers: list, initial_capital_per_stock: float):
        self.tickers = tickers
        self.capital_per_stock = initial_capital_per_stock

    def run(self, price_data: dict):
        print("\n" + "="*70)
        print("📈 PASSIVE STRATEGY: Buy & Hold")
        print("="*70)
        all_dates = sorted(price_data[self.tickers[0]].index)
        history = []
        first_date = all_dates[0]
        positions = {}
        for ticker in self.tickers:
            if ticker not in price_data:
                continue
            data = price_data[ticker].loc[:first_date]
            if len(data) > 0:
                price = data['Close'].iloc[-1]
                if hasattr(price, 'item'):
                    price = price.item()
                shares = self.capital_per_stock / price
                positions[ticker] = {'shares': shares, 'avg_cost': price}
        for date in all_dates:
            value = 0
            for ticker, pos in positions.items():
                if ticker in price_data:
                    data = price_data[ticker].loc[:date]
                    if len(data) > 0:
                        price = data['Close'].iloc[-1]
                        if hasattr(price, 'item'):
                            price = price.item()
                        value += pos['shares'] * price
            history.append({'date': date, 'value': value})
        final_value = history[-1]['value'] if history else TOTAL_INITIAL_CAPITAL
        ret = (final_value / TOTAL_INITIAL_CAPITAL - 1) * 100
        print(f"  Final Value: R${final_value:,.2f}")
        print(f"  Return: {ret:+.2f}%")
        return history


class ActiveStrategy:
    MAX_POSITIONS = 10
    TRAILING_STOP_INITIAL = 0.88
    TRAILING_STOP_WIN = 0.90

    def __init__(self, tickers: list, initial_capital_per_stock: float):
        self.tickers = tickers
        self.capital_per_stock = initial_capital_per_stock
        self.positions = {}
        self.cash = TOTAL_INITIAL_CAPITAL
        self.trades = []
        self.portfolio_history = []
        self._position_peaks = {}
        self._position_dates = {}

    def run(self, price_data: dict):
        print("\n" + "="*70)
        print("🤖 ACTIVE STRATEGY: KIPP Sandbox — BEAR MARKET TEST")
        print("="*70)
        print(f"Initial Capital: ${TOTAL_INITIAL_CAPITAL:.2f}")
        print(f"Rebalancing: Weekly (every 5 trading days)")
        print()

        self.runner = SimpleProductionRunner(use_news=False, n_workers=1)
        all_dates = sorted(price_data[self.tickers[0]].index)
        rebalance_dates = all_dates[::5]

        print(f"  Trading days: {len(all_dates)} | Rebalance points: {len(rebalance_dates)}")
        print()

        for i, rebalance_date in enumerate(rebalance_dates):
            if i % 10 == 0 or i == 0:
                print(f"  Rebalance {i+1}/{len(rebalance_dates)}...")

            # Phase 0: Manage existing positions (trailing stop in bear)
            for ticker, pos in list(self.positions.items()):
                if ticker not in price_data:
                    continue
                data = price_data[ticker].loc[:rebalance_date].copy()
                if len(data) < 2:
                    continue
                current_price = data['Close'].iloc[-1]
                if hasattr(current_price, 'item'):
                    current_price = current_price.item()

                if ticker not in self._position_peaks:
                    self._position_peaks[ticker] = current_price
                if current_price > self._position_peaks[ticker]:
                    self._position_peaks[ticker] = current_price

                # Trailing stop: sell if price drops below stop level
                stop_level = pos['avg_cost'] * self.TRAILING_STOP_INITIAL
                if self._position_peaks[ticker] > pos['avg_cost'] * 1.05:
                    stop_level = max(stop_level, self._position_peaks[ticker] * self.TRAILING_STOP_WIN)

                if current_price < stop_level:
                    shares = pos['shares']
                    sale_value = shares * current_price
                    pnl = sale_value - shares * pos['avg_cost']
                    self.cash += sale_value
                    del self.positions[ticker]
                    self._position_peaks.pop(ticker, None)
                    self._position_dates.pop(ticker, None)
                    self.trades.append({
                        'date': rebalance_date, 'ticker': ticker, 'action': 'SELL',
                        'shares': shares, 'price': current_price,
                        'value': sale_value, 'pnl': pnl, 'reason': 'TRAILING_STOP'
                    })

            # Phase 1: Collect BUY candidates + SELL signals
            candidates = []
            for ticker in self.tickers:
                if ticker not in price_data:
                    continue
                data = price_data[ticker].loc[:rebalance_date].copy()
                if len(data) < 50:
                    continue

                result = self.runner.analyze_ticker(ticker, data=data, as_of_date=rebalance_date)
                if result is None:
                    continue

                sentiment = calculate_price_based_sentiment(data)
                result['news_sentiment'] = sentiment

                current_price = data['Close'].iloc[-1]
                if hasattr(current_price, 'item'):
                    current_price = current_price.item()

                signal = result['signal']
                position_size = result['position_size']
                fused_score = result.get('fused_score', 0.0)

                if signal in ["BUY", "STRONG_BUY"] and abs(sentiment) > 0.1:
                    sigmoid_boost = 1 / (1 + np.exp(-5 * sentiment))
                    position_size *= sigmoid_boost

                # SELL signal for existing positions
                if ticker in self.positions and signal in ["SELL", "STRONG_SELL"]:
                    pos = self.positions[ticker]
                    shares = pos['shares']
                    sale_value = shares * current_price
                    pnl = sale_value - shares * pos['avg_cost']
                    self.cash += sale_value
                    del self.positions[ticker]
                    self._position_peaks.pop(ticker, None)
                    self._position_dates.pop(ticker, None)
                    self.trades.append({
                        'date': rebalance_date, 'ticker': ticker, 'action': 'SELL',
                        'shares': shares, 'price': current_price,
                        'value': sale_value, 'pnl': pnl, 'reason': 'SELL signal'
                    })
                    continue

                # Collect BUY candidates
                if signal in ["BUY", "STRONG_BUY"] and ticker not in self.positions:
                    candidates.append({
                        'ticker': ticker, 'signal': signal,
                        'position_size': min(1.0, position_size),
                        'fused_score': fused_score,
                        'confidence': result.get('confidence', 0.0),
                        'current_price': current_price,
                    })

            # Phase 2: Enter NEW positions — ranked by fused_score
            if candidates and len(self.positions) < self.MAX_POSITIONS:
                candidates.sort(key=lambda x: x['fused_score'], reverse=True)
                open_slots = self.MAX_POSITIONS - len(self.positions)

                for c in candidates[:open_slots]:
                    if c['fused_score'] <= 0:
                        continue
                    alloc = self.cash / max(open_slots, 1) * min(1.0, c['position_size'])
                    alloc = min(alloc, self.cash * 0.3)  # max 30% per position
                    if alloc <= 0:
                        continue

                    shares = alloc / c['current_price']
                    self.positions[c['ticker']] = {
                        'shares': shares,
                        'avg_cost': c['current_price'],
                    }
                    self._position_peaks[c['ticker']] = c['current_price']
                    self._position_dates[c['ticker']] = rebalance_date
                    self.cash -= alloc

                    self.trades.append({
                        'date': rebalance_date, 'ticker': c['ticker'],
                        'action': 'BUY', 'shares': shares,
                        'price': c['current_price'], 'value': alloc, 'pnl': 0,
                        'reason': f"{c['signal']} (fused={c['fused_score']:.2f})"
                    })

            # Mark to market
            value = self.cash
            for ticker, pos in self.positions.items():
                if ticker in price_data:
                    data = price_data[ticker].loc[:rebalance_date]
                    if len(data) > 0:
                        price = data['Close'].iloc[-1]
                        if hasattr(price, 'item'):
                            price = price.item()
                        value += pos['shares'] * price
            self.portfolio_history.append({'date': rebalance_date, 'value': value})

        final_value = self.portfolio_history[-1]['value'] if self.portfolio_history else TOTAL_INITIAL_CAPITAL
        ret = (final_value / TOTAL_INITIAL_CAPITAL - 1) * 100
        buys = sum(1 for t in self.trades if t['action'] == 'BUY')
        sells = sum(1 for t in self.trades if t['action'] == 'SELL')
        wins = sum(1 for t in self.trades if t['action'] == 'SELL' and t['pnl'] > 0)
        total_pnl = sum(t['pnl'] for t in self.trades if t['action'] == 'SELL')

        print(f"\n  💰 Final Cash: R${self.cash:,.2f}")
        print(f"  💰 Final Value: R${final_value:,.2f}")
        print(f"  📈 Return: {ret:+.2f}%")
        print(f"  📊 Trades: {len(self.trades)} ({buys} buys, {sells} sells)")
        print(f"  🎯 Win Rate: {wins/sells*100:.1f}%" if sells else "  🎯 No sells")
        print(f"  💵 Realized P&L: R${total_pnl:+,.2f}")
        return self.portfolio_history


def calculate_metrics(portfolio_history: list, initial_capital: float) -> dict:
    values = [p['value'] for p in portfolio_history]
    returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
    total_return = (values[-1] / initial_capital - 1) * 100 if values else 0
    volatility = np.std(returns) * np.sqrt(52) * 100 if len(returns) > 1 else 0  # weekly
    sharpe = (total_return - 5) / volatility if volatility > 0 else 0
    peak = values[0]
    max_dd = max((peak - v) / peak for v in values) if values else 0
    return {
        'total_return': total_return, 'volatility': volatility,
        'sharpe_ratio': sharpe, 'max_drawdown': max_dd * 100
    }


def download_price_data(tickers: list, start_date: datetime, end_date: datetime) -> dict:
    print("\n" + "="*70)
    print("📥 DOWNLOADING PRICE DATA — BEAR MARKET (COVID 2020)")
    print("="*70)
    print(f"Period: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
    print()
    price_data = {}
    for ticker in tickers:
        print(f"  {ticker}...", end=" ", flush=True)
        try:
            data = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"),
                              end=end_date.strftime("%Y-%m-%d"), progress=False)
            if not data.empty and len(data) >= 50:
                price_data[ticker] = data
                print(f"✅ {len(data)} days")
            else:
                print(f"❌ Only {len(data)} days")
        except Exception as e:
            print(f"❌ {e}")
    print(f"\n✅ Downloaded {len(price_data)}/{len(tickers)} stocks")
    return price_data


def main():
    print("\n" + "="*70)
    print("🐻 BEAR MARKET BACKTEST — COVID CRASH 2020")
    print("="*70)
    print(f"Test Period: {START_DATE.strftime('%Y-%m-%d')} → {END_DATE.strftime('%Y-%m-%d')}")
    print(f"Initial Capital: R$ {TOTAL_INITIAL_CAPITAL:,.2f}")
    print(f"Stocks: {len(TEST_STOCKS)}")
    print()

    price_data = download_price_data(TEST_STOCKS, START_DATE, END_DATE)
    if len(price_data) < 5:
        print("\n❌ Insufficient data")
        return

    # Passive (buy & hold IBOV proxy = equal weight)
    passive = PassiveStrategy(list(price_data.keys()), INITIAL_CAPITAL_PER_STOCK)
    passive_history = passive.run(price_data)

    # Active (KIPP sandbox)
    active = ActiveStrategy(list(price_data.keys()), INITIAL_CAPITAL_PER_STOCK)
    active_history = active.run(price_data)

    passive_metrics = calculate_metrics(passive_history, TOTAL_INITIAL_CAPITAL)
    active_metrics = calculate_metrics(active_history, TOTAL_INITIAL_CAPITAL)

    # Comparison
    print("\n" + "="*70)
    print("🐻 BEAR MARKET — COMPREHENSIVE COMPARISON")
    print("="*70)
    print()
    print(f"  {'Metric':<20} {'Passive':>15} {'Active':>15} {'Winner':>10}")
    print(f"  {'-'*65}")

    w = 'Active' if active_metrics['total_return'] > passive_metrics['total_return'] else 'Passive'
    print(f"  {'Total Return':<20} {passive_metrics['total_return']:>+14.2f}% {active_metrics['total_return']:>+14.2f}% {w:>10}")

    w = 'Active' if active_metrics['sharpe_ratio'] > passive_metrics['sharpe_ratio'] else 'Passive'
    print(f"  {'Sharpe Ratio':<20} {passive_metrics['sharpe_ratio']:>14.2f} {active_metrics['sharpe_ratio']:>14.2f} {w:>10}")

    w = 'Active' if active_metrics['max_drawdown'] < passive_metrics['max_drawdown'] else 'Passive'
    print(f"  {'Max Drawdown':<20} {passive_metrics['max_drawdown']:>13.2f}% {active_metrics['max_drawdown']:>13.2f}% {w:>10}")

    w = 'Active' if active_metrics['volatility'] < passive_metrics['volatility'] else 'Passive'
    print(f"  {'Volatility':<20} {passive_metrics['volatility']:>13.2f}% {active_metrics['volatility']:>13.2f}% {w:>10}")

    print()
    alpha = active_metrics['total_return'] - passive_metrics['total_return']
    if alpha > 0:
        print(f"  🎯 Alpha: {alpha:+.2f}% — ✅ Active OUTPERFORMED in bear market!")
    else:
        print(f"  🎯 Alpha: {alpha:.2f}% — ❌ Active underperformed")
    print()

    # Save
    with open('bear_backtest_results.txt', 'w') as f:
        f.write("🐻 BEAR MARKET BACKTEST — COVID 2020\n")
        f.write("="*70 + "\n\n")
        f.write("PASSIVE STRATEGY (Buy & Hold):\n")
        for k, v in passive_metrics.items():
            f.write(f"  {k}: {v:.2f}\n")
        f.write("\nACTIVE STRATEGY (KIPP Sandbox):\n")
        for k, v in active_metrics.items():
            f.write(f"  {k}: {v:.2f}\n")
        f.write(f"\nAlpha: {alpha:.2f}%\n")
    print("✅ Results saved to bear_backtest_results.txt")
    print("="*70)


if __name__ == '__main__':
    main()
