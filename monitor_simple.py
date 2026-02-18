#!/usr/bin/env python3
"""
Simplified Live Market Monitor
Fast, lightweight, no multiprocessing issues.
"""

import sys
sys.path.insert(0, '/home/ulluboz/.openclaw/workspace/stock-signals')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path

# Simple trend detection (no ML, no multiprocessing)
def detect_trend(data):
    """Simple trend detection using MAs"""
    if len(data) < 200:
        return 'unknown', 0
    
    close = data['Close']
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    
    current = close.iloc[-1]
    ma20_val = ma20.iloc[-1]
    ma50_val = ma50.iloc[-1]
    ma200_val = ma200.iloc[-1]
    
    # Trend strength
    if current > ma20_val > ma50_val > ma200_val:
        return 'strong_uptrend', 0.9
    elif current > ma50_val > ma200_val:
        return 'uptrend', 0.7
    elif current > ma50_val:
        return 'weak_uptrend', 0.5
    elif current < ma20_val < ma50_val < ma200_val:
        return 'strong_downtrend', 0.9
    elif current < ma50_val < ma200_val:
        return 'downtrend', 0.7
    elif current < ma50_val:
        return 'weak_downtrend', 0.5
    else:
        return 'consolidation', 0.3

def calculate_rsi(close, period=14):
    """Calculate RSI"""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else 50

def calculate_volume_signal(data):
    """Volume analysis"""
    if len(data) < 20:
        return 0
    
    vol = data['Volume']
    avg_vol = vol.rolling(20).mean().iloc[-1]
    current_vol = vol.iloc[-1]
    
    if avg_vol == 0:
        return 0
    
    ratio = current_vol / avg_vol
    
    if ratio > 2.0:
        return 0.3  # High volume
    elif ratio > 1.5:
        return 0.2
    elif ratio > 1.2:
        return 0.1
    else:
        return 0

def analyze_ticker(ticker):
    """Analyze a single ticker"""
    try:
        data = yf.download(ticker, period='1y', progress=False)
        
        if data.empty or len(data) < 200:
            return None
        
        close = data['Close']
        
        # Trend
        trend, trend_strength = detect_trend(data)
        
        # RSI
        rsi = calculate_rsi(close)
        
        # Volume
        vol_signal = calculate_volume_signal(data)
        
        # Price change
        price = float(close.iloc[-1])
        prev_close = float(close.iloc[-2]) if len(close) > 1 else price
        change_pct = ((price / prev_close) - 1) * 100
        
        # Generate signal
        signal = 'HOLD'
        confidence = 0.5
        
        if trend in ['strong_uptrend', 'uptrend']:
            signal = 'BUY'
            confidence = trend_strength
            # RSI boost/reduce
            if rsi < 30:
                confidence += 0.1  # Oversold bounce
            elif rsi > 70:
                confidence -= 0.1  # Overbought
        elif trend in ['strong_downtrend', 'downtrend']:
            signal = 'SELL'
            confidence = trend_strength
            if rsi > 70:
                confidence += 0.1
            elif rsi < 30:
                confidence -= 0.1
        
        # Volume confirmation
        confidence += vol_signal * 0.1
        
        # Clamp confidence
        confidence = max(0.1, min(1.0, confidence))
        
        return {
            'ticker': ticker.replace('.SA', ''),
            'price': price,
            'change_pct': change_pct,
            'signal': signal,
            'confidence': confidence,
            'trend': trend,
            'rsi': rsi,
            'volume_ratio': vol_signal
        }
    except Exception as e:
        return None

def main():
    print("\n" + "="*70)
    print(f"🎯 QUICK MARKET SCAN - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Load tickers
    with open('/home/ulluboz/.openclaw/workspace/stock-signals/data/validated_tickers.json', 'r') as f:
        ticker_data = json.load(f)
    
    all_tickers = ticker_data.get('all_tickers', [])[:50]  # Limit to top 50 for speed
    print(f"\n📊 Scanning {len(all_tickers)} stocks...")
    
    results = []
    for ticker in all_tickers:
        ticker_sa = f"{ticker}.SA" if not ticker.endswith('.SA') else ticker
        result = analyze_ticker(ticker_sa)
        if result:
            results.append(result)
    
    # Filter for actionable signals
    buy_signals = [r for r in results if r['signal'] == 'BUY' and r['confidence'] >= 0.6]
    sell_signals = [r for r in results if r['signal'] == 'SELL' and r['confidence'] >= 0.6]
    
    # Sort by confidence
    buy_signals.sort(key=lambda x: x['confidence'], reverse=True)
    sell_signals.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Output
    output = []
    output.append("\n" + "="*70)
    output.append(f"🎯 MARKET SIGNALS - {datetime.now().strftime('%H:%M:%S')}")
    output.append("="*70)
    output.append(f"\n📊 Scanned: {len(results)} | BUY: {len(buy_signals)} | SELL: {len(sell_signals)}")
    
    if buy_signals:
        output.append("\n🟢 TOP BUY SIGNALS:")
        for i, r in enumerate(buy_signals[:10], 1):
            output.append(f"  {i}. {r['ticker']:8s} R${r['price']:>8.2f} ({r['change_pct']:>+5.1f}%) conf:{r['confidence']:.0%} RSI:{r['rsi']:.0f}")
    
    if sell_signals:
        output.append("\n🔴 TOP SELL SIGNALS:")
        for i, r in enumerate(sell_signals[:5], 1):
            output.append(f"  {i}. {r['ticker']:8s} R${r['price']:>8.2f} ({r['change_pct']:>+5.1f}%) conf:{r['confidence']:.0%} RSI:{r['rsi']:.0f}")
    
    output.append("\n" + "="*70)
    output.append("⏰ Next scan in 10 minutes")
    output.append("="*70)
    
    alert_text = "\n".join(output)
    print(alert_text)
    
    # Save
    Path('/home/ulluboz/.openclaw/workspace/stock-signals/live_alerts.txt').write_text(alert_text)
    
    with open('/home/ulluboz/.openclaw/workspace/stock-signals/live_alerts.json', 'w') as f:
        json.dump({
            'generated_at': datetime.now().isoformat(),
            'buy_signals': buy_signals[:10],
            'sell_signals': sell_signals[:5]
        }, f, indent=2, default=str)
    
    print("\n✅ Scan complete")

if __name__ == "__main__":
    main()
