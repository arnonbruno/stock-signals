#!/usr/bin/env python3
"""
Daily Market Scanner with Signal Reporting
Runs via cron, sends alerts to Telegram with full reasoning.
"""

import sys
sys.path.insert(0, '.')

from datetime import datetime
from production_simple import SimpleProductionRunner
from src.strategy.regime_detection import get_regime_detector, get_adaptive_params
import yfinance as yf

def get_market_regime():
    """Get current market regime from IBOV."""
    try:
        ibov = yf.download('^BVSP', period='1y', progress=False)
        if len(ibov) < 200:
            return {'regime': 'sideways', 'strength': 0.5}
        
        detector = get_regime_detector()
        regime_info = detector.detect_regime(ibov)
        return regime_info
    except:
        return {'regime': 'sideways', 'strength': 0.5}

def format_signal_report(signals, regime_info):
    """Format signals with full reasoning."""
    
    if not signals:
        return "No actionable signals today."
    
    report = []
    report.append("="*70)
    report.append(f"📊 DAILY MARKET SCAN - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("="*70)
    report.append("")
    
    # Regime info
    regime = regime_info.get('regime', 'sideways').upper()
    strength = regime_info.get('strength', 0.5)
    report.append(f"🎯 MARKET REGIME: {regime} (strength: {strength:.0%})")
    
    params = get_adaptive_params().get_parameters(regime, strength)
    report.append(f"   Confidence threshold: {params['confidence_threshold']:.0%}")
    report.append(f"   Max position size: {params['max_position_size']:.0%}")
    report.append(f"   Cash buffer: {params['max_cash_pct']:.0%}")
    report.append("")
    
    # Top signals
    report.append("="*70)
    report.append(f"🚨 TOP SIGNALS ({len(signals)} actionable)")
    report.append("="*70)
    report.append("")
    
    for i, signal in enumerate(signals[:10], 1):
        ticker = signal['ticker']
        action = signal['signal']
        price = signal['price']
        trend = signal['trend']
        confidence = signal.get('confidence', 0)
        position = signal.get('position_size', 0)
        sentiment = signal.get('news_sentiment', 0)
        features = signal.get('features', {})
        
        # Signal reasoning
        reasons = []
        
        # Trend reasoning
        if trend == 'uptrend':
            reasons.append(f"✅ UPTREND detected ({confidence:.0%} confidence)")
        elif trend == 'downtrend':
            reasons.append(f"❌ DOWNTREND detected ({confidence:.0%} confidence)")
        else:
            reasons.append(f"⏸️ SIDEWAYS/consolidating ({confidence:.0%} confidence)")
        
        # Volume reasoning
        if features.get('unusual_volume'):
            reasons.append(f"📈 UNUSUAL VOLUME (+5% confidence boost)")
        
        # Volatility reasoning
        vol_regime = features.get('volatility_regime', 'medium')
        if vol_regime == 'high':
            reasons.append(f"⚡ HIGH VOLATILITY (-10% confidence adjustment)")
        
        # Sentiment reasoning
        if abs(sentiment) > 0.15:
            if sentiment > 0:
                reasons.append(f"📰 POSITIVE NEWS sentiment ({sentiment:+.2f})")
            else:
                reasons.append(f"📰 NEGATIVE NEWS sentiment ({sentiment:+.2f})")
        
        # Format output
        emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "⚪"
        report.append(f"{i}️⃣  {ticker} - {action} {emoji} @ R${price:.2f}")
        report.append(f"    Position size: {position:.0%}")
        report.append(f"    Reasoning:")
        for reason in reasons:
            report.append(f"       {reason}")
        report.append("")
    
    # Summary
    buy_count = sum(1 for s in signals if s['signal'] == 'BUY')
    sell_count = sum(1 for s in signals if s['signal'] == 'SELL')
    
    report.append("="*70)
    report.append(f"📊 SUMMARY: {buy_count} BUY | {sell_count} SELL | {len(signals)-buy_count-sell_count} HOLD")
    report.append("="*70)
    
    return "\n".join(report)

def main():
    print("🔍 Starting daily market scan...")
    
    # Get market regime
    regime_info = get_market_regime()
    print(f"   Market regime: {regime_info['regime'].upper()} (strength: {regime_info['strength']:.0%})")
    
    # Run analysis
    runner = SimpleProductionRunner(use_news=False)  # Fast scan without news
    results = runner.run()
    
    if not results:
        print("❌ No results from analysis")
        return
    
    # Filter actionable signals
    actionable = [r for r in results if r['signal'] in ['BUY', 'SELL']]
    actionable_sorted = sorted(actionable, key=lambda x: abs(x.get('conviction', 0)), reverse=True)
    
    # Generate report
    report = format_signal_report(actionable_sorted, regime_info)
    
    # Save report
    with open('daily_scan_report.txt', 'w') as f:
        f.write(report)
    
    # Print to stdout
    print(report)
    print("\n✅ Report saved to daily_scan_report.txt")
    
    # Return for Telegram integration
    return report

if __name__ == "__main__":
    main()
