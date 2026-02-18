#!/usr/bin/env python3
"""
Live Market Monitor - Integrated Signal System
Runs every 10 minutes during market hours.

Features:
- Optimized thresholds from walk-forward validation
- Signal fusion with all indicators
- Smart news caching (newsdata.io budget-aware)
- Targeted recommendations (max 15 stocks)
- Signal drivers explanation
- Position sizing based on confidence
- Entry/exit levels with stop-loss
- Reversal signals to watch
"""

import sys
sys.path.insert(0, '/home/ulluboz/.openclaw/workspace/stock-signals')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path

from production_enhanced import EnhancedProductionRunner
from src.config import get_thresholds, reload_config
from src.news.free_news_client import FreeNewsClient


# ============================================================================
# CONFIGURATION
# ============================================================================

MAX_RECOMMENDATIONS = 15
MIN_CONFIDENCE = 0.50  # Minimum confidence to include
MIN_SCORE = 0.20       # Minimum fused score
NEWS_REFRESH_TOP_N = 5 # Only refresh news for top N movers (API budget)

# Alert output paths
ALERT_FILE = Path('/home/ulluboz/.openclaw/workspace/stock-signals/live_alerts.txt')
ALERT_JSON = Path('/home/ulluboz/.openclaw/workspace/stock-signals/live_alerts.json')


# ============================================================================
# SIGNAL DRIVER ANALYSIS
# ============================================================================

def analyze_signal_drivers(result: dict) -> dict:
    """
    Analyze what's driving the signal.
    Returns dict with main drivers and their contributions.
    """
    drivers = {
        'primary': [],
        'secondary': [],
        'confidence_boosters': [],
        'risk_factors': []
    }
    
    signal = result.get('signal', 'HOLD')
    confidence = result.get('confidence', 0)
    trend = result.get('trend', 'unknown')
    fused_score = result.get('fused_score', 0)
    news_sentiment = result.get('news_sentiment', 0)
    
    # Trend driver
    if trend == 'uptrend':
        drivers['primary'].append({
            'factor': 'Trend',
            'value': 'Uptrend',
            'impact': f'+{confidence*0.4:.0%} confidence',
            'description': 'Price above 50-day MA with bullish momentum'
        })
    elif trend == 'downtrend':
        drivers['primary'].append({
            'factor': 'Trend',
            'value': 'Downtrend',
            'impact': f'-{confidence*0.3:.0%} confidence',
            'description': 'Price below 50-day MA with bearish momentum'
        })
    
    # Fused score driver
    if abs(fused_score) > 0.3:
        drivers['primary'].append({
            'factor': 'Signal Fusion',
            'value': f'{fused_score:+.2f}',
            'impact': 'Strong multi-indicator alignment',
            'description': 'Multiple indicators agree on direction'
        })
    
    # News sentiment driver
    if abs(news_sentiment) > 0.1:
        impact = 'bullish' if news_sentiment > 0 else 'bearish'
        drivers['secondary'].append({
            'factor': 'News Sentiment',
            'value': f'{news_sentiment:+.2f}',
            'impact': impact,
            'description': 'Recent news supports the signal'
        })
    
    # Confidence level
    if confidence > 0.7:
        drivers['confidence_boosters'].append({
            'factor': 'High Confidence',
            'value': f'{confidence:.0%}',
            'impact': 'Strong signal quality',
            'description': 'Multiple factors align with high conviction'
        })
    
    # Risk factors (things that could reverse the signal)
    if abs(news_sentiment) > 0.2 and signal == 'BUY' and news_sentiment < 0:
        drivers['risk_factors'].append({
            'factor': 'Negative News',
            'value': f'{news_sentiment:.2f}',
            'impact': 'Could reverse BUY signal',
            'description': 'Watch for sentiment shift'
        })
    
    if trend == 'uptrend' and confidence < 0.6:
        drivers['risk_factors'].append({
            'factor': 'Weak Trend',
            'value': f'{confidence:.0%}',
            'impact': 'Trend could stall',
            'description': 'Monitor for trend exhaustion'
        })
    
    return drivers


def calculate_entry_exit_levels(result: dict, thresholds) -> dict:
    """
    Calculate entry, target, and stop-loss levels.
    """
    price = result.get('price', 0)
    signal = result.get('signal', 'HOLD')
    confidence = result.get('confidence', 0)
    
    if signal == 'BUY':
        # Entry: current price
        entry = price
        
        # Stop loss: based on threshold and volatility
        stop_loss_pct = thresholds.stop_loss
        stop_loss = price * (1 - stop_loss_pct)
        
        # Target: 2:1 risk-reward minimum
        risk = price - stop_loss
        target_1 = price + risk * 2
        target_2 = price + risk * 3
        
        return {
            'entry': entry,
            'stop_loss': stop_loss,
            'stop_loss_pct': stop_loss_pct,
            'target_1': target_1,
            'target_2': target_2,
            'risk_reward': 2.0
        }
    
    elif signal == 'SELL':
        # For sells, we're exiting or shorting
        entry = price
        stop_loss = price * (1 + thresholds.stop_loss)
        
        return {
            'exit_price': entry,
            'stop_loss': stop_loss,
            'stop_loss_pct': thresholds.stop_loss
        }
    
    return {}


def calculate_position_allocation(result: dict, thresholds) -> dict:
    """
    Calculate position size based on confidence and risk.
    Uses Kelly Criterion principles with confidence scaling.
    """
    confidence = result.get('confidence', 0)
    fused_score = result.get('fused_score', 0)
    signal = result.get('signal', 'HOLD')
    
    if signal not in ['BUY', 'SELL']:
        return {'allocation_pct': 0, 'reason': 'No actionable signal'}
    
    # Base allocation (Kelly fraction)
    base_allocation = 0.20  # 20% base
    
    # Scale by confidence
    confidence_multiplier = confidence / 0.6  # Normalize to 60% as baseline
    
    # Scale by fused score strength
    score_multiplier = min(abs(fused_score) / 0.3, 1.5)  # Cap at 1.5x
    
    # Calculate final allocation
    allocation = base_allocation * confidence_multiplier * score_multiplier
    
    # Apply limits
    allocation = max(0.10, min(allocation, 0.40))  # 10-40% range
    
    return {
        'allocation_pct': allocation,
        'confidence_multiplier': confidence_multiplier,
        'score_multiplier': score_multiplier,
        'reason': f'Base {base_allocation:.0%} × conf {confidence_multiplier:.2f} × score {score_multiplier:.2f}'
    }


def get_reversal_signals(result: dict, price_data: dict) -> list:
    """
    Identify signals that could reverse the current recommendation.
    """
    reversal_signals = []
    signal = result.get('signal', 'HOLD')
    ticker = result.get('ticker', '')
    confidence = result.get('confidence', 0)
    
    if signal == 'BUY':
        # What would reverse a BUY?
        reversal_signals.append({
            'signal': 'Trend reversal',
            'trigger': 'Price drops below 50-day MA',
            'action': 'Consider tightening stop-loss or exiting'
        })
        
        if confidence < 0.7:
            reversal_signals.append({
                'signal': 'Low conviction breakdown',
                'trigger': 'Price drops 3% from entry',
                'action': 'Exit if momentum weakens'
            })
        
        reversal_signals.append({
            'signal': 'Negative news catalyst',
            'trigger': 'Major negative news event',
            'action': 'Re-evaluate immediately'
        })
    
    elif signal == 'SELL':
        reversal_signals.append({
            'signal': 'Trend reversal',
            'trigger': 'Price breaks above 50-day MA with volume',
            'action': 'Cover short or re-evaluate'
        })
        
        reversal_signals.append({
            'signal': 'Positive news catalyst',
            'trigger': 'Major positive news event',
            'action': 'Monitor for sentiment shift'
        })
    
    return reversal_signals


# ============================================================================
# FORMATTING
# ============================================================================

def format_recommendation(result: dict, drivers: dict, levels: dict, 
                         allocation: dict, reversals: list) -> str:
    """Format a single recommendation for output."""
    lines = []
    
    ticker = result['ticker']
    signal = result['signal']
    price = result['price']
    confidence = result['confidence']
    
    # Header
    emoji = '🟢' if signal == 'BUY' else '🔴' if signal == 'SELL' else '⚪'
    lines.append(f"\n{emoji} **{ticker}** - {signal}")
    lines.append(f"   Price: R$ {price:.2f} | Confidence: {confidence:.0%}")
    
    # Position allocation
    if allocation.get('allocation_pct', 0) > 0:
        lines.append(f"\n   📊 **Position Size:** {allocation['allocation_pct']:.0%}")
        if allocation.get('reason'):
            lines.append(f"      _{allocation['reason']}_")
    
    # Entry/Exit levels
    if levels:
        lines.append(f"\n   📍 **Levels:**")
        if signal == 'BUY':
            lines.append(f"      Entry: R$ {levels.get('entry', price):.2f}")
            lines.append(f"      Stop Loss: R$ {levels.get('stop_loss', 0):.2f} ({levels.get('stop_loss_pct', 0):.0%})")
            lines.append(f"      Target 1: R$ {levels.get('target_1', 0):.2f} (2:1 R:R)")
            lines.append(f"      Target 2: R$ {levels.get('target_2', 0):.2f} (3:1 R:R)")
        elif signal == 'SELL':
            lines.append(f"      Exit: R$ {levels.get('exit_price', price):.2f}")
            lines.append(f"      Stop Loss: R$ {levels.get('stop_loss', 0):.2f}")
    
    # Signal drivers
    if drivers.get('primary'):
        lines.append(f"\n   🔍 **Why this signal:**")
        for driver in drivers['primary']:
            lines.append(f"      • {driver['factor']}: {driver['value']} - {driver['impact']}")
    
    if drivers.get('secondary'):
        for driver in drivers['secondary']:
            lines.append(f"      • {driver['factor']}: {driver['value']} ({driver['impact']})")
    
    # Confidence boosters
    if drivers.get('confidence_boosters'):
        lines.append(f"\n   ✅ **Confidence boosters:**")
        for booster in drivers['confidence_boosters']:
            lines.append(f"      • {booster['factor']}: {booster['value']}")
    
    # Risk factors
    if drivers.get('risk_factors'):
        lines.append(f"\n   ⚠️ **Risk factors:**")
        for risk in drivers['risk_factors']:
            lines.append(f"      • {risk['factor']}: {risk['impact']}")
    
    # Reversal signals to watch
    if reversals:
        lines.append(f"\n   👀 **Watch for:**")
        for rev in reversals[:2]:  # Max 2 reversal signals
            lines.append(f"      • {rev['trigger']} → {rev['action']}")
    
    lines.append("")
    
    return "\n".join(lines)


def format_full_alert(recommendations: list, stats: dict) -> str:
    """Format the complete alert message."""
    lines = []
    
    # Header
    lines.append("=" * 70)
    lines.append(f"🎯 LIVE MARKET SIGNALS")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    
    # Market summary
    lines.append(f"\n📊 **Market Summary:**")
    lines.append(f"   Analyzed: {stats.get('total_analyzed', 0)} stocks")
    lines.append(f"   BUY signals: {stats.get('buy_count', 0)}")
    lines.append(f"   SELL signals: {stats.get('sell_count', 0)}")
    lines.append(f"   News cache: {stats.get('news_cached', 0)} cached, {stats.get('news_fresh', 0)} fresh")
    
    # Recommendations
    lines.append("\n" + "=" * 70)
    lines.append("🚨 **TOP RECOMMENDATIONS** (sorted by conviction)")
    lines.append("=" * 70)
    
    for rec in recommendations:
        lines.append(format_recommendation(
            rec['result'],
            rec['drivers'],
            rec['levels'],
            rec['allocation'],
            rec['reversals']
        ))
    
    # Footer
    lines.append("=" * 70)
    lines.append("⏰ Next update in 10 minutes")
    lines.append("=" * 70)
    
    return "\n".join(lines)


# ============================================================================
# MAIN MONITORING FUNCTION
# ============================================================================

def run_monitor():
    """
    Main monitoring function.
    Called every 10 minutes by cron.
    """
    print("\n" + "=" * 70)
    print(f"🎯 LIVE MARKET MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Reload config to get latest thresholds
    reload_config()
    thresholds = get_thresholds('default')
    
    print(f"\n⚙️ Using thresholds:")
    print(f"   Buy: conf ≥ {thresholds.buy_confidence:.0%}, score ≥ {thresholds.min_score:.0%}")
    print(f"   Sell: conf ≥ {thresholds.sell_confidence:.0%}")
    print(f"   Stop Loss: {thresholds.stop_loss:.0%}")
    
    # Initialize runner
    runner = EnhancedProductionRunner(
        use_news=True,
        use_fusion=True,
        use_regime=False,  # Use default thresholds
        n_workers=4
    )
    
    # Run analysis
    print(f"\n📊 Analyzing market with signal fusion...")
    results = runner.run(parallel=True)
    
    if not results:
        print("❌ No results from analysis")
        return
    
    # Get stats
    total = len(results)
    buy_signals = [r for r in results if r['signal'] == 'BUY']
    sell_signals = [r for r in results if r['signal'] == 'SELL']
    
    print(f"   Total analyzed: {total}")
    print(f"   BUY signals: {len(buy_signals)}")
    print(f"   SELL signals: {len(sell_signals)}")
    
    # Smart news refresh for top movers
    news_fresh = 0
    if runner.news_client:
        # Sort by confidence and get top movers
        sorted_results = sorted(results, key=lambda x: x.get('confidence', 0), reverse=True)
        top_movers = [r['ticker'] for r in sorted_results[:NEWS_REFRESH_TOP_N]]
        
        if top_movers:
            print(f"\n🔄 Refreshing news for top {len(top_movers)} movers...")
            fresh_sentiments = runner.news_client.refresh_sentiment_batch(top_movers)
            news_fresh = len(fresh_sentiments)
            
            # Update results
            for result in results:
                if result['ticker'] in fresh_sentiments:
                    result['news_sentiment'] = fresh_sentiments[result['ticker']]
    
    # Filter and sort for recommendations
    actionable = [r for r in results if r['signal'] in ['BUY', 'SELL'] 
                  and r.get('confidence', 0) >= MIN_CONFIDENCE
                  and abs(r.get('fused_score', 0)) >= MIN_SCORE]
    
    # Sort by conviction (confidence * score)
    actionable.sort(key=lambda x: x.get('confidence', 0) * abs(x.get('fused_score', 0)), reverse=True)
    
    # Take top N
    top_recommendations = actionable[:MAX_RECOMMENDATIONS]
    
    print(f"\n🎯 Top {len(top_recommendations)} recommendations:")
    
    # Build detailed recommendations
    detailed_recommendations = []
    for result in top_recommendations:
        # Analyze drivers
        drivers = analyze_signal_drivers(result)
        
        # Calculate levels
        levels = calculate_entry_exit_levels(result, thresholds)
        
        # Calculate allocation
        allocation = calculate_position_allocation(result, thresholds)
        
        # Get reversal signals
        reversals = get_reversal_signals(result, {})
        
        detailed_recommendations.append({
            'result': result,
            'drivers': drivers,
            'levels': levels,
            'allocation': allocation,
            'reversals': reversals
        })
        
        # Print summary
        signal = result['signal']
        ticker = result['ticker']
        conf = result.get('confidence', 0)
        alloc = allocation.get('allocation_pct', 0)
        print(f"   {ticker}: {signal} ({conf:.0%} conf, {alloc:.0%} allocation)")
    
    # Build stats
    stats = {
        'total_analyzed': total,
        'buy_count': len(buy_signals),
        'sell_count': len(sell_signals),
        'news_cached': len([r for r in results if r.get('news_sentiment', 0) != 0]) - news_fresh,
        'news_fresh': news_fresh
    }
    
    # Format and save alert
    alert_text = format_full_alert(detailed_recommendations, stats)
    
    # Save to file
    ALERT_FILE.write_text(alert_text)
    
    # Save JSON for programmatic access
    alert_json = {
        'generated_at': datetime.now().isoformat(),
        'stats': stats,
        'thresholds': {
            'buy_confidence': thresholds.buy_confidence,
            'sell_confidence': thresholds.sell_confidence,
            'min_score': thresholds.min_score,
            'stop_loss': thresholds.stop_loss
        },
        'recommendations': [
            {
                'ticker': r['result']['ticker'],
                'signal': r['result']['signal'],
                'price': r['result']['price'],
                'confidence': r['result']['confidence'],
                'allocation_pct': r['allocation'].get('allocation_pct', 0),
                'entry': r['levels'].get('entry'),
                'stop_loss': r['levels'].get('stop_loss'),
                'target_1': r['levels'].get('target_1'),
                'drivers': r['drivers'],
                'reversals': r['reversals']
            }
            for r in detailed_recommendations
        ]
    }
    
    ALERT_JSON.write_text(json.dumps(alert_json, indent=2, default=str))
    
    # Print alert
    print("\n" + alert_text)
    
    print(f"\n✅ Alert saved to:")
    print(f"   {ALERT_FILE}")
    print(f"   {ALERT_JSON}")
    
    # Log API usage if news client
    if runner.news_client:
        try:
            cache_stats = runner.news_client.get_cache_stats()
            print(f"\n💾 News cache: {cache_stats['total_entries']} entries ({cache_stats['fresh']} fresh)")
        except:
            pass
    
    return detailed_recommendations


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_monitor()
