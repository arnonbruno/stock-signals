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

MAX_RECOMMENDATIONS = 10
MIN_CONFIDENCE = 0.50  # Minimum confidence to include
MIN_SCORE = 0.20       # Minimum fused score

# Two-pass architecture settings
NEWS_CANDIDATES = 10   # How many stocks to fetch news for in Pass 2 (was 30)
NEWS_REFRESH_TOP_N = 5 # Legacy: refresh news for top N (not used in two-pass)

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
        'risk_factors': [],
        'reversal_signals': []  # NEW: What could flip this signal
    }
    
    signal = result.get('signal', 'HOLD')
    confidence = result.get('confidence', 0)
    trend = result.get('trend', 'unknown')
    fused_score = result.get('fused_score', 0)
    news_sentiment = result.get('news_sentiment', 0)
    features = result.get('features', {})
    
    # === TREND DRIVER (more descriptive) ===
    if trend == 'uptrend':
        trend_desc = "Price above 50-day MA, forming higher lows"
        if confidence > 0.7:
            trend_desc += ", strong buying pressure"
        drivers['primary'].append({
            'factor': 'Trend',
            'value': 'Bullish Uptrend',
            'impact': f'+{confidence*0.4:.0%} confidence',
            'description': trend_desc
        })
    elif trend == 'downtrend':
        trend_desc = "Price below 50-day MA, forming lower highs"
        if confidence > 0.7:
            trend_desc += ", strong selling pressure"
        drivers['primary'].append({
            'factor': 'Trend',
            'value': 'Bearish Downtrend',
            'impact': f'{confidence*0.3:.0%} confidence',
            'description': trend_desc
        })
    elif trend == 'consolidation':
        drivers['primary'].append({
            'factor': 'Trend',
            'value': 'Sideways/Range-bound',
            'impact': 'Neutral bias',
            'description': 'Price oscillating between support/resistance'
        })
    
    # === FUSION SCORE DRIVER (breakdown what indicators agree) ===
    if abs(fused_score) > 0.4:
        fusion_desc = ""
        if fused_score > 0:
            fusion_desc = "RSI not overbought + MACD bullish + Volume supporting"
            if fused_score > 0.5:
                fusion_desc += " + Multiple timeframes aligned"
        else:
            fusion_desc = "RSI not oversold + MACD bearish + Volume on sells"
            if fused_score < -0.5:
                fusion_desc += " + Multiple timeframes aligned"
        
        drivers['primary'].append({
            'factor': 'Signal Fusion',
            'value': f'{fused_score:+.2f}',
            'impact': 'Strong' if abs(fused_score) > 0.5 else 'Moderate',
            'description': fusion_desc
        })
    elif abs(fused_score) > 0.2:
        drivers['secondary'].append({
            'factor': 'Signal Fusion',
            'value': f'{fused_score:+.2f}',
            'impact': 'Weak alignment',
            'description': 'Some indicators agree, others mixed'
        })
    
    # === VOLUME ANALYSIS ===
    volume_momentum = features.get('volume_momentum', 1.0)
    unusual_volume = features.get('unusual_volume', False)
    
    if unusual_volume:
        drivers['secondary'].append({
            'factor': 'Volume',
            'value': 'Unusual spike',
            'impact': 'Confirms conviction',
            'description': 'Trading volume 2x+ above average - strong participation'
        })
    elif volume_momentum > 1.2:
        drivers['secondary'].append({
            'factor': 'Volume',
            'value': 'Above average',
            'impact': 'Supports move',
            'description': f'Volume {volume_momentum:.1f}x normal - good participation'
        })
    
    # === VOLATILITY REGIME ===
    vol_regime = features.get('volatility_regime', 'medium')
    if vol_regime == 'high':
        drivers['secondary'].append({
            'factor': 'Volatility',
            'value': 'High',
            'impact': '⚠️ Increased risk',
            'description': 'Larger than normal price swings - use smaller position'
        })
    
    # === NEWS SENTIMENT ===
    if abs(news_sentiment) > 0.1:
        if news_sentiment > 0:
            drivers['secondary'].append({
                'factor': 'News Sentiment',
                'value': f'{news_sentiment:+.2f}',
                'impact': 'Bullish catalyst',
                'description': 'Recent news positive - supports buying pressure'
            })
        else:
            drivers['secondary'].append({
                'factor': 'News Sentiment',
                'value': f'{news_sentiment:+.2f}',
                'impact': 'Bearish overhang',
                'description': 'Recent news negative - supports selling pressure'
            })
    
    # === CONFIDENCE LEVEL ===
    if confidence > 0.75:
        drivers['confidence_boosters'].append({
            'factor': 'High Confidence',
            'value': f'{confidence:.0%}',
            'impact': 'Strong signal quality',
            'description': 'Multiple factors strongly aligned'
        })
    
    # === RISK FACTORS ===
    if vol_regime == 'high' and signal == 'BUY':
        drivers['risk_factors'].append({
            'factor': 'High Volatility',
            'value': vol_regime,
            'impact': 'Wider stops needed',
            'description': 'Price may swing 5%+ in a day'
        })
    
    # === REVERSAL SIGNALS (What could flip this recommendation) ===
    if signal == 'BUY':
        # What would make us exit/sell?
        drivers['reversal_signals'].append({
            'trigger': 'Price drops below 50-day MA',
            'action': 'Exit or tighten stop-loss',
            'probability': 'Medium'
        })
        
        if fused_score > 0.4:
            drivers['reversal_signals'].append({
                'trigger': 'RSI breaks above 70 (overbought)',
                'action': 'Take partial profits',
                'probability': 'Medium'
            })
        
        if vol_regime == 'high':
            drivers['reversal_signals'].append({
                'trigger': 'Price drops 5%+ from entry in single day',
                'action': 'Exit immediately - momentum broken',
                'probability': 'High'
            })
        
        if news_sentiment < 0:
            drivers['reversal_signals'].append({
                'trigger': 'Major negative news event',
                'action': 'Re-evaluate thesis immediately',
                'probability': 'Low but high impact'
            })
    
    elif signal == 'SELL':
        drivers['reversal_signals'].append({
            'trigger': 'Price breaks above 50-day MA with volume',
            'action': 'Cover short / consider reversal',
            'probability': 'Medium'
        })
        
        if fused_score < -0.4:
            drivers['reversal_signals'].append({
                'trigger': 'RSI drops below 30 (oversold)',
                'action': 'Bounce likely - tighten stops',
                'probability': 'Medium'
            })
        
        drivers['reversal_signals'].append({
            'trigger': 'Major positive news catalyst',
            'action': 'Exit position - sentiment shift',
            'probability': 'Low but high impact'
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
                         allocation: dict, reversals: list, index: int = 1,
                         news_cache=None) -> str:
    """Format a single recommendation in compact Telegram-friendly format."""
    lines = []
    
    ticker = result['ticker']
    signal = result['signal']
    price = result['price']
    confidence = result.get('confidence', 0)
    trend = result.get('trend', 'unknown')
    fused_score = result.get('fused_score', 0)
    news_sentiment = result.get('news_sentiment', 0)
    
    # Number emoji
    number_emojis = ['1️⃣', '2️⃣', '3️⃣', '4️⃣', '5️⃣', '6️⃣', '7️⃣', '8️⃣', '9️⃣', '🔟']
    num_emoji = number_emojis[min(index - 1, 9)]
    
    # Signal emoji
    signal_emoji = '🟢' if signal == 'BUY' else '🔴'
    
    # Conviction label
    if confidence >= 0.8:
        conviction_label = "VERY HIGH"
    elif confidence >= 0.7:
        conviction_label = "HIGH"
    elif confidence >= 0.6:
        conviction_label = "MEDIUM-HIGH"
    else:
        conviction_label = "MEDIUM"
    
    lines.append(f"\n{num_emoji}  {ticker} - {signal} {signal_emoji}")
    lines.append(f"    Price: R${price:.2f}")
    
    # Drivers section
    lines.append(f"    └─ Drivers:")
    
    # Trend
    lines.append(f"       • Trend: {trend.upper()} ({confidence:.0%} confidence)")
    
    # Fusion score (real indicator)
    if abs(fused_score) > 0.3:
        fusion_label = "Strong" if abs(fused_score) > 0.5 else "Moderate"
        direction = "bullish" if fused_score > 0 else "bearish"
        lines.append(f"       • Fusion: {fusion_label} {direction} alignment ({fused_score:+.2f})")
    
    # News sentiment with headlines
    lines.append(f"       • News: {news_sentiment:+.2f} sentiment")
    
    # Get article headlines from cache if available
    cached_articles = []
    if news_cache and hasattr(news_cache, 'get_full'):
        cached = news_cache.get_full(ticker)
        if cached and cached.get('articles'):
            cached_articles = cached['articles']
    
    if abs(news_sentiment) > 0.10:
        if news_sentiment > 0.10:
            lines.append(f"         ✅ Positive news boost (+{news_sentiment*0.20:.0%} position)")
        else:
            lines.append(f"         ⚠️ Negative news headwind ({news_sentiment*0.15:.0%} position)")
        
        # Show top headlines
        if cached_articles:
            lines.append(f"         📰 Headlines:")
            for article in cached_articles[:2]:
                title = article.get('title', '')[:55]
                if title:
                    source = article.get('source_publication', '')
                    lines.append(f"            • {title}... ({source})")
    else:
        lines.append(f"         😐 Neutral news (no impact)")
        # Still show headlines if available
        if cached_articles:
            lines.append(f"         📰 Recent:")
            for article in cached_articles[:1]:
                title = article.get('title', '')[:55]
                if title:
                    lines.append(f"            • {title}...")
    
    # Position size
    pos_pct = allocation.get('allocation_pct', 0)
    lines.append(f"       • Position: {pos_pct:.0%} ({conviction_label} conviction)")
    
    # Entry/Stop/Target for BUY signals
    if signal == 'BUY' and levels:
        lines.append(f"    └─ Levels:")
        lines.append(f"       Entry: R${levels.get('entry', price):.2f} | Stop: R${levels.get('stop_loss', 0):.2f}")
        lines.append(f"       Targets: R${levels.get('target_1', 0):.2f} (2:1) | R${levels.get('target_2', 0):.2f} (3:1)")
    
    # What to watch for (reversal signals)
    reversal_signals = drivers.get('reversal_signals', [])
    if reversal_signals:
        lines.append(f"    └─ Watch for:")
        for rev in reversal_signals[:2]:
            lines.append(f"       ⚠️ {rev['trigger']} → {rev['action']}")
    
    return "\n".join(lines)


def format_full_alert(recommendations: list, stats: dict, news_cache=None) -> str:
    """Format the complete alert message in compact Telegram-friendly format."""
    lines = []
    
    # Header
    lines.append("=" * 70)
    lines.append(f"🚨 TOP TRADING OPPORTUNITIES")
    lines.append(f"Generated: {datetime.now().strftime('%H:%M:%S')}")
    lines.append("=" * 70)
    
    # Recommendations
    for i, rec in enumerate(recommendations, 1):
        lines.append(format_recommendation(
            rec['result'],
            rec['drivers'],
            rec['levels'],
            rec['allocation'],
            rec['reversals'],
            index=i,
            news_cache=news_cache
        ))
    
    # Summary
    buy_count = sum(1 for r in recommendations if r['result']['signal'] == 'BUY')
    sell_count = sum(1 for r in recommendations if r['result']['signal'] == 'SELL')
    lines.append("=" * 70)
    lines.append(f"📊 Summary: {buy_count} BUY signals | {sell_count} SELL signals")
    lines.append("=" * 70)
    
    return "\n".join(lines)


# ============================================================================
# MAIN MONITORING FUNCTION
# ============================================================================

def run_monitor():
    """
    Two-pass monitoring function for efficient news processing.
    
    PASS 1: Technical screening (fast, ~1 min for 150 stocks)
    - Run analysis WITHOUT news
    - Select top 30 candidates by preliminary fusion score
    
    PASS 2: News enhancement (slower, but only 30 stocks)
    - Fetch news + translate Portuguese → English
    - Recalculate fusion with sentiment
    - Re-rank and output top 10
    
    Total time: ~1 min + ~10 min = 11 min (vs 5+ hours for single-pass)
    """
    print("\n" + "=" * 70)
    print(f"🎯 LIVE MARKET MONITOR (Two-Pass) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Reload config to get latest thresholds
    reload_config()
    thresholds = get_thresholds('default')
    
    print(f"\n⚙️ Using thresholds:")
    print(f"   Buy: conf ≥ {thresholds.buy_confidence:.0%}, score ≥ {thresholds.min_score:.0%}")
    print(f"   Sell: conf ≥ {thresholds.sell_confidence:.0%}")
    print(f"   Stop Loss: {thresholds.stop_loss:.0%}")
    
    # ========================================================================
    # PASS 1: Technical Screening (Fast)
    # ========================================================================
    print("\n" + "=" * 70)
    print("📊 PASS 1: Technical Screening (no news)")
    print("=" * 70)
    
    runner_pass1 = EnhancedProductionRunner(
        use_news=False,  # Skip news for speed
        use_fusion=True,
        use_regime=False,
        n_workers=1
    )
    
    print(f"\n🔄 Analyzing all stocks with technical indicators only...")
    results_pass1 = runner_pass1.run(parallel=False)
    
    if not results_pass1:
        print("❌ No results from Pass 1")
        return
    
    # Filter actionable signals
    actionable_pass1 = [r for r in results_pass1 if r['signal'] in ['BUY', 'SELL'] 
                        and r.get('confidence', 0) >= MIN_CONFIDENCE]
    
    # Sort by preliminary fusion score
    actionable_pass1.sort(key=lambda x: abs(x.get('fused_score', 0)), reverse=True)
    
    # Select top candidates for news enrichment
    candidates = actionable_pass1[:NEWS_CANDIDATES]
    
    print(f"\n✅ Pass 1 complete:")
    print(f"   Total analyzed: {len(results_pass1)}")
    print(f"   Actionable signals: {len(actionable_pass1)}")
    print(f"   Candidates for news: {len(candidates)}")
    
    if not candidates:
        print("❌ No actionable candidates found")
        return
    
    # ========================================================================
    # PASS 2: News Enhancement (Slow but limited)
    # ========================================================================
    print("\n" + "=" * 70)
    print(f"📰 PASS 2: News Enhancement for top {len(candidates)} candidates")
    print("=" * 70)
    
    # Initialize news client
    news_client = FreeNewsClient()
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Fetch news for candidates (with translation + Google RSS fallback)
    print(f"\n🔄 Fetching news with translation layer...")
    print(f"   (newsdata.io → Google News RSS fallback)")
    
    news_sentiments = {}
    for i, candidate in enumerate(candidates, 1):
        ticker = candidate['ticker']
        print(f"   [{i}/{len(candidates)}] {ticker}...", end=" ", flush=True)
        
        try:
            # Force fresh fetch (use_cache=False)
            sentiment = news_client.get_sentiment(ticker, today, use_cache=False)
            news_sentiments[ticker] = sentiment
            print(f"sentiment: {sentiment:+.2f}")
        except Exception as e:
            print(f"ERROR: {e}")
            news_sentiments[ticker] = 0.0
    
    print(f"\n✅ News fetched for {len(news_sentiments)} tickers")
    
    # ========================================================================
    # Recalculate Fusion Scores with News
    # ========================================================================
    print("\n" + "=" * 70)
    print("🔄 Recalculating fusion scores with news sentiment")
    print("=" * 70)
    
    # Update candidates with news sentiment and recalculate fusion
    for candidate in candidates:
        ticker = candidate['ticker']
        news_sentiment = news_sentiments.get(ticker, 0.0)
        candidate['news_sentiment'] = news_sentiment
        
        # Recalculate fused score with news
        # Fusion formula from production_enhanced.py:
        # fused = 0.40 * trend + 0.25 * momentum + 0.20 * volatility + 0.15 * sentiment
        trend_score = candidate.get('trend_score', 0)
        momentum_score = candidate.get('momentum_score', 0)
        volatility_score = candidate.get('volatility_score', 0)
        
        # Normalize news sentiment to same scale as other components
        # Sentiment is -1 to +1, other scores are -1 to +1
        news_score = news_sentiment
        
        # Weighted fusion (from SignalFusion class)
        fused_score = (
            0.40 * trend_score +
            0.25 * momentum_score +
            0.20 * volatility_score +
            0.15 * news_score
        )
        
        candidate['fused_score'] = fused_score
        
        # Update confidence based on news alignment
        base_confidence = candidate.get('confidence', 0.5)
        
        # Boost confidence if news aligns with signal
        signal = candidate['signal']
        if signal == 'BUY' and news_sentiment > 0.10:
            candidate['confidence'] = min(0.95, base_confidence + 0.05)
        elif signal == 'BUY' and news_sentiment < -0.10:
            candidate['confidence'] = max(0.30, base_confidence - 0.05)
        elif signal == 'SELL' and news_sentiment < -0.10:
            candidate['confidence'] = min(0.95, base_confidence + 0.05)
        elif signal == 'SELL' and news_sentiment > 0.10:
            candidate['confidence'] = max(0.30, base_confidence - 0.05)
    
    # Re-sort by updated fusion score
    candidates.sort(key=lambda x: abs(x.get('fused_score', 0)), reverse=True)
    
    # Take top N recommendations
    top_recommendations = candidates[:MAX_RECOMMENDATIONS]
    
    # Stats
    buy_count = len([c for c in candidates if c['signal'] == 'BUY'])
    sell_count = len([c for c in candidates if c['signal'] == 'SELL'])
    
    print(f"\n📊 Final Results:")
    print(f"   BUY signals: {buy_count}")
    print(f"   SELL signals: {sell_count}")
    print(f"   Top recommendations: {len(top_recommendations)}")
    
    # ========================================================================
    # Format Output
    # ========================================================================
    print(f"\n🎯 Top {len(top_recommendations)} recommendations:")
    
    # Build detailed recommendations
    detailed_recommendations = []
    for idx, result in enumerate(top_recommendations, 1):
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
            'reversals': reversals,
            'index': idx
        })
        
        # Print summary
        signal = result['signal']
        ticker = result['ticker']
        conf = result.get('confidence', 0)
        alloc = allocation.get('allocation_pct', 0)
        print(f"   {idx}. {ticker}: {signal} ({conf:.0%} conf, {alloc:.0%} allocation)")
    
    # Build stats
    stats = {
        'total_analyzed': len(results_pass1),
        'buy_count': buy_count,
        'sell_count': sell_count,
        'news_enriched': len(news_sentiments)
    }
    
    # Format and save alert
    alert_text = format_full_alert(detailed_recommendations, stats, news_cache=news_client.cache)
    
    # Save to file
    ALERT_FILE.write_text(alert_text)
    print(f"\n✅ Alert saved to {ALERT_FILE}")
    
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
    
    # Log cache stats
    try:
        cache_stats = news_client.get_cache_stats()
        print(f"\n💾 News cache: {cache_stats['total_entries']} entries ({cache_stats['fresh']} fresh)")
    except:
        pass
    
    return detailed_recommendations


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_monitor()
