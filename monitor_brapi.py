#!/usr/bin/env python3
"""
Market Monitor - Two-Pass Analysis
1. First pass: All stocks with tech + fundamentals (cached)
2. Second pass: News sentiment for top candidates only
3. Final scores include news impact
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from pathlib import Path
import json
import math

from run_production import ProductionRunner
from src.news.free_news_client import FreeNewsClient
from src.alerts.alert_generator import generate_trading_alerts


def clean_for_json(obj):
    """Convert NaN values for JSON serialization."""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def main():
    print(f"\n{'='*60}")
    print(f"📊 MARKET MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # ============================================================
    # PASS 1: Technical + Fundamental Analysis (all stocks)
    # ============================================================
    print("📈 PASS 1: Technical + Fundamental Analysis")
    print("-" * 60)
    
    runner = ProductionRunner(use_fundamentals=True, n_workers=4)
    results = runner.run()
    
    if not results:
        print("❌ No results generated")
        return
    
    # Sort by composite score
    results.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
    
    # Count signals
    signals = [r.get('signal') for r in results]
    print(f"\n📊 Pass 1 Results:")
    print(f"   STRONG_BUY: {signals.count('STRONG_BUY')}")
    print(f"   BUY: {signals.count('BUY')}")
    print(f"   HOLD: {signals.count('HOLD')}")
    print(f"   SELL: {signals.count('SELL')}")
    print(f"   STRONG_SELL: {signals.count('STRONG_SELL')}")
    
    # ============================================================
    # PASS 2: News Sentiment for Top Candidates
    # ============================================================
    print(f"\n📰 PASS 2: News Sentiment (top 20 candidates)")
    print("-" * 60)
    
    # Get top 20 candidates for news analysis
    top_candidates = [r for r in results if r.get('signal') in ['STRONG_BUY', 'BUY']][:20]
    
    if top_candidates:
        try:
            news_client = FreeNewsClient()
            today = datetime.now().strftime('%Y-%m-%d')
            
            for i, result in enumerate(top_candidates, 1):
                ticker = result.get('ticker', '')
                print(f"   [{i}/20] {ticker}...", end=" ", flush=True)
                
                try:
                    # Get sentiment (this fetches news and caches it)
                    sentiment = news_client.get_sentiment(ticker, today)
                    
                    # Get cached articles (if available)
                    cached = news_client.cache.get_full(ticker)
                    articles = cached.get('articles', []) if cached else []
                    
                    # Store news data
                    result['news_sentiment'] = sentiment
                    result['news_articles'] = articles[:3]  # Top 3 headlines
                    
                    # Adjust composite score based on news
                    # News can shift score by up to ±10 points
                    news_adjustment = sentiment * 10
                    old_composite = result.get('composite_score', 50)
                    new_composite = max(0, min(100, old_composite + news_adjustment))
                    result['composite_score'] = new_composite
                    
                    # Update signal if news changed it significantly
                    if new_composite >= 75:
                        result['signal'] = 'STRONG_BUY'
                    elif new_composite >= 60:
                        result['signal'] = 'BUY'
                    elif new_composite >= 40:
                        result['signal'] = 'HOLD'
                    elif new_composite >= 30:
                        result['signal'] = 'SELL'
                    else:
                        result['signal'] = 'STRONG_SELL'
                    
                    # Adjust position size based on news
                    old_position = result.get('position_size', 0.15)
                    if sentiment > 0.2:
                        result['position_size'] = min(0.25, old_position * 1.15)
                    elif sentiment < -0.2:
                        result['position_size'] = old_position * 0.7
                    
                    print(f"sentiment: {sentiment:+.2f} ({len(articles)} articles)")
                    
                except Exception as e:
                    print(f"⚠️ {e}")
                    result['news_sentiment'] = 0.0
            
            # Re-sort after news adjustments
            results.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
            
        except Exception as e:
            print(f"⚠️ News client error: {e}")
    
    # ============================================================
    # Generate Alerts
    # ============================================================
    print(f"\n{'='*60}")
    print("📝 Generating Alerts")
    print("-" * 60)
    
    # Count final signals
    final_signals = [r.get('signal') for r in results]
    strong_signals = [r for r in results if r.get('signal') in ['STRONG_BUY', 'STRONG_SELL']]
    
    if strong_signals:
        alert = generate_trading_alerts(results, top_n=5)
        
        # Save to file for cron delivery
        alert_path = Path('live_alerts.txt')
        with open(alert_path, 'w') as f:
            f.write(alert)
        
        print(f"✅ Alert generated with {len(strong_signals)} strong signals")
        print(f"   Saved to: {alert_path}")
    else:
        # Write empty alert
        with open('live_alerts.txt', 'w') as f:
            f.write(f"📊 Market Update - {datetime.now().strftime('%H:%M')}\n\n")
            f.write("No STRONG_BUY or STRONG_SELL signals at this time.\n\n")
            buy_count = final_signals.count('BUY')
            if buy_count > 0:
                f.write(f"💡 {buy_count} BUY signals available (lower conviction).\n")
        
        print("😐 No strong signals today")
    
    # Save full results
    with open('data/full_results.json', 'w') as f:
        json.dump(clean_for_json(results), f, indent=2, default=str)
    
    # Summary
    print(f"\n📊 Final Results (after news):")
    print(f"   STRONG_BUY: {final_signals.count('STRONG_BUY')}")
    print(f"   BUY: {final_signals.count('BUY')}")
    print(f"   HOLD: {final_signals.count('HOLD')}")
    print(f"   SELL: {final_signals.count('SELL')}")
    print(f"   STRONG_SELL: {final_signals.count('STRONG_SELL')}")
    
    print(f"\n{'='*60}")
    print(f"✅ Done at {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
