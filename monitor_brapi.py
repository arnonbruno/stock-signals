#!/usr/bin/env python3
"""
Market Monitor - BrAPI + Fundamentals
Runs hourly during market hours and sends alerts to Telegram.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from pathlib import Path
import json

from production_simple import SimpleProductionRunner
from src.alerts.alert_generator import generate_trading_alerts


def main():
    print(f"\n{'='*60}")
    print(f"📊 MARKET MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # Run analysis with fundamentals + news for top picks
    runner = SimpleProductionRunner(
        use_news=True,  # Fetch news for top recommendations
        use_fundamentals=True
    )
    
    results = runner.run()
    
    if not results:
        print("❌ No results generated")
        return
    
    # Filter to actionable signals only
    actionable = [r for r in results if r.get('signal') in ['STRONG_BUY', 'BUY', 'SELL', 'STRONG_SELL']]
    strong_signals = [r for r in actionable if r.get('signal') in ['STRONG_BUY', 'STRONG_SELL']]
    
    print(f"\n📊 Results: {len(results)} analyzed")
    print(f"   STRONG_BUY: {sum(1 for r in results if r.get('signal') == 'STRONG_BUY')}")
    print(f"   BUY: {sum(1 for r in results if r.get('signal') == 'BUY')}")
    print(f"   HOLD: {sum(1 for r in results if r.get('signal') == 'HOLD')}")
    print(f"   SELL: {sum(1 for r in results if r.get('signal') == 'SELL')}")
    print(f"   STRONG_SELL: {sum(1 for r in results if r.get('signal') == 'STRONG_SELL')}")
    
    # Generate alert
    if strong_signals:
        alert = generate_trading_alerts(results, top_n=5)
        
        # Save to file for cron delivery
        alert_path = Path('live_alerts.txt')
        with open(alert_path, 'w', encoding='utf-8') as f:
            f.write(alert)
        
        print(f"\n✅ Alert generated with {len(strong_signals)} strong signals")
        print(f"   Saved to: {alert_path}")
    else:
        print(f"\n😐 No strong signals today")
        # Write empty alert
        with open('live_alerts.txt', 'w', encoding='utf-8') as f:
            f.write(f"📊 Market Update - {datetime.now().strftime('%H:%M')}\n\n")
            f.write("No STRONG_BUY or STRONG_SELL signals at this time.\n\n")
            buy_count = sum(1 for r in results if r.get('signal') == 'BUY')
            if buy_count > 0:
                f.write(f"💡 {buy_count} BUY signals available (lower conviction).\n")
    
    # Also save full results
    with open('data/full_results.json', 'w') as f:
        # Convert any NaN values for JSON serialization
        import math
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(v) for v in obj]
            elif isinstance(obj, float) and math.isnan(obj):
                return None
            return obj
        
        json.dump(clean_for_json(results), f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"✅ Done in {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
