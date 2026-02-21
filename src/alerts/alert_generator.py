#!/usr/bin/env python3
"""
Alert generator for trading signals.
Formats signals for Telegram delivery.
"""

from typing import List, Dict
from datetime import datetime


def generate_trading_alerts(results: List[Dict], top_n: int = 5) -> str:
    """
    Generate formatted trading alerts for Telegram.
    
    Args:
        results: List of analysis results from production runner
        top_n: Number of top signals to include
    
    Returns:
        Formatted string for Telegram
    """
    if not results:
        return "📊 No trading signals at this time."
    
    # Sort by conviction
    sorted_results = sorted(results, key=lambda x: abs(x.get('conviction', 0)), reverse=True)
    
    # Filter to actionable signals
    buy_signals = [r for r in sorted_results if r['signal'] in ['STRONG_BUY', 'BUY']][:top_n]
    sell_signals = [r for r in sorted_results if r['signal'] in ['STRONG_SELL', 'SELL']][:3]
    
    lines = []
    lines.append("🚨 TOP TRADING OPPORTUNITIES")
    lines.append(f"Generated: {datetime.now().strftime('%H:%M:%S')}")
    lines.append("")
    
    # BUY signals
    for i, r in enumerate(buy_signals, 1):
        fund = r.get('fundamentals', {})
        features = r.get('features', {})
        
        # Number emoji
        num_emoji = ['1️⃣', '2️⃣', '3️⃣', '4️⃣', '5️⃣'][i-1]
        
        lines.append(f"{num_emoji}  {r['ticker'].replace('.SA', '')} - {r['signal']} 🟢")
        lines.append(f"    Price: R${r['price']:.2f}")
        lines.append(f"    └─ Drivers:")
        
        # Trend
        trend_conf = r.get('confidence', 0) * 100
        lines.append(f"       • Trend: {r['trend'].upper()} ({trend_conf:.0f}% confidence)")
        
        # News
        if r.get('news_sentiment'):
            ns = r['news_sentiment']
            lines.append(f"       • News: {ns:+.2f} sentiment")
            if ns > 0.1:
                lines.append(f"         ✅ Positive news boost")
            elif ns < -0.1:
                lines.append(f"         ⚠️ Negative news headwind")
        
        # Fundamentals
        if fund:
            lines.append(f"       • Fundamentals: Grade {fund.get('fundamental_grade', 'N/A')}")
            
            if fund.get('pe_ratio') and fund.get('pb_ratio'):
                lines.append(f"         P/E: {fund['pe_ratio']:.1f} | P/B: {fund['pb_ratio']:.2f}")
            
            if fund.get('roe'):
                lines.append(f"         ROE: {fund['roe']:.1f}% | Div: {fund.get('div_yield', 0):.1f}%")
            
            # Strengths
            if fund.get('strengths'):
                lines.append(f"         ✅ {', '.join(fund['strengths'][:2])}")
            
            # Weaknesses
            if fund.get('weaknesses'):
                lines.append(f"         ⚠️ {', '.join(fund['weaknesses'][:2])}")
        
        # Position
        pos_pct = r.get('position_size', 0) * 100
        conviction_label = "VERY HIGH" if pos_pct > 50 else "HIGH" if pos_pct > 30 else "MEDIUM"
        lines.append(f"       • Position: {pos_pct:.0f}% ({conviction_label} conviction)")
        
        # Action notes
        if fund and fund.get('action_notes'):
            lines.append(f"    └─ Watch for:")
            for note in fund['action_notes'][:2]:
                lines.append(f"       {note}")
        
        lines.append("")
    
    # Summary
    lines.append("=" * 70)
    lines.append(f"📊 Summary: {len(buy_signals)} BUY signals | {len(sell_signals)} SELL signals")
    lines.append("=" * 70)
    
    # SELL signals (compact)
    if sell_signals:
        lines.append("")
        lines.append("🔴 SELL SIGNALS:")
        for r in sell_signals:
            fund = r.get('fundamentals', {})
            grade = fund.get('fundamental_grade', 'N/A') if fund else 'N/A'
            lines.append(f"  • {r['ticker'].replace('.SA', '')} - {r['trend']} | Grade: {grade}")
    
    return "\n".join(lines)


def generate_fundamental_report(results: List[Dict], category: str = "value") -> str:
    """
    Generate a fundamental-focused report.
    
    Args:
        results: List of analysis results
        category: "value", "quality", or "momentum"
    
    Returns:
        Formatted string for Telegram
    """
    if not results:
        return "📊 No results to report."
    
    lines = []
    lines.append(f"📊 {category.upper()} STOCKS REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    
    # Filter by category
    filtered = []
    for r in results:
        fund = r.get('fundamentals', {})
        if not fund:
            continue
        
        if category == "value" and fund.get('is_value_pick'):
            filtered.append(r)
        elif category == "quality" and fund.get('is_quality_pick'):
            filtered.append(r)
        elif category == "momentum" and fund.get('is_momentum_pick'):
            filtered.append(r)
    
    # Sort by composite score
    filtered.sort(key=lambda x: x.get('fundamentals', {}).get('composite_score', 0), reverse=True)
    
    if not filtered:
        lines.append(f"No {category} stocks found in current analysis.")
        return "\n".join(lines)
    
    for i, r in enumerate(filtered[:10], 1):
        fund = r.get('fundamentals', {})
        
        lines.append(f"{i:2}. {r['ticker'].replace('.SA', '')} - Grade {fund.get('fundamental_grade')}")
        lines.append(f"    Value: {fund.get('value_score', 0):.0f} | Quality: {fund.get('quality_score', 0):.0f} | Growth: {fund.get('growth_score', 0):.0f}")
        
        if fund.get('pe_ratio'):
            lines.append(f"    P/E: {fund['pe_ratio']:.1f} | P/B: {fund.get('pb_ratio', 0):.2f} | ROE: {fund.get('roe', 0):.1f}%")
        
        if fund.get('strengths'):
            lines.append(f"    ✅ {', '.join(fund['strengths'][:2])}")
        
        lines.append("")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test with sample data
    sample = [
        {
            'ticker': 'PETR4.SA',
            'price': 37.97,
            'signal': 'BUY',
            'trend': 'uptrend',
            'confidence': 0.75,
            'position_size': 0.45,
            'fundamentals': {
                'fundamental_grade': 'A',
                'value_score': 85,
                'quality_score': 70,
                'growth_score': 55,
                'pe_ratio': 6.31,
                'pb_ratio': 1.16,
                'roe': 18.3,
                'div_yield': 8.5,
                'strengths': ['Low P/E (6.3)', 'High dividend yield (8.5%)'],
                'action_notes': ['💡 QUALITY MOMENTUM: High quality in uptrend']
            }
        }
    ]
    
    print(generate_trading_alerts(sample))