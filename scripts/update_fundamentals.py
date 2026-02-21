#!/usr/bin/env python3
"""
Daily fundamental data scraper for Brazilian stocks.

Scrapes fundamentus.com.br for all monitored stocks and caches results.
Should run once per day (e.g., via cron at 6:00 AM before market open).

Usage:
    python scripts/update_fundamentals.py [--force]
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fundamentals.fundamentus_scraper import FundamentusScraper
from src.fundamentals.scorer import FundamentalScorer, score_fundamentals
import json


def load_monitored_tickers() -> list:
    """Load list of monitored tickers."""
    config_path = Path(__file__).parent.parent / 'data' / 'validated_tickers.json'
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get all tickers, remove .SA suffix
    tickers = [t.replace('.SA', '') for t in config.get('all_tickers', [])]
    return tickers


def update_fundamentals(force: bool = False):
    """Update fundamental data for all monitored stocks."""
    scraper = FundamentusScraper()
    
    # Check if cache is stale
    if not force and not scraper.is_cache_stale():
        print("Cache is fresh (less than 24 hours old). Use --force to update anyway.")
        return
    
    # Load tickers
    tickers = load_monitored_tickers()
    print(f"Updating fundamentals for {len(tickers)} stocks...")
    
    # Scrape all tickers
    start_time = datetime.now()
    data = scraper.scrape_all(tickers, delay=0.3)
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Save to cache
    scraper.save_cache(data)
    
    # Calculate scores
    scorer = FundamentalScorer()
    scores = score_fundamentals(data, scorer)
    
    # Save scores
    scores_path = scraper.cache_dir / 'fundamental_scores.json'
    scores_data = {
        'timestamp': datetime.now().isoformat(),
        'total_stocks': len(scores),
        'scores': [
            {
                'ticker': s.ticker,
                'composite_score': s.composite_score,
                'grade': s.grade,
                'value_score': s.value_score,
                'quality_score': s.quality_score,
                'growth_score': s.growth_score,
                'is_value': s.is_value,
                'is_quality': s.is_quality,
                'is_growth': s.is_growth,
                'is_undervalued': s.is_undervalued,
                'strengths': s.strengths,
                'weaknesses': s.weaknesses,
            }
            for s in scores
        ]
    }
    
    with open(scores_path, 'w', encoding='utf-8') as f:
        json.dump(scores_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nScraped {len(data)} stocks in {elapsed:.1f}s")
    print(f"Scores saved to {scores_path}")
    
    # Print summary
    grades = {}
    for s in scores:
        grades[s.grade] = grades.get(s.grade, 0) + 1
    
    print(f"\n=== Grade Distribution ===")
    for grade in ['A', 'B', 'C', 'D', 'F']:
        count = grades.get(grade, 0)
        print(f"{grade}: {count} stocks")
    
    # Top 10 by composite score
    print(f"\n=== Top 10 by Composite Score ===")
    for i, s in enumerate(scores[:10], 1):
        print(f"{i:2}. {s.ticker}: {s.composite_score:.1f} ({s.grade})")
    
    # Top value stocks
    print(f"\n=== Top 10 Value Stocks ===")
    value_stocks = [s for s in scores if s.is_value]
    for i, s in enumerate(value_stocks[:10], 1):
        print(f"{i:2}. {s.ticker}: Value={s.value_score:.1f}, Quality={s.quality_score:.1f}")
    
    # Top quality stocks
    print(f"\n=== Top 10 Quality Stocks ===")
    quality_stocks = [s for s in scores if s.is_quality]
    for i, s in enumerate(quality_stocks[:10], 1):
        print(f"{i:2}. {s.ticker}: Quality={s.quality_score:.1f}, Value={s.value_score:.1f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Update fundamental data')
    parser.add_argument('--force', action='store_true', help='Force update even if cache is fresh')
    args = parser.parse_args()
    
    update_fundamentals(force=args.force)