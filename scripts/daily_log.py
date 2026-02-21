#!/usr/bin/env python3
"""
Daily signal logger - Creates CSV for Google Sheets import.

Exports analysis results to CSV format that can be:
1. Manually imported to Google Sheets
2. Automatically synced via gog CLI (after OAuth setup)

Usage:
    python scripts/daily_log.py [--upload]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict

# Output paths
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "logs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_FILE = OUTPUT_DIR / "signals_log.csv"
JSON_FILE = OUTPUT_DIR / "signals_log.json"

# Column headers
HEADERS = [
    "date", "time", "ticker", "price", "signal", "composite_score",
    "technical_score", "fundamental_score", "trend", "confidence",
    "volume_momentum", "unusual_volume", "volatility_regime",
    "news_sentiment", "fundamental_grade", "value_score", "quality_score",
    "growth_score", "pe_ratio", "pb_ratio", "roe", "roic", "div_yield",
    "debt_equity", "is_value", "is_quality", "is_momentum", "is_avoid",
    "position_size", "position_pct", "strengths", "weaknesses", "action_notes"
]


def format_row(result: Dict) -> Dict:
    """Format a single analysis result as a row."""
    fund = result.get('fundamentals', {})
    features = result.get('features', {})
    
    def bool_str(val):
        if isinstance(val, str):
            return val.lower() == 'true'
        return bool(val)
    
    return {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'time': datetime.now().strftime('%H:%M:%S'),
        'ticker': result.get('ticker', '').replace('.SA', ''),
        'price': round(result.get('price', 0), 2),
        'signal': result.get('signal', ''),
        'composite_score': round(result.get('conviction', 0) * 100, 1),
        'technical_score': round(result.get('confidence', 0) * 100, 1),
        'fundamental_score': round(fund.get('composite_score', 0), 1) if fund else '',
        'trend': result.get('trend', ''),
        'confidence': round(result.get('confidence', 0) * 100, 1),
        'volume_momentum': round(features.get('volume_momentum', 1.0), 2),
        'unusual_volume': features.get('unusual_volume', False),
        'volatility_regime': features.get('volatility_regime', ''),
        'news_sentiment': round(result.get('news_sentiment', 0), 2),
        'fundamental_grade': fund.get('fundamental_grade', '') if fund else '',
        'value_score': round(fund.get('value_score', 0)) if fund else '',
        'quality_score': round(fund.get('quality_score', 0)) if fund else '',
        'growth_score': round(fund.get('growth_score', 0)) if fund else '',
        'pe_ratio': round(fund.get('pe_ratio', 0), 1) if fund and fund.get('pe_ratio') else '',
        'pb_ratio': round(fund.get('pb_ratio', 0), 2) if fund and fund.get('pb_ratio') else '',
        'roe': round(fund.get('roe', 0), 1) if fund and fund.get('roe') else '',
        'roic': round(fund.get('roic', 0), 1) if fund and fund.get('roic') else '',
        'div_yield': round(fund.get('div_yield', 0), 1) if fund and fund.get('div_yield') else '',
        'debt_equity': round(fund.get('debt_equity', 0), 2) if fund and fund.get('debt_equity') else '',
        'is_value': bool_str(fund.get('is_value_pick', False)) if fund else False,
        'is_quality': bool_str(fund.get('is_quality_pick', False)) if fund else False,
        'is_momentum': bool_str(fund.get('is_momentum_pick', False)) if fund else False,
        'is_avoid': bool_str(fund.get('is_avoid', False)) if fund else False,
        'position_size': round(result.get('position_size', 0), 4),
        'position_pct': round(result.get('position_size', 0) * 100, 0),
        'strengths': "; ".join(fund.get('strengths', [])) if fund else '',
        'weaknesses': "; ".join(fund.get('weaknesses', [])) if fund else '',
        'action_notes': "; ".join(fund.get('action_notes', [])) if fund else '',
    }


def append_to_csv(results: List[Dict]) -> int:
    """Append results to CSV file."""
    rows = [format_row(r) for r in results if r]
    
    if not rows:
        return 0
    
    # Check if file exists to determine if we need headers
    needs_header = not CSV_FILE.exists()
    
    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)
    
    return len(rows)


def append_to_json(results: List[Dict]) -> int:
    """Append results to JSON file."""
    rows = [format_row(r) for r in results if r]
    
    if not rows:
        return 0
    
    # Load existing data
    if JSON_FILE.exists():
        with open(JSON_FILE, 'r') as f:
            data = json.load(f)
    else:
        data = []
    
    # Append new rows
    data.extend(rows)
    
    # Save
    with open(JSON_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    
    return len(rows)


def sync_to_google_sheets() -> bool:
    """Sync CSV to Google Sheets using gog CLI."""
    # Check if gog is authenticated
    import subprocess
    result = subprocess.run(['gog', 'auth', 'list'], capture_output=True, text=True)
    
    if 'No tokens' in result.stdout or result.returncode != 0:
        print("❌ gog not authenticated. Run:")
        print("   1. Create Google Cloud project and OAuth credentials")
        print("   2. gog auth credentials /path/to/client_secret.json")
        print("   3. gog auth add your@email.com --services sheets")
        return False
    
    # Get sheet ID
    sheet_id_file = Path(__file__).parent.parent / "data" / "sheets_id.txt"
    if not sheet_id_file.exists():
        print("❌ No Google Sheet ID found. Create a sheet and save its ID to:")
        print(f"   {sheet_id_file}")
        return False
    
    with open(sheet_id_file, 'r') as f:
        sheet_id = f.read().strip()
    
    # Read CSV and append to sheet
    import csv
    with open(CSV_FILE, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    if not rows:
        print("❌ No data to sync")
        return False
    
    # Convert to JSON for gog
    values_json = json.dumps(rows)
    
    # Append to sheet
    cmd = [
        'gog', 'sheets', 'append',
        sheet_id,
        'Sheet1!A:AG',
        '--values-json', values_json,
        '--insert', 'INSERT_ROWS'
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Synced {len(rows)} rows to Google Sheets")
        
        # Clear CSV after successful sync
        CSV_FILE.unlink()
        print("✅ Cleared local CSV")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to sync: {e.stderr}")
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Log trading signals to CSV/Sheets')
    parser.add_argument('--sync', action='store_true', help="Sync to Google Sheets")
    parser.add_argument('--results-file', type=str, help="Load results from JSON file")
    args = parser.parse_args()
    
    if args.sync:
        success = sync_to_google_sheets()
        sys.exit(0 if success else 1)
    
    # Load results
    if args.results_file:
        with open(args.results_file, 'r') as f:
            results = json.load(f)
    else:
        # Run analysis
        print("Running analysis...")
        from production_simple import SimpleProductionRunner
        runner = SimpleProductionRunner(use_news=True, use_fundamentals=True)
        results = runner.run()
    
    # Save to CSV and JSON
    csv_count = append_to_csv(results)
    json_count = append_to_json(results)
    
    print(f"✅ Logged {csv_count} signals to {CSV_FILE}")
    print(f"✅ Logged {json_count} signals to {JSON_FILE}")
    print(f"\nTo sync to Google Sheets:")
    print(f"  1. Create a Google Sheet")
    print(f"  2. Save the sheet ID to data/sheets_id.txt")
    print(f"  3. Run: python scripts/daily_log.py --sync")