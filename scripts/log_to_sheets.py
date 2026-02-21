#!/usr/bin/env python3
"""
Log trading signals to Google Sheets.

Creates a daily log of all analyzed stocks with:
- Technical indicators
- Fundamental scores
- Signal decisions
- Position sizing

Usage:
    python scripts/log_to_sheets.py [--dry-run]

Requires gog CLI authentication:
    gog auth credentials /path/to/client_secret.json
    gog auth add your@email.com --services sheets
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


# Google Sheets configuration
SHEET_NAME = "Stock Signals Log"
SHEET_ID_FILE = Path(__file__).parent.parent / "data" / "sheets_id.txt"

# Column headers for the sheet
HEADERS = [
    # Timestamp
    "Date",
    "Time",
    
    # Stock Info
    "Ticker",
    "Price",
    
    # Signal
    "Signal",
    "Composite Score",
    "Technical Score",
    "Fundamental Score",
    
    # Technical Indicators
    "Trend",
    "Confidence",
    "Trend Consensus",
    
    # Volume/Volatility
    "Volume Momentum",
    "Unusual Volume",
    "Volatility Regime",
    
    # News
    "News Sentiment",
    
    # Fundamentals
    "Fundamental Grade",
    "Value Score",
    "Quality Score",
    "Growth Score",
    "P/E",
    "P/B",
    "ROE",
    "ROIC",
    "Div Yield",
    "Debt/Equity",
    
    # Picks
    "Is Value",
    "Is Quality",
    "Is Momentum",
    "Is Avoid",
    
    # Position
    "Position Size",
    "Position %",
    
    # Strengths/Weaknesses
    "Strengths",
    "Weaknesses",
    "Action Notes",
]


def get_or_create_sheet() -> Optional[str]:
    """Get existing sheet ID or create new sheet."""
    # Check if we have a saved sheet ID
    if SHEET_ID_FILE.exists():
        with open(SHEET_ID_FILE, 'r') as f:
            sheet_id = f.read().strip()
            if sheet_id:
                return sheet_id
    
    # Create new sheet
    print(f"Creating new Google Sheet: {SHEET_NAME}")
    
    # Use gog to create sheet (would need proper API)
    # For now, return None and prompt user to create manually
    print(f"\nPlease create a Google Sheet named '{SHEET_NAME}' and paste its ID here.")
    print("The ID is the long string in the URL: https://docs.google.com/spreadsheets/d/[SHEET_ID]/edit")
    
    sheet_id = input("Sheet ID: ").strip()
    
    if sheet_id:
        # Save for future use
        SHEET_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SHEET_ID_FILE, 'w') as f:
            f.write(sheet_id)
        return sheet_id
    
    return None


def format_row(result: Dict, date: str, time: str) -> List[str]:
    """Format a single analysis result as a row."""
    fund = result.get('fundamentals', {})
    features = result.get('features', {})
    
    # Handle boolean strings from JSON
    def bool_str(val):
        if isinstance(val, str):
            return val
        return str(val) if val is not None else ""
    
    # Format lists
    strengths = ", ".join(fund.get('strengths', [])) if fund else ""
    weaknesses = ", ".join(fund.get('weaknesses', [])) if fund else ""
    action_notes = ", ".join(fund.get('action_notes', [])) if fund else ""
    
    return [
        # Timestamp
        date,
        time,
        
        # Stock Info
        result.get('ticker', '').replace('.SA', ''),
        f"{result.get('price', 0):.2f}",
        
        # Signal
        result.get('signal', ''),
        f"{result.get('conviction', 0) * 100:.1f}",
        f"{result.get('confidence', 0) * 100:.1f}",
        f"{fund.get('composite_score', 0):.1f}" if fund else "",
        
        # Technical
        result.get('trend', ''),
        f"{result.get('confidence', 0) * 100:.1f}%",
        result.get('trend', ''),  # Trend consensus
        
        # Volume/Volatility
        f"{features.get('volume_momentum', 1.0):.2f}",
        str(features.get('unusual_volume', False)),
        features.get('volatility_regime', ''),
        
        # News
        f"{result.get('news_sentiment', 0):.2f}",
        
        # Fundamentals
        fund.get('fundamental_grade', '') if fund else "",
        f"{fund.get('value_score', 0):.0f}" if fund else "",
        f"{fund.get('quality_score', 0):.0f}" if fund else "",
        f"{fund.get('growth_score', 0):.0f}" if fund else "",
        f"{fund.get('pe_ratio', 0):.1f}" if fund and fund.get('pe_ratio') else "",
        f"{fund.get('pb_ratio', 0):.2f}" if fund and fund.get('pb_ratio') else "",
        f"{fund.get('roe', 0):.1f}" if fund and fund.get('roe') else "",
        f"{fund.get('roic', 0):.1f}" if fund and fund.get('roic') else "",
        f"{fund.get('div_yield', 0):.1f}" if fund and fund.get('div_yield') else "",
        f"{fund.get('debt_equity', 0):.2f}" if fund and fund.get('debt_equity') else "",
        
        # Picks
        bool_str(fund.get('is_value_pick', False)),
        bool_str(fund.get('is_quality_pick', False)),
        bool_str(fund.get('is_momentum_pick', False)),
        bool_str(fund.get('is_avoid', False)),
        
        # Position
        f"{result.get('position_size', 0):.4f}",
        f"{result.get('position_size', 0) * 100:.0f}%",
        
        # Notes
        strengths[:200] if strengths else "",  # Truncate for Sheets
        weaknesses[:200] if weaknesses else "",
        action_notes[:200] if action_notes else "",
    ]


def append_to_sheet(sheet_id: str, rows: List[List[str]], dry_run: bool = False) -> bool:
    """Append rows to the Google Sheet using gog CLI."""
    if dry_run:
        print(f"\n[DRY RUN] Would append {len(rows)} rows to sheet {sheet_id}")
        for i, row in enumerate(rows[:3]):
            print(f"  Row {i+1}: {row[:5]}...")
        return True
    
    # Convert rows to JSON
    values_json = json.dumps(rows)
    
    # Use gog sheets append
    cmd = [
        'gog', 'sheets', 'append',
        sheet_id,
        'Sheet1!A:AO',  # All columns
        '--values-json', values_json,
        '--insert', 'INSERT_ROWS'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Appended {len(rows)} rows to sheet")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to append to sheet: {e.stderr}")
        return False


def initialize_headers(sheet_id: str, dry_run: bool = False) -> bool:
    """Initialize sheet with headers if empty."""
    if dry_run:
        print(f"[DRY RUN] Would initialize headers: {HEADERS[:5]}...")
        return True
    
    # Use gog sheets update to set headers
    values_json = json.dumps([HEADERS])
    
    cmd = [
        'gog', 'sheets', 'update',
        sheet_id,
        'Sheet1!A1:AO1',
        '--values-json', values_json,
        '--input', 'USER_ENTERED'
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Initialized headers")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to initialize headers: {e.stderr}")
        return False


def log_results(results: List[Dict], dry_run: bool = False) -> bool:
    """
    Log analysis results to Google Sheets.
    
    Args:
        results: List of analysis results from production_simple.py
        dry_run: If True, don't actually write to sheet
    
    Returns:
        True if successful, False otherwise
    """
    # Get or create sheet
    sheet_id = get_or_create_sheet()
    if not sheet_id:
        print("❌ No sheet ID available")
        return False
    
    # Current timestamp
    now = datetime.now()
    date = now.strftime('%Y-%m-%d')
    time = now.strftime('%H:%M:%S')
    
    # Format results as rows
    rows = [format_row(r, date, time) for r in results if r]
    
    if not rows:
        print("❌ No results to log")
        return False
    
    # Append to sheet
    return append_to_sheet(sheet_id, rows, dry_run)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Log trading signals to Google Sheets')
    parser.add_argument('--dry-run', action='store_true', help="Don't write to sheet")
    parser.add_argument('--init', action='store_true', help="Initialize headers only")
    parser.add_argument('--results-file', type=str, help="Load results from JSON file")
    args = parser.parse_args()
    
    if args.init:
        sheet_id = get_or_create_sheet()
        if sheet_id:
            initialize_headers(sheet_id, args.dry_run)
        sys.exit(0)
    
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
    
    # Log results
    success = log_results(results, args.dry_run)
    sys.exit(0 if success else 1)