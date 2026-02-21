#!/usr/bin/env python3
"""
Google Sheets integration with ALL fundamental metrics from Fundamentus.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Paths
TOKEN_FILE = Path.home() / 'sheets_token.json'
CREDENTIALS_FILE = Path.home() / '.openclaw' / 'credentials' / 'gmail_credentials.json'
SHEET_ID_FILE = Path(__file__).parent.parent / 'data' / 'sheets_id.txt'

# Sheet name
SHEET_NAME = "Stock Signals Log"

# Column headers - ALL metrics
HEADERS = [
    # Timestamp & Basic Info
    "Date", "Time", "Ticker", "Company Name", "Sector", "Subsector",
    "Price", "Signal", "Composite Score",
    
    # Technical Analysis
    "Technical Score", "Trend", "Confidence",
    "Volume Momentum", "Unusual Volume", "Volatility Regime",
    "News Sentiment",
    
    # Fundamental Scores (from scorer)
    "Fundamental Grade", "Value Score", "Quality Score", "Growth Score",
    
    # Valuation Metrics (from Fundamentus)
    "P/E Ratio", "P/B Ratio", "P/S Ratio", "P/EBIT", 
    "EV/EBITDA", "EV/EBIT",
    
    # Profitability Metrics (from Fundamentus)
    "ROE", "ROIC", 
    "Gross Margin", "EBIT Margin", "Net Margin",
    
    # Financial Health (from Fundamentus)
    "Debt/Equity", "Current Ratio",
    "Div Yield %",
    
    # Growth (from Fundamentus)
    "Revenue Growth 5Y",
    
    # Per Share Metrics (from Fundamentus)
    "EPS", "BVPS",
    
    # Scale Metrics (from Fundamentus)
    "Market Cap (M)", "Total Assets (M)", "Total Equity (M)",
    "Net Debt (M)", "EBIT (M)", "Net Income (M)", "Revenue (M)",
    
    # Flags
    "Is Value Pick", "Is Quality Pick", "Is Growth Pick", "Is Avoid",
    
    # Position Sizing
    "Position Size", "Position %",
    
    # Analysis Notes
    "Strengths", "Weaknesses"
]


def get_credentials() -> Optional[Credentials]:
    """Load credentials from token file."""
    if not TOKEN_FILE.exists():
        print(f"❌ Token file not found: {TOKEN_FILE}")
        return None
    
    with open(TOKEN_FILE, 'r') as f:
        token_data = json.load(f)
    
    with open(CREDENTIALS_FILE, 'r') as f:
        client_data = json.load(f)
    
    creds = Credentials(
        token=token_data.get('access_token') or token_data.get('token'),
        refresh_token=token_data.get('refresh_token'),
        token_uri=token_data.get('token_uri', 'https://oauth2.googleapis.com/token'),
        client_id=client_data['installed']['client_id'],
        client_secret=client_data['installed']['client_secret'],
        scopes=['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive.file']
    )
    
    return creds


def get_sheets_service():
    """Get Google Sheets service."""
    creds = get_credentials()
    if not creds:
        return None
    return build('sheets', 'v4', credentials=creds)


def get_drive_service():
    """Get Google Drive service."""
    creds = get_credentials()
    if not creds:
        return None
    return build('drive', 'v3', credentials=creds)


def find_or_create_sheet(service, drive_service) -> Optional[str]:
    """Find existing sheet or create new one."""
    if SHEET_ID_FILE.exists():
        with open(SHEET_ID_FILE, 'r') as f:
            sheet_id = f.read().strip()
            if sheet_id:
                # Verify sheet still exists
                try:
                    service.spreadsheets().get(spreadsheetId=sheet_id).execute()
                    print(f"✅ Using existing sheet: {sheet_id}")
                    return sheet_id
                except:
                    pass
    
    # Create new sheet
    spreadsheet = {
        'properties': {
            'title': SHEET_NAME
        }
    }
    
    spreadsheet = service.spreadsheets().create(
        body=spreadsheet,
        fields='spreadsheetId'
    ).execute()
    
    sheet_id = spreadsheet.get('spreadsheetId')
    print(f"✅ Created new sheet: {sheet_id}")
    
    SHEET_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SHEET_ID_FILE, 'w') as f:
        f.write(sheet_id)
    
    return sheet_id


def init_headers(service, sheet_id: str) -> bool:
    """Initialize sheet with headers."""
    try:
        body = {'values': [HEADERS]}
        
        service.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range='Sheet1!A1',
            valueInputOption='USER_ENTERED',
            body=body
        ).execute()
        
        print(f"✅ Initialized {len(HEADERS)} columns")
        return True
    except HttpError as e:
        print(f"❌ Could not initialize headers: {e}")
        return False


def format_row(result: Dict, raw_fundamentals: Dict = None) -> List:
    """Format a single analysis result as a row with ALL metrics."""
    fund = result.get('fundamentals', {})
    features = result.get('features', {})
    
    # Merge raw fundamentals if provided
    if raw_fundamentals:
        fund = {**raw_fundamentals, **fund}
    
    def num_str(val, fmt=".2f"):
        if val is None:
            return ""
        try:
            return f"{float(val):{fmt}}"
        except:
            return str(val) if val else ""
    
    def bool_str(val):
        return "Yes" if val else "No"
    
    def millions(val):
        if val is None:
            return ""
        try:
            return f"{float(val)/1000:.1f}"  # Convert to millions
        except:
            return ""
    
    # Format lists
    strengths = "; ".join(fund.get('strengths', [])) if fund else ""
    weaknesses = "; ".join(fund.get('weaknesses', [])) if fund else ""
    
    return [
        # Timestamp & Basic Info
        datetime.now().strftime('%Y-%m-%d'),
        datetime.now().strftime('%H:%M:%S'),
        result.get('ticker', '').replace('.SA', ''),
        fund.get('company_name', '') if fund else '',
        fund.get('sector', '') if fund else '',
        fund.get('subsector', '') if fund else '',
        num_str(result.get('price', 0), ".2f"),
        result.get('signal', ''),
        num_str(result.get('composite_score', 0), ".0f"),
        
        # Technical Analysis
        num_str(result.get('tech_score', 0), ".0f"),
        result.get('trend', ''),
        num_str(result.get('confidence', 0) * 100, ".0f") + "%",
        num_str(features.get('volume_momentum', 1.0), ".2f"),
        bool_str(features.get('unusual_volume', False)),
        features.get('volatility_regime', ''),
        num_str(result.get('news_sentiment', 0), ".2f"),
        
        # Fundamental Scores
        fund.get('grade', '') if fund else '',
        num_str(fund.get('value_score', 0), ".0f"),
        num_str(fund.get('quality_score', 0), ".0f"),
        num_str(fund.get('growth_score', 0), ".0f"),
        
        # Valuation Metrics
        num_str(fund.get('pe_ratio'), ".1f"),
        num_str(fund.get('pb_ratio'), ".2f"),
        num_str(fund.get('ps_ratio'), ".2f"),
        num_str(fund.get('p_ebit'), ".1f"),
        num_str(fund.get('ev_ebitda'), ".1f"),
        num_str(fund.get('ev_ebit'), ".1f"),
        
        # Profitability Metrics
        num_str(fund.get('roe'), ".1f") if fund.get('roe') else "",
        num_str(fund.get('roic'), ".1f") if fund.get('roic') else "",
        num_str(fund.get('gross_margin'), ".1f") if fund.get('gross_margin') else "",
        num_str(fund.get('ebit_margin'), ".1f") if fund.get('ebit_margin') else "",
        num_str(fund.get('net_margin'), ".1f") if fund.get('net_margin') else "",
        
        # Financial Health
        num_str(fund.get('debt_equity'), ".2f") if fund.get('debt_equity') else "",
        num_str(fund.get('current_ratio'), ".2f") if fund.get('current_ratio') else "",
        num_str(fund.get('div_yield'), ".1f") if fund.get('div_yield') else "",
        
        # Growth
        num_str(fund.get('revenue_growth_5y'), ".1f") if fund.get('revenue_growth_5y') else "",
        
        # Per Share Metrics
        num_str(fund.get('eps'), ".2f"),
        num_str(fund.get('bvps'), ".2f"),
        
        # Scale Metrics (in millions)
        millions(fund.get('market_cap')),
        millions(fund.get('total_assets')),
        millions(fund.get('total_equity')),
        millions(fund.get('net_debt')),
        millions(fund.get('ebit')),
        millions(fund.get('net_income')),
        millions(fund.get('revenue')),
        
        # Flags
        bool_str(fund.get('is_value_pick', fund.get('is_value', False))),
        bool_str(fund.get('is_quality_pick', fund.get('is_quality', False))),
        bool_str(fund.get('is_growth', False)),
        bool_str(result.get('is_avoid', False)),
        
        # Position Sizing
        num_str(result.get('position_size', 0), ".4f"),
        num_str(result.get('position_size', 0) * 100, ".0f") + "%",
        
        # Notes
        strengths[:200] if strengths else "",
        weaknesses[:200] if weaknesses else "",
    ]


def append_results(results: List[Dict], dry_run: bool = False) -> bool:
    """Append analysis results to Google Sheet with ALL fundamental metrics."""
    sheets_service = get_sheets_service()
    drive_service = get_drive_service()
    
    if not sheets_service or not drive_service:
        print("❌ Could not initialize Google services")
        return False
    
    # Get or create sheet
    sheet_id = find_or_create_sheet(sheets_service, drive_service)
    if not sheet_id:
        print("❌ Could not get sheet ID")
        return False
    
    # Initialize headers (always re-init to ensure all columns)
    init_headers(sheets_service, sheet_id)
    
    # Load raw fundamentals cache
    fund_cache_path = Path(__file__).parent.parent / 'data' / 'fundamentals' / 'fundamentals_cache.json'
    raw_fundamentals = {}
    if fund_cache_path.exists():
        with open(fund_cache_path, 'r') as f:
            cache_data = json.load(f)
            raw_fundamentals = cache_data.get('data', {})
    
    # Format results as rows
    rows = []
    for r in results:
        if r:
            ticker = r.get('ticker', '').replace('.SA', '')
            raw_fund = raw_fundamentals.get(ticker, {})
            rows.append(format_row(r, raw_fund))
    
    if not rows:
        print("❌ No results to log")
        return False
    
    if dry_run:
        print(f"\n[DRY RUN] Would append {len(rows)} rows to sheet {sheet_id}")
        return True
    
    # Append to sheet
    try:
        body = {'values': rows}
        
        result = sheets_service.spreadsheets().values().append(
            spreadsheetId=sheet_id,
            range='Sheet1!A:A',
            valueInputOption='USER_ENTERED',
            insertDataOption='INSERT_ROWS',
            body=body
        ).execute()
        
        print(f"✅ Appended {len(rows)} rows to Google Sheet")
        print(f"   Sheet ID: {sheet_id}")
        print(f"   URL: https://docs.google.com/spreadsheets/d/{sheet_id}/edit")
        
        return True
    except HttpError as e:
        print(f"❌ Could not append to sheet: {e}")
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Log trading signals to Google Sheets')
    parser.add_argument('--dry-run', action='store_true', help="Don't write to sheet")
    parser.add_argument('--results-file', type=str, help="Load results from JSON file")
    args = parser.parse_args()
    
    if args.results_file:
        with open(args.results_file, 'r') as f:
            results = json.load(f)
    else:
        print("Please provide --results-file")
        sys.exit(1)
    
    success = append_results(results, args.dry_run)
    sys.exit(0 if success else 1)