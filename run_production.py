#!/usr/bin/env python3
"""
Full production analysis using cached BrAPI quotes
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path
import time
import concurrent.futures

from src.signals.trend_detector_v2 import TrendDetectorV2
from src.features.feature_engineering import get_feature_engineer
from src.fundamentals.integration import FundamentalIntegrator
from src.data.brapi_client import BrAPIClient


def load_tickers():
    with open('data/validated_tickers.json', 'r') as f:
        data = json.load(f)
    return data.get('all_tickers', [])


def load_cached_quotes():
    """Load quotes from cache."""
    cache_path = Path('data/quotes_cache.json')
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            return json.load(f)
    return {}


class ProductionRunner:
    def __init__(self, use_fundamentals=True, n_workers=4):
        api_key = os.environ.get('BRAPI_API_KEY', '1PinDyFUxXXBdvkdGN9Bi2')
        self.brapi = BrAPIClient(api_key=api_key)
        self.trend_detector = TrendDetectorV2()
        self.use_fundamentals = use_fundamentals
        self.n_workers = n_workers
        
        if use_fundamentals:
            self.fundamental_integrator = FundamentalIntegrator()
        
        # Load fundamentals cache once
        self.fundamentals_cache = {}
        fund_cache_path = Path('data/fundamentals/fundamentals_cache.json')
        if fund_cache_path.exists():
            with open(fund_cache_path, 'r') as f:
                cache_data = json.load(f)
                self.fundamentals_cache = cache_data.get('data', {})
        
        print(f"✅ Sistema inicializado (fundamentals={'ON' if use_fundamentals else 'OFF'})")
        print(f"   Cached quotes: Will use existing cache")
        print(f"   Fundamentals cache: {len(self.fundamentals_cache)} stocks")
    
    def analyze_ticker(self, ticker, quote_data):
        try:
            print(f"📊 {ticker}...", end=" ", flush=True)
            
            # Get historical data
            data = self.brapi.get_historical(ticker, range_="3mo")
            
            if data is None or len(data) < 50:
                print(f"⚠️ Insufficient data ({len(data) if data is not None else 0} days)")
                return None
            
            # Current price from quote
            current_price = quote_data.get('regularMarketPrice', data['Close'].iloc[-1]) if quote_data else data['Close'].iloc[-1]
            
            # Trend detection
            trend_result = self.trend_detector.detect_trend(data)
            consensus = trend_result.get('consensus', 'unknown')
            confidence = trend_result.get('confidence', 0.0)
            
            # Features
            feature_engineer = get_feature_engineer()
            features = feature_engineer.generate_all_features(ticker, data)
            
            # Volume adjustment
            volume_features = features.get('volume', {})
            if volume_features.get('unusual_volume', False):
                confidence *= 1.05
            
            vol_regime = features.get('volatility', {}).get('regime', 'medium')
            if vol_regime == 'high':
                confidence *= 0.90
            
            # Map trend
            if consensus in ['uptrend', 'bull_pullback']:
                trend = "uptrend"
            elif consensus in ['downtrend', 'bear_bounce']:
                trend = "downtrend"
            else:
                trend = "consolidation"
            
            print(f"[TREND {trend} @ {confidence:.0%}]", end=" ", flush=True)
            
            # Fundamentals
            fundamentals = None
            if self.use_fundamentals:
                fund_score_data = self.fundamental_integrator.get_fundamental_score(ticker)
                if fund_score_data:
                    fundamentals = fund_score_data.copy()
                    
                    # Add raw metrics from cache
                    raw_data = self.fundamentals_cache.get(ticker, {})
                    if raw_data:
                        fundamentals['pe_ratio'] = raw_data.get('pe_ratio')
                        fundamentals['pb_ratio'] = raw_data.get('pb_ratio')
                        fundamentals['roe'] = raw_data.get('roe')
                        fundamentals['roic'] = raw_data.get('roic')
                        fundamentals['div_yield'] = raw_data.get('div_yield')
                        fundamentals['debt_equity'] = raw_data.get('debt_equity')
                        fundamentals['fundamental_grade'] = fundamentals.get('grade', '')
                        fundamentals['is_value_pick'] = fundamentals.get('is_value', False)
                        fundamentals['is_quality_pick'] = fundamentals.get('is_quality', False)
            
            # Calculate scores
            tech_score = confidence * 100
            fund_score = fundamentals.get('composite_score', 50) if fundamentals else 50
            composite = (tech_score * 0.5) + (fund_score * 0.5)
            
            # Signal
            if composite >= 75:
                signal = "STRONG_BUY"
            elif composite >= 60:
                signal = "BUY"
            elif composite >= 40:
                signal = "HOLD"
            elif composite >= 30:
                signal = "SELL"
            else:
                signal = "STRONG_SELL"
            
            # Position sizing
            position_size = 0.20  # Default 20%
            if fundamentals:
                if fundamentals.get('is_quality_pick'):
                    position_size *= 1.2
                if fundamentals.get('is_value_pick'):
                    position_size *= 1.15
                if fund_score < 40:
                    position_size *= 0.5
            
            print(f"{signal}")
            
            return {
                'ticker': ticker,
                'price': current_price,
                'signal': signal,
                'trend': trend,
                'confidence': confidence,
                'news_sentiment': 0.0,  # Not fetching news in this run
                'composite_score': composite,
                'tech_score': tech_score,
                'fund_score': fund_score,
                'position_size': position_size,
                'features': features,
                'fundamentals': fundamentals,
                'is_avoid': fundamentals.get('is_avoid', False) if fundamentals else False
            }
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def run(self):
        tickers = load_tickers()
        quotes = load_cached_quotes()
        
        print(f"\n{'='*70}")
        print(f"🚀 PRODUÇÃO (BrAPI) - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"📊 Analyzing {len(tickers)} tickers")
        print(f"📦 Using {len(quotes)} cached quotes")
        print(f"{'='*70}\n")
        
        results = []
        
        # Analyze tickers that have quotes
        tickers_with_quotes = [t for t in tickers if t in quotes]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {}
            for ticker in tickers_with_quotes:
                future = executor.submit(self.analyze_ticker, ticker, quotes.get(ticker))
                futures[future] = ticker
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
        
        # Sort by composite score
        results.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
        
        # Summary
        print(f"\n{'='*70}")
        signals = [r['signal'] for r in results]
        print(f"📊 Summary: {signals.count('STRONG_BUY')} STRONG_BUY | "
              f"{signals.count('BUY')} BUY | {signals.count('HOLD')} HOLD | "
              f"{signals.count('SELL')} SELL | {signals.count('STRONG_SELL')} STRONG_SELL")
        print(f"{'='*70}\n")
        
        return results


if __name__ == '__main__':
    runner = ProductionRunner(use_fundamentals=True, n_workers=4)
    results = runner.run()
    
    # Save results
    with open('data/full_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Saved {len(results)} results to data/full_results.json")
    
    # Sync to Google Sheets
    from scripts.sheets_sync import append_results
    success = append_results(results)
    if success:
        print("✅ Synced to Google Sheets!")