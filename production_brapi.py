#!/usr/bin/env python3
"""
Production Runner using BrAPI instead of yfinance
- No rate limiting
- Near real-time prices
- Batch quotes for efficiency
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
from typing import Dict, List
import json
from pathlib import Path
import time
import concurrent.futures

from src.signals.trend_detector_v2 import TrendDetectorV2
from src.news.free_news_client import FreeNewsClient
from src.features.feature_engineering import get_feature_engineer
from src.fundamentals.integration import FundamentalIntegrator, format_integrated_signal
from src.data.brapi_client import BrAPIClient


def load_tickers() -> List[str]:
    """Load validated tickers from JSON file."""
    config_path = Path(__file__).parent / 'data' / 'validated_tickers.json'
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            data = json.load(f)
        return data.get('all_tickers', [])
    else:
        print("⚠️ validated_tickers.json not found, using fallback list")
        return ["PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3"]


def load_api_key() -> str:
    """Load BrAPI key from config file."""
    # Try environment variable first
    api_key = os.environ.get('BRAPI_API_KEY')
    
    if not api_key:
        # Try config file
        config_path = Path(__file__).parent / 'data' / 'brapi_config.yaml'
        if config_path.exists():
            with open(config_path, 'r') as f:
                for line in f:
                    if 'brapi_api_key:' in line:
                        api_key = line.split(':')[1].strip()
                        break
    
    if not api_key:
        # Try .env file
        env_path = Path(__file__).parent / '.env'
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    if 'BRAPI_API_KEY=' in line:
                        api_key = line.split('=')[1].strip()
                        break
    
    return api_key


class BrAPIProductionRunner:
    """Production runner using BrAPI for data fetching"""
    
    _model_cache = {}
    
    def __init__(self, use_news: bool = True, use_fundamentals: bool = True, 
                 n_workers: int = 4, batch_size: int = 20):
        """
        Initialize production runner with BrAPI.
        
        Args:
            use_news: Enable news sentiment analysis
            use_fundamentals: Enable fundamental analysis
            n_workers: Number of parallel workers for historical data
            batch_size: Batch size for quote requests (default 20 for free tier)
        """
        api_key = load_api_key()
        self.brapi = BrAPIClient(api_key=api_key)
        self.trend_detector = TrendDetectorV2()
        self.use_news = use_news
        self.use_fundamentals = use_fundamentals
        self.n_workers = n_workers
        self.batch_size = batch_size
        
        if use_news:
            if 'news_client' not in self._model_cache:
                self._model_cache['news_client'] = FreeNewsClient()
            self.news_client = self._model_cache['news_client']
        
        if use_fundamentals:
            self.fundamental_integrator = FundamentalIntegrator()
        
        print(f"✅ Sistema inicializado com BrAPI (news={'ON' if use_news else 'OFF'}, "
              f"fundamentals={'ON' if use_fundamentals else 'OFF'}, workers={n_workers})")
    
    def get_historical_data(self, ticker: str, days: int = 120) -> pd.DataFrame:
        """
        Get historical OHLCV data from BrAPI.
        
        Args:
            ticker: Stock ticker (without .SA suffix)
            days: Number of days of history
            
        Returns:
            DataFrame with OHLCV data indexed by date
        """
        # Map days to BrAPI range
        range_map = {
            30: "1mo",
            60: "2mo", 
            90: "3mo",
            120: "3mo",  # BrAPI doesn't have 4mo, use 3mo
            180: "6mo",
            365: "1y"
        }
        range_ = range_map.get(min(days, 365), "3mo")
        
        return self.brapi.get_historical(ticker, range_=range_)
    
    def get_batch_prices(self, tickers: List[str]) -> Dict[str, Dict]:
        """
        Get current prices for multiple tickers in batches.
        
        Uses batch API for efficiency - no rate limiting!
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dict mapping ticker to quote data
        """
        return self.brapi.get_quotes(tickers, use_cache=True)
    
    def calculate_kelly_position(self, data: pd.DataFrame, confidence: float) -> float:
        """Calculate position size using Kelly Criterion."""
        from src.config import get_config
        config = get_config()
        
        try:
            if len(data) < 30:
                return config.DEFAULT_POSITION_SIZE
            
            # Calculate returns
            returns = data['Close'].pct_change().dropna()
            
            if len(returns) < 20:
                return config.DEFAULT_POSITION_SIZE
            
            # Calculate win rate and payoff ratio
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            
            if len(negative_returns) == 0 or len(positive_returns) == 0:
                return config.DEFAULT_POSITION_SIZE
            
            win_rate = len(positive_returns) / len(returns)
            avg_win = positive_returns.mean()
            avg_loss = abs(negative_returns.mean())
            
            if avg_loss == 0:
                return config.DEFAULT_POSITION_SIZE
            
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - p
            
            kelly = (b * p - q) / b
            
            if kelly <= 0:
                return config.MIN_POSITION_SIZE
            
            position = kelly * config.KELLY_FRACTION * confidence
            
            return max(config.MIN_POSITION_SIZE, min(config.MAX_POSITION_SIZE, position))
        
        except Exception:
            return config.DEFAULT_POSITION_SIZE
    
    def get_news_sentiment(self, ticker: str) -> dict:
        """Get sentiment for today."""
        if not self.use_news:
            return {"sentiment": 0.0, "articles": [], "dates": []}
        
        try:
            date = datetime.now().strftime("%Y-%m-%d")
            sentiment = self.news_client.get_sentiment(ticker, date)
            return {
                "sentiment": sentiment,
                "articles": [],
                "dates": [date],
                "daily_scores": {date: sentiment}
            }
        except Exception as e:
            print(f"    ⚠️ News error: {e}")
            return {"sentiment": 0.0, "articles": [], "dates": []}
    
    def analyze_ticker(self, ticker: str, quote_data: Dict = None) -> Dict:
        """
        Analyze single ticker using BrAPI data.
        
        Args:
            ticker: Stock ticker (without .SA suffix)
            quote_data: Pre-fetched quote data (for batch optimization)
        """
        try:
            print(f"📊 {ticker}...", end=" ")
            
            # Get historical data for technical analysis
            data = self.get_historical_data(ticker, days=120)
            
            if data is None or len(data) < 50:
                print(f"⚠️ Insufficient data ({len(data) if data is not None else 0} days)")
                return None
            
            # Get current price from quote or data
            if quote_data:
                current_price = quote_data.get('regularMarketPrice', data['Close'].iloc[-1])
            else:
                current_price = data['Close'].iloc[-1]
            
            # Trend detection
            trend_result = self.trend_detector.detect_trend(data)
            consensus = trend_result.get('consensus', 'unknown')
            confidence = trend_result.get('confidence', 0.0)
            
            # Feature engineering
            feature_engineer = get_feature_engineer()
            features = feature_engineer.generate_all_features(ticker, data)
            
            # Adjust confidence based on features
            volume_features = features.get('volume', {})
            volatility_features = features.get('volatility', {})
            
            if volume_features.get('unusual_volume', False):
                confidence *= 1.05
            
            vol_regime = volatility_features.get('regime', 'medium')
            if vol_regime == 'high':
                confidence *= 0.90
            
            # Map consensus to trend
            if consensus in ['uptrend', 'bull_pullback']:
                trend = "uptrend"
            elif consensus in ['downtrend', 'bear_bounce']:
                trend = "downtrend"
            else:
                trend = "consolidation"
            
            print(f"[TREND] {trend} @ {confidence:.1%}", end=" ")
            
            # News sentiment
            news_result = self.get_news_sentiment(ticker)
            news_sentiment = news_result.get('sentiment', 0.0)
            print(f"[NEWS] sentiment: {news_sentiment:+.2f}", end=" ")
            
            # Fundamental analysis
            fundamentals = None
            if self.use_fundamentals:
                fund_score_data = self.fundamental_integrator.get_fundamental_score(ticker)
                if fund_score_data:
                    fundamentals = fund_score_data.copy()
                    
                    # Load raw metrics from cache
                    raw_cache_path = Path(__file__).parent / 'data' / 'fundamentals' / 'fundamentals_cache.json'
                    if raw_cache_path.exists():
                        try:
                            with open(raw_cache_path, 'r') as f:
                                raw_cache = json.load(f)
                            raw_data = raw_cache.get('data', {}).get(ticker, {})
                            if raw_data:
                                # Add raw metrics to fundamentals
                                fundamentals['pe_ratio'] = raw_data.get('pe_ratio')
                                fundamentals['pb_ratio'] = raw_data.get('pb_ratio')
                                fundamentals['roe'] = raw_data.get('roe')
                                fundamentals['roic'] = raw_data.get('roic')
                                fundamentals['div_yield'] = raw_data.get('div_yield')
                                fundamentals['debt_equity'] = raw_data.get('debt_equity')
                                fundamentals['fundamental_grade'] = fundamentals.get('grade', '')
                                fundamentals['is_value_pick'] = fundamentals.get('is_value', False)
                                fundamentals['is_quality_pick'] = fundamentals.get('is_quality', False)
                                fundamentals['is_momentum_pick'] = fundamentals.get('is_growth', False)
                        except Exception as e:
                            print(f"    ⚠️ Could not load raw fundamentals: {e}")
            
            # Generate signal
            result = self._generate_signal(
                ticker=ticker,
                price=current_price,
                trend=trend,
                trend_confidence=confidence,
                news_sentiment=news_sentiment,
                features=features,
                fundamentals=fundamentals
            )
            
            # Print signal
            signal = result.get('signal', 'HOLD')
            print(f"{signal} ({trend})")
            
            return result
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def _generate_signal(self, ticker: str, price: float, trend: str,
                         trend_confidence: float, news_sentiment: float,
                         features: Dict, fundamentals: Dict) -> Dict:
        """Generate trading signal combining all factors."""
        from src.config import get_config
        config = get_config()
        
        # Calculate composite score
        tech_score = trend_confidence * 100
        
        # Adjust for news
        news_adj = news_sentiment * 20  # Scale news to 0-20 range
        tech_score = max(0, min(100, tech_score + news_adj))
        
        # Fundamental score
        fund_score = 0
        if fundamentals:
            fund_score = fundamentals.get('composite_score', 50)
        
        # Combined score (50/50 weight)
        composite = (tech_score * 0.5) + (fund_score * 0.5)
        
        # Generate signal
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
        
        # Avoid flag
        is_avoid = False
        if fundamentals:
            is_avoid = fundamentals.get('is_avoid', False)
            if is_avoid:
                signal = "AVOID"
        
        # Position sizing
        position_size = self.calculate_kelly_position(
            pd.DataFrame(),  # Will use default if no data
            trend_confidence
        )
        
        # Adjust position for fundamentals
        if fundamentals and not is_avoid:
            if fundamentals.get('is_quality_pick'):
                position_size *= 1.2
            if fundamentals.get('is_value_pick'):
                position_size *= 1.15
            if fund_score < 40:
                position_size *= 0.5
        
        return {
            'ticker': ticker,
            'price': price,
            'signal': signal,
            'trend': trend,
            'confidence': trend_confidence,
            'news_sentiment': news_sentiment,
            'composite_score': composite,
            'tech_score': tech_score,
            'fund_score': fund_score,
            'position_size': position_size,
            'features': features,
            'fundamentals': fundamentals,
            'is_avoid': is_avoid
        }
    
    def run(self) -> List[Dict]:
        """Run full analysis on all tickers."""
        tickers = load_tickers()
        
        print(f"\n{'='*70}")
        print(f"🚀 PRODUÇÃO (BrAPI) - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"📊 Analyzing {len(tickers)} tickers (IBOV + SMLL)")
        print(f"{'='*70}\n")
        
        # Step 1: Batch fetch all prices first (super fast!)
        print("⚡ Fetching batch prices...")
        start_time = time.time()
        
        all_quotes = {}
        for i in range(0, len(tickers), self.batch_size):
            batch = tickers[i:i+self.batch_size]
            quotes = self.get_batch_prices(batch)
            all_quotes.update(quotes)
            print(f"   Batch {i//self.batch_size + 1}: {len(quotes)} quotes")
        
        price_time = time.time() - start_time
        print(f"   ✅ Got {len(all_quotes)} prices in {price_time:.1f}s")
        print()
        
        # Step 2: Analyze each ticker (historical data + technicals)
        results = []
        
        # Use ThreadPoolExecutor for parallel analysis
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {}
            for ticker in tickers:
                quote_data = all_quotes.get(ticker)
                if quote_data:
                    future = executor.submit(self.analyze_ticker, ticker, quote_data)
                    futures[future] = ticker
            
            for future in concurrent.futures.as_completed(futures):
                ticker = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"   ❌ {ticker}: {e}")
        
        # Sort by composite score
        results.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
        
        # Print summary
        print(f"\n{'='*70}")
        signals = [r['signal'] for r in results]
        print(f"📊 Summary: {signals.count('STRONG_BUY')} STRONG_BUY | "
              f"{signals.count('BUY')} BUY | {signals.count('HOLD')} HOLD | "
              f"{signals.count('SELL')} SELL | {signals.count('STRONG_SELL')} STRONG_SELL")
        print(f"{'='*70}\n")
        
        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Production analysis with BrAPI')
    parser.add_argument('--no-news', action='store_true', help='Disable news analysis')
    parser.add_argument('--no-fundamentals', action='store_true', help='Disable fundamentals')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    args = parser.parse_args()
    
    runner = BrAPIProductionRunner(
        use_news=not args.no_news,
        use_fundamentals=not args.no_fundamentals,
        n_workers=args.workers
    )
    
    results = runner.run()
    
    # Save results
    output_path = Path(__file__).parent / 'data' / 'full_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {output_path}")