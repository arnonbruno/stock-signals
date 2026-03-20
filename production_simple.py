#!/usr/bin/env python3
"""
Production Runner with Integrated Fundamental Analysis
Generates buy/sell signals combining technical analysis + value investing metrics
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
from typing import Dict, List
import json
from pathlib import Path

from src.signals.trend_detector_v2 import TrendDetectorV2
from src.news.free_news_client import FreeNewsClient
from src.features.feature_engineering import get_feature_engineer
from src.fundamentals.integration import FundamentalIntegrator, format_integrated_signal


def load_tickers() -> List[str]:
    """Load validated tickers from JSON file."""
    config_path = Path(__file__).parent / 'data' / 'validated_tickers.json'
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            data = json.load(f)
        tickers = data.get('all_tickers', [])
        # Add .SA suffix for Yahoo Finance
        return [f"{t}.SA" if not t.endswith('.SA') else t for t in tickers]
    else:
        # Fallback to hardcoded list
        print("⚠️ validated_tickers.json not found, using fallback list")
        return IBOV_FALLBACK + SMLL_FALLBACK


# Fallback tickers (used only if JSON file missing)
IBOV_FALLBACK = [
    'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'BBAS3.SA', 'ABEV3.SA',
    'B3SA3.SA', 'SUZB3.SA', 'RENT3.SA', 'WEGE3.SA'
]

SMLL_FALLBACK = [
    'CURY3.SA', 'EZTC3.SA', 'JHSF3.SA', 'CYRE3.SA', 'MRVE3.SA'
]

from multiprocessing import Pool, cpu_count
import functools


class SimpleProductionRunner:
    """Production runner with integrated fundamental + technical analysis"""
    
    # Class-level model cache (shared across instances)
    _model_cache = {}
    
    def __init__(self, use_news: bool = True, use_fundamentals: bool = True, n_workers: int = None):
        self.trend_detector = TrendDetectorV2()
        self.use_news = use_news
        self.use_fundamentals = use_fundamentals
        # Cap workers at 8 to prevent resource exhaustion
        max_workers = min(cpu_count(), 8)
        self.n_workers = min(n_workers, max_workers) if n_workers else max_workers
        
        if use_news:
            # Check if model already cached
            if 'news_client' not in self._model_cache:
                self._model_cache['news_client'] = FreeNewsClient()
            self.news_client = self._model_cache['news_client']
        
        if use_fundamentals:
            self.fundamental_integrator = FundamentalIntegrator()
        
        print(f"✅ Sistema inicializado (news={'ON' if use_news else 'OFF'}, "
              f"fundamentals={'ON' if use_fundamentals else 'OFF'}, workers={self.n_workers})")
    
    def calculate_kelly_position(self, data: pd.DataFrame, confidence: float) -> float:
        """
        Calculate position size using TRUE Kelly Criterion with confidence adjustment.
        
        Kelly Formula: f* = (bp - q) / b
        where: b = average win / average loss (payoff ratio/odds)
               p = win probability (win rate)
               q = 1 - p (loss probability)
        
        Uses Half-Kelly for safety: position = 0.5 * kelly * confidence
        
        Args:
            data: Price data DataFrame with 'Close' column
            confidence: Signal confidence from trend detection (0-1)
        
        Returns:
            Position size as fraction of portfolio (0.10 to 0.60)
        """
        from src.config import get_config
        config = get_config()
        
        try:
            # Calculate returns from close prices
            close_prices = data['Close'].squeeze()
            returns = close_prices.pct_change().dropna()
            
            # Need minimum data points for statistical significance
            if len(returns) < config.KELLY_MIN_DATA_POINTS:
                return config.DEFAULT_POSITION_SIZE
            
            # Separate positive and negative returns
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            
            # Need both winning and losing trades to calculate Kelly
            if len(positive_returns) == 0 or len(negative_returns) == 0:
                return config.DEFAULT_POSITION_SIZE
            
            # Calculate Kelly parameters
            p = len(positive_returns) / len(returns)  # Win probability
            q = 1 - p  # Loss probability
            
            avg_win = positive_returns.mean()  # Average winning return
            avg_loss = abs(negative_returns.mean())  # Average losing return (absolute)
            
            # Payoff ratio (odds) - how much we win vs how much we lose
            if avg_loss == 0:
                return config.DEFAULT_POSITION_SIZE
            
            b = avg_win / avg_loss  # Payoff ratio
            
            # Kelly formula: f* = (bp - q) / b
            # This gives the optimal fraction of capital to risk
            kelly = (b * p - q) / b
            
            # Kelly can be negative if edge is negative (don't trade)
            # or > 1 if edge is very high (cap at reasonable levels)
            if kelly <= 0:
                return config.MIN_POSITION_SIZE
            
            # Apply Half-Kelly for safety (reduces volatility and drawdowns)
            # Also scale by confidence from signal quality
            position = kelly * config.KELLY_FRACTION * confidence
            
            # Enforce bounds
            return max(config.MIN_POSITION_SIZE, min(config.MAX_POSITION_SIZE, position))
        
        except Exception as e:
            return config.DEFAULT_POSITION_SIZE
    
    def get_data(self, ticker: str, days: int = 120, max_retries: int = 3, 
                 end_date: datetime = None) -> pd.DataFrame:
        """
        Download recent data with error handling and retries.
        
        Implements exponential backoff for network failures.
        Returns None if all retries fail.
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days of historical data
            max_retries: Maximum retry attempts
            end_date: Optional end date for backtesting (prevents look-ahead bias)
        """
        end_date = end_date or datetime.now()
        start_date = end_date - timedelta(days=days)
        
        for attempt in range(max_retries):
            try:
                data = yf.download(
                    ticker,
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    progress=False
                )
                
                if data.empty:
                    print(f"    ⚠️ No data returned for {ticker}")
                    return None
                
                return data
            
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 1.0  # Exponential backoff: 1s, 2s, 4s
                    print(f"    ⚠️ Attempt {attempt + 1}/{max_retries} failed for {ticker}: {e}")
                    print(f"       Retrying in {wait_time}s...")
                    import time
                    time.sleep(wait_time)
                else:
                    print(f"    ❌ All retries exhausted for {ticker}: {e}")
                    return None
        
        return None
    
    def get_news_sentiment(self, ticker: str) -> dict:
        """Get sentiment for today only (newsdata.io + cache)"""
        if not self.use_news:
            return {"sentiment": 0.0, "articles": [], "dates": []}
        
        try:
            # Fetch sentiment for TODAY ONLY using FreeNewsClient
            # (which handles newsdata.io + Investing.com fallback + caching)
            date = datetime.now().strftime("%Y-%m-%d")
            sentiment = self.news_client.get_sentiment(ticker, date)
            
            return {
                "sentiment": sentiment,
                "articles": [],  # Already cached in news_client.cache
                "dates": [date],
                "daily_scores": {date: sentiment}
            }
        except Exception as e:
            print(f"    ⚠️ News error: {e}")
            return {"sentiment": 0.0, "articles": [], "dates": []}
    
    def analyze_ticker(self, ticker: str, data: pd.DataFrame = None, 
                       as_of_date: datetime = None) -> Dict:
        """Analyze single ticker
        
        Args:
            ticker: Stock ticker symbol
            data: Optional pre-loaded DataFrame (for backtesting). If None, downloads fresh data.
            as_of_date: Optional date for backtesting (prevents look-ahead bias by using only historical data)
        """
        try:
            # Download data if not provided (production mode)
            if data is None:
                data = self.get_data(ticker, end_date=as_of_date)
            
            if len(data) < 50:
                return None
            
            # Current price
            price_val = data['Close'].iloc[-1]
            current_price = float(price_val.item()) if hasattr(price_val, 'item') else float(price_val)
            
            # Trend detection (core validated logic)
            trend_result = self.trend_detector.detect_trend(data)
            consensus = trend_result.get('consensus', 'unknown')
            confidence = trend_result.get('confidence', 0.0)
            
            # Feature engineering: volume, volatility, sector features
            feature_engineer = get_feature_engineer()
            features = feature_engineer.generate_all_features(ticker, data)
            
            # Adjust confidence based on feature signals
            volume_features = features.get('volume', {})
            volatility_features = features.get('volatility', {})
            
            # Volume confirmation boost
            if volume_features.get('unusual_volume', False):
                confidence *= 1.05  # 5% boost for unusual volume
                print(f"     [VOL] Unusual volume detected (+5% confidence)")
            
            # Volatility regime adjustment
            vol_regime = volatility_features.get('regime', 'medium')
            if vol_regime == 'high':
                confidence *= 0.90  # Reduce confidence in high volatility
                print(f"     [VOL] High volatility regime (-10% confidence)")
            
            # Store features for reporting
            feature_summary = {
                'volume_momentum': volume_features.get('volume_momentum', 1.0),
                'unusual_volume': volume_features.get('unusual_volume', False),
                'volatility_regime': vol_regime,
                'vix_equivalent': volatility_features.get('vix_equivalent', 20.0)
            }
            
            # Print trend details
            print(f"\n     [TREND] {consensus} @ {confidence:.1%} confidence")
            
            # Map consensus to simple trend (UPPERCASE to match integration.py expectations)
            if consensus in ['uptrend', 'bull_pullback']:
                trend = "UPTREND"
            elif consensus in ['downtrend', 'bear_bounce']:
                trend = "DOWNTREND"
            else:
                trend = "SIDEWAYS"
            
            # News sentiment (if enabled)
            if self.use_news:
                print(f"     [NEWS] newsdata.io...", end=" ", flush=True)
            
            news_data = self.get_news_sentiment(ticker)
            news_sentiment = news_data.get("sentiment", 0.0)
            news_articles = news_data.get("articles", [])
            
            if self.use_news and news_articles:
                print(f"✅ {len(news_articles)} articles, sentiment: {news_sentiment:+.2f}")
            elif self.use_news:
                print(f"⚠️  No articles found, sentiment: {news_sentiment:+.2f}")
            
            from src.config import get_config
            config = get_config()
            
            signal = "HOLD"
            position_size = 0.0
            conviction = 0.0
            technical_score = confidence * 100  # Convert to 0-100 scale
            
            # Regime-aware thresholds: map volatility regime to market regime
            regime_map = {'low': 'bull', 'medium': 'sideways', 'high': 'bear'}
            market_regime = regime_map.get(vol_regime, 'default')
            thresholds = config.get_thresholds(market_regime)
            MIN_CONFIDENCE = thresholds.buy_confidence
            SELL_CONFIDENCE = thresholds.sell_confidence
            
            print(f"     [REGIME] {market_regime} -> buy_conf={MIN_CONFIDENCE:.2f}, sell_conf={SELL_CONFIDENCE:.2f}")
            
            if trend == "UPTREND" and confidence >= MIN_CONFIDENCE:
                signal = "BUY"
                conviction = confidence
                
                # Kelly Criterion position sizing (volatility-adjusted)
                position_size = self.calculate_kelly_position(data, confidence)
                
                # News boost: Sigmoid scaling (SOTA approach)
                # Uses logistic function to prevent over-amplification
                if self.use_news and abs(news_sentiment) > 0.1:
                    # Sigmoid scaling: maps sentiment to 0.5-1.5 range
                    # This provides multiplicative boost instead of additive
                    sigmoid_boost = 1 / (1 + np.exp(-5 * news_sentiment))  # 0.5 to 1.0
                    position_size *= sigmoid_boost  # Multiplicative scaling
                    
                    # Update conviction based on news agreement with trend
                    if (news_sentiment > 0 and trend == "UPTREND") or (news_sentiment < 0 and trend == "DOWNTREND"):
                        conviction = min(1.0, conviction * 1.1)  # 10% boost when aligned
                
                # Cap at 80% for safety
                position_size = min(0.80, position_size)
                    
            elif trend == "DOWNTREND" and confidence >= SELL_CONFIDENCE:
                signal = "SELL"
                conviction = -confidence
                position_size = 1.0  # Exit completely
            
            # Integrate fundamental analysis
            fundamental_data = None
            if self.use_fundamentals:
                integrated_score = self.fundamental_integrator.integrate(
                    ticker=ticker,
                    technical_score=technical_score,
                    trend=trend,
                    confidence=confidence
                )
                
                # Override signal based on integrated analysis
                signal = integrated_score.recommendation
                conviction = integrated_score.composite_score / 100  # Normalize to 0-1
                
                # Adjust position size based on fundamentals
                if signal in ["BUY", "STRONG_BUY"]:
                    # High quality stocks can get larger positions
                    if integrated_score.is_quality_pick:
                        position_size = min(0.80, position_size * 1.2)
                    
                    # Value picks get moderate positions (contrarian)
                    if integrated_score.is_value_pick:
                        position_size = max(0.15, min(0.50, position_size))
                    
                    # Poor fundamentals reduce position size
                    if integrated_score.fundamental_score < 40:
                        position_size *= 0.5
                    
                    # Avoid flag blocks trading entirely
                    if integrated_score.is_avoid:
                        signal = "HOLD"
                        position_size = 0.0
                
                fundamental_data = {
                    'composite_score': integrated_score.composite_score,
                    'fundamental_grade': integrated_score.fundamental_grade,
                    'value_score': integrated_score.value_score,
                    'quality_score': integrated_score.quality_score,
                    'is_value_pick': integrated_score.is_value_pick,
                    'is_quality_pick': integrated_score.is_quality_pick,
                    'is_momentum_pick': integrated_score.is_momentum_pick,
                    'is_avoid': integrated_score.is_avoid,
                    'pe_ratio': integrated_score.pe_ratio,
                    'pb_ratio': integrated_score.pb_ratio,
                    'roe': integrated_score.roe,
                    'div_yield': integrated_score.div_yield,
                    'strengths': integrated_score.strengths,
                    'weaknesses': integrated_score.weaknesses,
                    'action_notes': integrated_score.action_notes,
                }
            
            return {
                "ticker": ticker,
                "price": current_price,
                "trend": trend,
                "confidence": confidence,  # Add confidence from trend detector
                "news_sentiment": news_sentiment,
                "news_articles": news_articles,
                "signal": signal,
                "conviction": conviction,
                "position_size": position_size,
                "features": feature_summary,  # Add feature engineering data
                "fundamentals": fundamental_data,  # Add fundamental data
            }
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return None
    
    def get_top_movers(self, results: List[Dict], top_n: int = 10) -> List[str]:
        """
        Identify top movers from analysis results.
        
        Used for smart news fetching: fetch fresh news only for high-conviction signals.
        
        Args:
            results: List of analysis results
            top_n: Number of top movers to return
        
        Returns:
            List of top mover tickers
        """
        if not results:
            return []
        
        # Sort by absolute conviction (both BUY and SELL signals matter)
        sorted_results = sorted(results, key=lambda x: abs(x.get('conviction', 0)), reverse=True)
        
        # Get top N tickers
        top_movers = [r['ticker'] for r in sorted_results[:top_n]]
        
        print(f"\n📊 Top {top_n} Movers (for smart news refresh):")
        for i, ticker in enumerate(top_movers, 1):
            result = next(r for r in sorted_results if r['ticker'] == ticker)
            print(f"   [{i}] {ticker} - {result['signal']} ({result['trend']}, conviction: {result['conviction']:.2f})")
        
        return top_movers
    
    def _analyze_ticker_wrapper(self, ticker: str) -> Dict:
        """Wrapper for parallel processing (needed for Pool.map)."""
        print(f"📊 {ticker}...", end=" ", flush=True)
        result = self.analyze_ticker(ticker)
        if result:
            print(f"{result['signal']} ({result['trend']})")
        else:
            print("SKIP")
        return result
    
    def run(self, tickers: List[str] = None, parallel: bool = True):
        """
        Run analysis on all tickers.
        
        Args:
            tickers: List of tickers to analyze (default: all from validated_tickers.json)
            parallel: Use parallel processing (default: True, ~8x faster)
        """
        if tickers is None:
            tickers = load_tickers()
        
        print(f"\n{'='*70}")
        print(f"🚀 PRODUÇÃO - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"📊 Analyzing {len(tickers)} tickers (IBOV + SMLL)")
        print(f"{'='*70}\n")
        
        results = []
        
        if parallel and len(tickers) > 1:
            # Parallel processing (8x faster for 150 tickers)
            print(f"⚡ Parallel mode: {self.n_workers} workers\n")
            
            with Pool(self.n_workers) as pool:
                raw_results = pool.map(self._analyze_ticker_wrapper, tickers)
            
            results = [r for r in raw_results if r is not None]
        else:
            # Sequential processing (for debugging or single ticker)
            for ticker in tickers:
                print(f"📊 {ticker}...", end=" ")
                result = self.analyze_ticker(ticker)
                if result:
                    print(f"{result['signal']} ({result['trend']})")
                    results.append(result)
                else:
                    print("SKIP")
        
        # Summary
        self._print_summary(results)
        
        # Zero-signal monitoring: alert if no BUY/SELL signals generated
        buy_signals = [r for r in results if r['signal'] in ('BUY', 'STRONG_BUY')]
        sell_signals = [r for r in results if r['signal'] in ('SELL', 'STRONG_SELL')]
        
        if results and not buy_signals and not sell_signals:
            print("\n" + "!" * 70)
            print("🚨 ZERO-SIGNAL ALERT: No BUY or SELL signals generated!")
            print(f"   Analyzed {len(results)} tickers, all returned HOLD.")
            print("   Possible causes:")
            print("   - Confidence thresholds too high for current regime")
            print("   - Trend/case mismatch between modules")
            print("   - Market in low-conviction sideways regime")
            print("!" * 70 + "\n")
        elif results and not buy_signals:
            print(f"\n⚠️  MONITORING: 0 BUY signals out of {len(results)} tickers. "
                  f"({len(sell_signals)} SELL)")
        
        return results
    
    def _print_summary(self, results: List[Dict]):
        """Print final table with fundamental data"""
        if not results:
            print("\n❌ No results")
            return
        
        # Sort by conviction
        results_sorted = sorted(results, key=lambda x: abs(x['conviction']), reverse=True)
        
        print(f"\n{'='*70}")
        print(f"📊 RESUMO - {len(results)} tickers")
        print(f"{'='*70}\n")
        
        # Header
        if self.use_fundamentals:
            print(f"{'Ticker':<10} {'Preço':>8} {'Sinal':<8} {'Grade':>4} {'Val':>4} {'Qual':>4} {'Pos%':>5}")
        else:
            print(f"{'Ticker':<10} {'Preço':>8} {'Sinal':<6} {'Trend':<10} {'News':>6} {'Pos%':>5}")
        print(f"{'-'*70}")
        
        # Rows
        for r in results_sorted:
            signal = r['signal']
            signal_emoji = {
                "STRONG_BUY": "🚀",
                "BUY": "🟢",
                "HOLD": "🟡",
                "SELL": "🔴",
                "STRONG_SELL": "💀"
            }.get(signal, "⚪")
            
            pos_pct = f"{r['position_size']*100:.0f}%" if r['position_size'] > 0 else "-"
            
            # Fundamental indicators
            fund = r.get('fundamentals', {})
            if self.use_fundamentals and fund:
                grade = fund.get('fundamental_grade', 'N/A')
                value = f"{fund.get('value_score', 0):.0f}" if fund.get('value_score') else "-"
                quality = f"{fund.get('quality_score', 0):.0f}" if fund.get('quality_score') else "-"
                
                # Add fundamental indicators
                fund_indicators = []
                if fund.get('is_value_pick'):
                    fund_indicators.append('💰')
                if fund.get('is_quality_pick'):
                    fund_indicators.append('⭐')
                if fund.get('is_momentum_pick'):
                    fund_indicators.append('📈')
                if fund.get('is_avoid'):
                    fund_indicators.append('⚠️')
                fund_str = ''.join(fund_indicators)
                
                print(
                    f"{r['ticker']:<10} "
                    f"R${r['price']:>7.2f} "
                    f"{signal_emoji} {signal:<6} "
                    f"{grade:>4} "
                    f"{value:>4} "
                    f"{quality:>4} "
                    f"{pos_pct:>5} "
                    f"{fund_str}"
                )
            else:
                # Original format without fundamentals
                news_str = f"{r['news_sentiment']:+.2f}" if self.use_news else "N/A"
                
                features = r.get('features', {})
                feature_indicators = []
                if features.get('unusual_volume'):
                    feature_indicators.append('📈')
                if features.get('volatility_regime') == 'high':
                    feature_indicators.append('⚡')
                feature_str = ''.join(feature_indicators) if feature_indicators else ''
                
                print(
                    f"{r['ticker']:<10} "
                    f"R${r['price']:>7.2f} "
                    f"{signal_emoji} {signal:<4} "
                    f"{r['trend']:<10} "
                    f"{news_str:>6} "
                    f"{pos_pct:>5} "
                    f"{feature_str}"
                )
        
        # Stats
        buy = [r for r in results if r['signal'] == "BUY"]
        sell = [r for r in results if r['signal'] == "SELL"]
        
        print(f"\n{'-'*70}")
        print(f"🟢 BUY: {len(buy)} | 🔴 SELL: {len(sell)} | ⚪ HOLD: {len(results) - len(buy) - len(sell)}")
        print(f"{'='*70}\n")
        
        # News details (if enabled)
        if self.use_news:
            print(f"\n{'='*70}")
            print(f"📰 NEWS ANALYSIS DETAILS")
            print(f"{'='*70}\n")
            
            # Show fundamental details for top signals
            if self.use_fundamentals:
                print(f"\n{'='*70}")
                print(f"📊 FUNDAMENTAL ANALYSIS - TOP SIGNALS")
                print(f"{'='*70}\n")
                
                # Show top 5 BUY signals
                buy_signals = [r for r in results_sorted if r['signal'] in ['BUY', 'STRONG_BUY']][:5]
                if buy_signals:
                    print("🟢 TOP BUY SIGNALS:\n")
                    for r in buy_signals:
                        fund = r.get('fundamentals', {})
                        if fund:
                            print(f"  {r['ticker']} - {r['signal']} @ R${r['price']:.2f}")
                            print(f"    Grade: {fund.get('fundamental_grade')} | "
                                  f"Value: {fund.get('value_score', 0):.0f} | "
                                  f"Quality: {fund.get('quality_score', 0):.0f}")
                            if fund.get('pe_ratio'):
                                print(f"    P/E: {fund['pe_ratio']:.1f} | "
                                      f"P/B: {fund.get('pb_ratio', 0):.2f} | "
                                      f"ROE: {fund.get('roe', 0):.1f}% | "
                                      f"Div: {fund.get('div_yield', 0):.1f}%")
                            if fund.get('strengths'):
                                print(f"    ✅ {', '.join(fund['strengths'][:2])}")
                            if fund.get('action_notes'):
                                for note in fund['action_notes'][:2]:
                                    print(f"    {note}")
                            print()
                
                # Show AVOID stocks
                avoid_stocks = [r for r in results_sorted if r.get('fundamentals', {}).get('is_avoid')]
                if avoid_stocks:
                    print("⚠️ AVOID (Poor fundamentals + Poor technicals):\n")
                    for r in avoid_stocks[:5]:
                        fund = r.get('fundamentals', {})
                        print(f"  {r['ticker']} - {fund.get('fundamental_grade')} grade, "
                              f"Value: {fund.get('value_score', 0):.0f}, Quality: {fund.get('quality_score', 0):.0f}")
                        if fund.get('weaknesses'):
                            print(f"    ❌ {', '.join(fund['weaknesses'][:2])}")
                    print()
            
            # News details (if enabled)
            if self.use_news:
                print(f"\n{'='*70}")
                print(f"📰 NEWS ANALYSIS DETAILS")
                print(f"{'='*70}\n")
                
                for r in results_sorted[:10]:  # Top 10 only
                    articles = r.get('news_articles', [])
                    sentiment = r.get('news_sentiment', 0.0)
                    
                    if articles:
                        print(f"📊 {r['ticker']} - Sentiment: {sentiment:+.2f} ({len(articles)} articles)")
                        for i, article in enumerate(articles[:3], 1):  # Show top 3 articles
                            title = article.get('title', 'No title')[:70]
                            art_sentiment = article.get('sentiment', 0.0)
                            date = article.get('date', 'N/A')
                            print(f"   [{i}] ({art_sentiment:+.2f}) {title}...")
                            print(f"       Date: {date} | Source: {article.get('source', 'unknown')}")
                        if len(articles) > 3:
                            print(f"   ... and {len(articles) - 3} more articles")
                    else:
                        print(f"📊 {r['ticker']} - No news found (Sentiment: {sentiment:+.2f})")
                    print()


def main():
    parser = argparse.ArgumentParser(description="Production Runner with Fundamental Analysis")
    parser.add_argument("--ticker", type=str, help="Single ticker (ex: PETR4.SA)")
    parser.add_argument("--no-news", action="store_true", help="Disable news (faster)")
    parser.add_argument("--no-fundamentals", action="store_true", help="Disable fundamentals (technical only)")
    
    args = parser.parse_args()
    
    runner = SimpleProductionRunner(
        use_news=not args.no_news,
        use_fundamentals=not args.no_fundamentals
    )
    
    if args.ticker:
        runner.run(tickers=[args.ticker])
    else:
        runner.run()


if __name__ == "__main__":
    main()
