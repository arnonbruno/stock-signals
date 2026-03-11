#!/usr/bin/env python3
"""
Enhanced Production Runner - SOTA Adaptive Trading System

Integrates:
1. Multi-indicator analysis (momentum, volatility, volume, trend)
2. Signal fusion with regime-aware weighting
3. Market regime detection (bull/bear/sideways)
4. Risk parity position sizing
5. Historical sentiment for backtesting

This is the production-ready SOTA version.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Core components
from src.signals.trend_detector_v2 import TrendDetectorV2
from src.news.free_news_client import FreeNewsClient
from src.features.feature_engineering import get_feature_engineer
from src.strategy.regime_detection import (
    get_regime_detector, 
    get_adaptive_params,
    calculate_price_based_sentiment
)
from src.risk.risk_parity import RiskParity

# New SOTA indicators
from src.indicators.momentum import MomentumIndicators
from src.indicators.volatility import VolatilityIndicators
from src.indicators.volume import VolumeIndicators
from src.indicators.trend import TrendIndicators
from src.indicators.signal_fusion import SignalFusion, fuse_all_signals

# Historical sentiment
from src.news.historical_sentiment import get_backtest_sentiment

# Config
from src.config import get_config, get_thresholds, ThresholdConfig

# Ticker lists
IBOV_TICKERS = [
    'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'BBAS3.SA', 'ABEV3.SA',
    'B3SA3.SA', 'SUZB3.SA', 'RENT3.SA', 'WEGE3.SA', 'MGLU3.SA', 'PCAR3.SA',
    'LREN3.SA', 'RAIZ4.SA', 'GGBR4.SA', 'ASAI3.SA', 'RDOR3.SA',
    'PETR3.SA', 'ITSA4.SA', 'BBDC3.SA', 'CMIG4.SA', 'ENGI11.SA', 'EQTL3.SA',
    'GGPS3.SA', 'GOAU4.SA', 'HAPV3.SA', 'HYPE3.SA', 'IGTI11.SA',
    'IRBR3.SA', 'KLBN11.SA', 'LWSA3.SA', 'MRVE3.SA', 'MULT3.SA',
    'PRIO3.SA', 'QUAL3.SA', 'RAIL3.SA', 'RADL3.SA',
    'SANB11.SA', 'SBSP3.SA', 'SMTO3.SA', 'TAEE11.SA', 'TIMS3.SA',
    'TOTS3.SA', 'UGPA3.SA', 'USIM5.SA', 'VBBR3.SA', 'VIVT3.SA',
    'YDUQ3.SA', 'AZUL4.SA', 'BPAC11.SA', 'CASH3.SA',
    'COGN3.SA', 'CPFE3.SA', 'CSAN3.SA', 'CVCB3.SA',
    'ECOR3.SA', 'FLRY3.SA', 'RECV3.SA', 'BEEF3.SA', 'CYRE3.SA', 'DXCO3.SA',
    'SLCE3.SA', 'VIVA3.SA', 'ALOS3.SA', 'ALPA4.SA'
]

SMLL_TICKERS = [
    'AURE3.SA', 'BMOB3.SA', 'BRAP4.SA', 'CMIN3.SA', 'DIRR3.SA', 'ESPA3.SA',
    'EVEN3.SA', 'GRND3.SA', 'IFCM3.SA', 'KEPL3.SA', 'LAVV3.SA', 'LEVE3.SA',
    'MDIA3.SA', 'MILS3.SA', 'ODPV3.SA', 'ORVR3.SA', 'POMO4.SA', 'POSI3.SA',
    'PSSA3.SA', 'PTBL3.SA', 'RAPT4.SA', 'SAPR11.SA', 'SEQL3.SA', 'SIMH3.SA',
    'TEND3.SA', 'TGMA3.SA', 'TRIS3.SA', 'UNIP6.SA', 'VLID3.SA',
    'AMBP3.SA', 'AMAR3.SA', 'BMGB4.SA', 'BRKM5.SA', 'CSED3.SA',
    'DESK3.SA', 'EZTC3.SA', 'FESA4.SA', 'GGBR3.SA', 'GMAT3.SA', 'HBOR3.SA',
    'JHSF3.SA', 'JSLG3.SA', 'LIGT3.SA', 'LPSB3.SA', 'MTRE3.SA',
    'ONCO3.SA', 'OPCT3.SA', 'PINE4.SA', 'PRNR3.SA', 'RANI3.SA', 'ROMI3.SA',
    'SEER3.SA', 'SGPS3.SA', 'SOJA3.SA', 'TCSA3.SA',
    'TFCO4.SA', 'TUPY3.SA', 'UCAS3.SA', 'VULC3.SA', 'WIZC3.SA', 'ALUP11.SA',
    'AZZA3.SA', 'BLAU3.SA', 'CEAB3.SA', 'CGRA4.SA', 'CTSA3.SA', 'FHER3.SA',
]

TICKERS = IBOV_TICKERS + SMLL_TICKERS

from multiprocessing import Pool, cpu_count


class EnhancedProductionRunner:
    """
    SOTA Production Runner with full indicator suite and regime adaptation.
    
    Features:
    - Multi-factor signal fusion
    - Market regime detection and adaptation
    - Risk parity position sizing
    - Historical sentiment for backtesting
    """
    
    _model_cache = {}
    
    def __init__(self, use_news: bool = True, n_workers: int = None,
                 use_fusion: bool = True, use_regime: bool = True):
        """
        Initialize enhanced runner.
        
        Args:
            use_news: Enable news sentiment
            n_workers: Number of parallel workers
            use_fusion: Use signal fusion (vs single trend detector)
            use_regime: Enable market regime adaptation
        """
        self.trend_detector = TrendDetectorV2()
        self.use_news = use_news
        # Cap workers at 8 to prevent resource exhaustion
        max_workers = min(cpu_count(), 8)
        self.n_workers = min(n_workers, max_workers) if n_workers else max_workers
        self.use_fusion = use_fusion
        self.use_regime = use_regime
        
        # Regime detection
        if use_regime:
            self.regime_detector = get_regime_detector()
            self.adaptive_params = get_adaptive_params()
            self.current_regime = None
            self.regime_params = None
        else:
            self.regime_detector = None
            self.adaptive_params = None
            self.current_regime = 'default'
            self.regime_params = None
        
        # Risk parity
        self.risk_parity = RiskParity()
        
        # News client
        if use_news:
            if 'news_client' not in self._model_cache:
                self._model_cache['news_client'] = FreeNewsClient()
            self.news_client = self._model_cache['news_client']
        
        # Market data for regime detection
        self.market_data = None
        
        # Thresholds loaded from config
        self.thresholds = get_thresholds('default')
        
        print(f"✅ Enhanced Runner initialized")
        print(f"   Fusion: {'ON' if use_fusion else 'OFF'} | Regime: {'ON' if use_regime else 'OFF'}")
        print(f"   News: {'ON' if use_news else 'OFF'} | Workers: {self.n_workers}")
        print(f"   Buy threshold: {self.thresholds.buy_confidence:.0%} | Sell: {self.thresholds.sell_confidence:.0%}")
    
    def reload_thresholds(self):
        """Reload thresholds from config file."""
        from src.config import reload_config
        reload_config()
        self.thresholds = get_thresholds(self.current_regime or 'default')
        print(f"✅ Thresholds reloaded for {self.current_regime or 'default'} regime")
    
    def fetch_market_data(self, days: int = 365):
        """Fetch IBOV index for regime detection."""
        try:
            self.market_data = yf.download('^BVSP', period=f'{days}d', progress=False)
            print(f"✅ Market data loaded: {len(self.market_data)} days")
        except Exception as e:
            print(f"⚠️ Could not fetch IBOV: {e}")
            self.market_data = None
    
    def detect_market_regime(self) -> Dict:
        """
        Detect current market regime.
        
        Returns:
            Dict with regime info and adaptive parameters
        """
        if not self.use_regime or self.market_data is None:
            return {
                'regime': 'sideways',
                'strength': 0.5,
                'params': {}
            }
        
        regime_info = self.regime_detector.detect_regime(self.market_data)
        params = self.adaptive_params.get_parameters(
            regime_info['regime'],
            regime_info['strength']
        )
        
        self.current_regime = regime_info['regime']
        self.regime_params = params
        
        # Update thresholds for current regime
        self.thresholds = get_thresholds(self.current_regime)
        
        print(f"\n🎯 Market Regime: {regime_info['regime'].upper()} (strength: {regime_info['strength']:.0%})")
        print(f"   Confidence threshold: {params['confidence_threshold']:.0%}")
        print(f"   Max position size: {params['max_position_size']:.0%}")
        print(f"   Cash buffer: {params['max_cash_pct']:.0%}")
        print(f"   Thresholds - Buy: {self.thresholds.buy_confidence:.0%}, Sell: {self.thresholds.sell_confidence:.0%}")
        
        return {
            'regime': regime_info['regime'],
            'strength': regime_info['strength'],
            'params': params
        }
    
    def calculate_kelly_position(self, data: pd.DataFrame, confidence: float) -> float:
        """
        Kelly Criterion position sizing with regime adjustment.
        """
        config = get_config()
        
        try:
            close = data['Close'].squeeze()
            returns = close.pct_change().dropna()
            
            if len(returns) < config.KELLY_MIN_DATA_POINTS:
                return config.DEFAULT_POSITION_SIZE
            
            positive = returns[returns > 0]
            negative = returns[returns < 0]
            
            if len(positive) == 0 or len(negative) == 0:
                return config.DEFAULT_POSITION_SIZE
            
            p = len(positive) / len(returns)
            q = 1 - p
            
            avg_win = positive.mean()
            avg_loss = abs(negative.mean())
            
            if avg_loss == 0:
                return config.DEFAULT_POSITION_SIZE
            
            b = avg_win / avg_loss
            kelly = (b * p - q) / b
            
            if kelly <= 0:
                return config.MIN_POSITION_SIZE
            
            # Half-Kelly with confidence scaling
            position = kelly * config.KELLY_FRACTION * confidence
            
            # Apply regime adjustment
            if self.use_regime and self.regime_params:
                position = self.adaptive_params.adjust_position_size(
                    position,
                    self.current_regime,
                    self.regime_params.get('strength', 0.5),
                    self.regime_params
                )
            
            return max(config.MIN_POSITION_SIZE, min(config.MAX_POSITION_SIZE, position))
        
        except Exception:
            return config.DEFAULT_POSITION_SIZE
    
    def get_data(self, ticker: str, days: int = 120, max_retries: int = 3,
                 end_date: datetime = None) -> pd.DataFrame:
        """Download price data with retries."""
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
                
                if not data.empty and len(data) >= 50:
                    return data
                
            except Exception as e:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)
        
        return None
    
    def get_news_sentiment(self, ticker: str) -> float:
        """Get news sentiment."""
        if not self.use_news:
            return 0.0
        
        try:
            date = datetime.now().strftime("%Y-%m-%d")
            return self.news_client.get_sentiment(ticker, date)
        except Exception:
            return 0.0
    
    def analyze_ticker(self, ticker: str, data: pd.DataFrame = None,
                       as_of_date: datetime = None) -> Dict:
        """
        Analyze ticker with full indicator suite.
        
        Args:
            ticker: Stock ticker
            data: Pre-loaded data (for backtesting)
            as_of_date: Date for backtesting
        
        Returns:
            Analysis results with fused signal
        """
        try:
            if data is None:
                data = self.get_data(ticker, end_date=as_of_date)
            
            if data is None or len(data) < 50:
                return None
            
            # Normalize columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            price = float(data['Close'].iloc[-1].item() if hasattr(data['Close'].iloc[-1], 'item') else data['Close'].iloc[-1])
            
            # Get regime for signal fusion
            regime = self.current_regime or 'sideways'
            regime_strength = self.regime_params.get('strength', 0.5) if self.regime_params else 0.5
            
            if self.use_fusion:
                # Full signal fusion (SOTA)
                fused = fuse_all_signals(data, regime=regime, regime_strength=regime_strength)
                
                signal = fused['signal']
                confidence = fused['confidence']
                fused_score = fused['score']
                
                # Extract top contributing signals for display
                category_details = fused.get('category_details', {})
                top_signals = []
                for category, details in category_details.items():
                    for sig_name, sig_signal, sig_strength in details.get('signals', []):
                        top_signals.append({
                            'name': sig_name,
                            'category': category,
                            'signal': sig_signal,
                            'strength': sig_strength
                        })
                # Sort by strength, take top 5
                top_signals.sort(key=lambda x: abs(x['strength']), reverse=True)
                key_indicators = top_signals[:5]
                
                # Also get trend for compatibility
                trend_result = self.trend_detector.detect_trend(data)
                trend = trend_result.get('consensus', 'neutral')
            else:
                # Original single-indicator approach
                trend_result = self.trend_detector.detect_trend(data)
                trend = trend_result.get('consensus', 'neutral')
                confidence = trend_result.get('confidence', 0.0)
                fused_score = confidence if trend == 'uptrend' else -confidence
                
                signal = 'BUY' if trend == 'uptrend' and confidence >= 0.5 else \
                        'SELL' if trend == 'downtrend' and confidence >= 0.5 else 'HOLD'
            
            # Feature engineering
            feature_engineer = get_feature_engineer()
            features = feature_engineer.generate_all_features(ticker, data)
            
            # Volume/ volatility adjustments
            volume_features = features.get('volume', {})
            volatility_features = features.get('volatility', {})
            
            if volume_features.get('unusual_volume', False):
                confidence *= 1.05
                print(f"     [VOL] Unusual volume (+5% confidence)")
            
            vol_regime = volatility_features.get('regime', 'medium')
            if vol_regime == 'high':
                confidence *= 0.90
                print(f"     [VOL] High volatility (-10% confidence)")
            
            # News sentiment
            news_sentiment = self.get_news_sentiment(ticker) if self.use_news else 0.0
            
            # Get regime-specific thresholds from config
            thresholds = get_thresholds(regime if self.use_regime else 'default')
            
            # Get min_confidence for buy signals
            min_confidence = thresholds.buy_confidence
            
            # Adjust based on regime params if available (for backward compatibility)
            if self.use_regime and self.regime_params:
                min_confidence = self.regime_params.get('confidence_threshold', thresholds.buy_confidence)
            
            # Position sizing
            position_size = 0.0
            conviction = 0.0
            
            if signal == 'BUY' and confidence >= min_confidence:
                position_size = self.calculate_kelly_position(data, confidence)
                
                # News boost: shifted sigmoid centered at 1.0
                # Positive sentiment boosts > 1.0, negative reduces < 1.0
                if abs(news_sentiment) > 0.1:
                    sigmoid_boost = 0.5 + (1 / (1 + np.exp(-5 * news_sentiment)))
                    position_size *= sigmoid_boost
                
                position_size = min(0.80, position_size)
                conviction = confidence
            
            elif signal == 'SELL' and confidence >= thresholds.sell_confidence:
                position_size = 1.0
                conviction = -confidence
            
            return {
                'ticker': ticker,
                'price': price,
                'trend': trend,
                'confidence': confidence,
                'news_sentiment': news_sentiment,
                'signal': signal,
                'conviction': conviction,
                'position_size': position_size,
                'fused_score': fused_score if self.use_fusion else 0,
                'key_indicators': key_indicators if self.use_fusion else [],
                'features': {
                    'volume_momentum': volume_features.get('volume_momentum', 1.0),
                    'unusual_volume': volume_features.get('unusual_volume', False),
                    'volatility_regime': vol_regime
                }
            }
        
        except Exception as e:
            print(f"  ❌ {ticker}: {e}")
            return None
    
    def _analyze_wrapper(self, ticker: str) -> Dict:
        """Parallel processing wrapper."""
        print(f"📊 {ticker}...", end=" ", flush=True)
        result = self.analyze_ticker(ticker)
        if result:
            print(f"{result['signal']} ({result['trend']})")
        else:
            print("SKIP")
        return result
    
    def run(self, tickers: List[str] = None, parallel: bool = True) -> List[Dict]:
        """
        Run full analysis.
        
        Args:
            tickers: List of tickers (default: all)
            parallel: Use parallel processing
        
        Returns:
            List of analysis results
        """
        if tickers is None:
            tickers = TICKERS
        
        print(f"\n{'='*70}")
        print(f"🚀 ENHANCED PRODUCTION - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*70}\n")
        
        # Fetch market data and detect regime
        if self.use_regime:
            self.fetch_market_data()
            self.detect_market_regime()
        
        results = []
        
        if parallel and len(tickers) > 1 and self.n_workers > 1:
            print(f"\n⚡ Parallel mode: {self.n_workers} workers\n")
            
            with Pool(self.n_workers) as pool:
                raw_results = pool.map(self._analyze_wrapper, tickers)
            
            results = [r for r in raw_results if r is not None]
        else:
            # Sequential mode (n_workers=1 or single ticker)
            for ticker in tickers:
                print(f"📊 {ticker}...", end=" ")
                result = self.analyze_ticker(ticker)
                if result:
                    print(f"{result['signal']} ({result['trend']})")
                    results.append(result)
                else:
                    print("SKIP")
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: List[Dict]):
        """Print results summary."""
        if not results:
            print("\n❌ No results")
            return
        
        sorted_results = sorted(results, key=lambda x: abs(x.get('conviction', 0)), reverse=True)
        
        print(f"\n{'='*70}")
        print(f"📊 RESUMO - {len(results)} tickers")
        print(f"{'='*70}\n")
        
        print(f"{'Ticker':<10} {'Preço':>10} {'Sinal':<6} {'Conf':>6} {'Pos':>6} {'Score':>6}")
        print(f"{'-'*70}")
        
        for r in sorted_results[:20]:  # Top 20
            emoji = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '⚪'}.get(r['signal'], '⚪')
            pos = f"{r['position_size']*100:.0f}%" if r['position_size'] > 0 else "-"
            score = f"{r.get('fused_score', 0):+.2f}" if self.use_fusion else "-"
            
            print(
                f"{r['ticker']:<10} "
                f"R${r['price']:>8.2f} "
                f"{emoji} {r['signal']:<4} "
                f"{r['confidence']:>5.0%} "
                f"{pos:>6} "
                f"{score:>6}"
            )
        
        buy = sum(1 for r in results if r['signal'] == 'BUY')
        sell = sum(1 for r in results if r['signal'] == 'SELL')
        
        print(f"\n{'-'*70}")
        print(f"🟢 BUY: {buy} | 🔴 SELL: {sell} | ⚪ HOLD: {len(results) - buy - sell}")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Enhanced Production Runner")
    parser.add_argument("--ticker", type=str, help="Single ticker")
    parser.add_argument("--no-news", action="store_true", help="Disable news")
    parser.add_argument("--no-fusion", action="store_true", help="Disable signal fusion")
    parser.add_argument("--no-regime", action="store_true", help="Disable regime detection")
    
    args = parser.parse_args()
    
    runner = EnhancedProductionRunner(
        use_news=not args.no_news,
        use_fusion=not args.no_fusion,
        use_regime=not args.no_regime
    )
    
    if args.ticker:
        runner.run(tickers=[args.ticker])
    else:
        runner.run()


if __name__ == "__main__":
    main()
