"""
Feature engineering pipeline for enhanced signal generation.

Implements SOTA features beyond simple price/news:
- Volume patterns
- Sector momentum
- Correlation features
- Options flow (if available)
"""

import numpy as np
import pandas as pd
from typing import Dict, List


class FeatureEngineer:
    """
    Generate advanced features for trading signals.
    
    Features:
    1. Volume patterns (volume momentum, unusual volume)
    2. Sector momentum (relative strength vs IBOV)
    3. Correlation features (peer correlation, market correlation)
    4. Volatility regime (VIX-like measure)
    """
    
    def __init__(self):
        self.features = {}
    
    def _normalize_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle yfinance MultiIndex columns."""
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    
    def add_volume_features(self, data: pd.DataFrame) -> Dict:
        """
        Extract volume-based features.
        
        Returns:
            Dict with volume features:
            - volume_momentum: 5-day vs 20-day volume ratio
            - unusual_volume: Is current volume > 2x average?
            - volume_trend: Increasing or decreasing?
        """
        data = self._normalize_columns(data)
        
        if 'Volume' not in data.columns or len(data) < 20:
            return {
                'volume_momentum': 1.0,
                'unusual_volume': False,
                'volume_trend': 'neutral'
            }
        
        volume = data['Volume']
        
        # Volume momentum (short vs long-term)
        vol_5d = volume.rolling(5).mean().iloc[-1]
        vol_20d = volume.rolling(20).mean().iloc[-1]
        volume_momentum = vol_5d / vol_20d if vol_20d > 0 else 1.0
        
        # Unusual volume detection
        vol_std = volume.rolling(20).std().iloc[-1]
        current_vol = volume.iloc[-1]
        unusual_volume = current_vol > (vol_20d + 2 * vol_std)
        
        # Volume trend
        vol_slope = np.polyfit(range(10), volume.iloc[-10:].values, 1)[0]
        volume_trend = 'increasing' if vol_slope > 0 else 'decreasing'
        
        return {
            'volume_momentum': float(volume_momentum),
            'unusual_volume': bool(unusual_volume),
            'volume_trend': volume_trend
        }
    
    def add_sector_momentum(self, ticker: str, data: pd.DataFrame, 
                           market_data: pd.DataFrame = None) -> Dict:
        """
        Calculate relative strength vs market/sector.
        
        Returns:
            Dict with:
            - relative_strength: Performance vs IBOV
            - sector_rank: Percentile rank
        """
        data = self._normalize_columns(data)
        if market_data is not None:
            market_data = self._normalize_columns(market_data)
        
        if len(data) < 20:
            return {
                'relative_strength': 1.0,
                'sector_rank': 0.5
            }
        
        # Calculate returns
        ticker_returns = data['Close'].pct_change(20).iloc[-1]
        
        # If market data available, calculate relative strength
        if market_data is not None and len(market_data) >= 20:
            market_returns = market_data['Close'].pct_change(20).iloc[-1]
            relative_strength = ticker_returns / market_returns if market_returns != 0 else 1.0
        else:
            relative_strength = 1.0
        
        # Sector rank (placeholder - would need sector classification)
        sector_rank = 0.5  # Default to median
        
        return {
            'relative_strength': float(relative_strength),
            'sector_rank': float(sector_rank)
        }
    
    def add_volatility_regime(self, data: pd.DataFrame) -> Dict:
        """
        Classify current volatility regime.
        
        Returns:
            Dict with:
            - volatility: Annualized volatility
            - regime: 'low', 'medium', 'high'
            - vix_equivalent: VIX-like measure
        """
        data = self._normalize_columns(data)
        
        if len(data) < 20:
            return {
                'volatility': 0.20,
                'regime': 'medium',
                'vix_equivalent': 20.0
            }
        
        returns = data['Close'].pct_change().dropna()
        
        # Annualized volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Regime classification
        if volatility < 0.15:
            regime = 'low'
        elif volatility < 0.30:
            regime = 'medium'
        else:
            regime = 'high'
        
        # VIX equivalent (rough approximation)
        vix_equivalent = volatility * 100
        
        return {
            'volatility': float(volatility),
            'regime': regime,
            'vix_equivalent': float(vix_equivalent)
        }
    
    def add_correlation_features(self, ticker: str, data: pd.DataFrame,
                                  peer_data: Dict[str, pd.DataFrame] = None) -> Dict:
        """
        Calculate correlation with peers and market.
        
        Returns:
            Dict with:
            - market_correlation: Correlation with IBOV
            - peer_correlation: Average correlation with sector peers
            - correlation_stability: How stable is the correlation?
        """
        data = self._normalize_columns(data)
        
        if len(data) < 20:
            return {
                'market_correlation': 0.5,
                'peer_correlation': 0.5,
                'correlation_stability': 'stable'
            }
        
        # Calculate ticker returns
        ticker_returns = data['Close'].pct_change().iloc[-20:]
        
        # Market correlation (placeholder)
        market_correlation = 0.5
        
        # Peer correlation (placeholder)
        peer_correlation = 0.5
        
        # Correlation stability
        correlation_stability = 'stable'
        
        return {
            'market_correlation': float(market_correlation),
            'peer_correlation': float(peer_correlation),
            'correlation_stability': correlation_stability
        }
    
    def generate_all_features(self, ticker: str, data: pd.DataFrame,
                              market_data: pd.DataFrame = None,
                              peer_data: Dict[str, pd.DataFrame] = None) -> Dict:
        """
        Generate all features for a ticker.
        
        Returns combined dict of all feature categories.
        """
        features = {}
        
        # Volume features
        features['volume'] = self.add_volume_features(data)
        
        # Sector momentum
        features['sector'] = self.add_sector_momentum(ticker, data, market_data)
        
        # Volatility regime
        features['volatility'] = self.add_volatility_regime(data)
        
        # Correlation features
        features['correlation'] = self.add_correlation_features(ticker, data, peer_data)
        
        return features


# Singleton instance
_feature_engineer = None


def get_feature_engineer() -> FeatureEngineer:
    """Get or create global feature engineer instance."""
    global _feature_engineer
    if _feature_engineer is None:
        _feature_engineer = FeatureEngineer()
    return _feature_engineer
