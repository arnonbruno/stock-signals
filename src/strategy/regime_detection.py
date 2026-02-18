#!/usr/bin/env python3
"""
Adaptive Market Regime Detection System

Detects market conditions (bull/bear/sideways) and adjusts strategy parameters:
- Bull market: Aggressive positioning, lower confidence threshold
- Bear market: Defensive positioning, higher confidence threshold  
- Sideways: Neutral settings

This makes the system perform well in ALL market conditions.
"""

import numpy as np
import pandas as pd
from typing import Dict
from scipy import stats


class MarketRegimeDetector:
    """
    Detects overall market regime using multiple indicators.
    
    Uses a combination of:
    1. Price trend (200-day MA)
    2. Market breadth (trend consistency)
    3. Volatility regime
    4. Momentum strength
    """
    
    def __init__(self, lookback_period: int = 200):
        self.lookback = lookback_period
        
    def detect_regime(self, market_data: pd.DataFrame) -> Dict:
        """
        Detect market regime from market index data (e.g., IBOV).
        
        Returns:
            Dict with:
            - regime: 'bull', 'bear', 'sideways'
            - strength: 0-1 (how strong is the regime signal)
            - confidence: 0-1 (how confident in the detection)
        """
        if len(market_data) < self.lookback:
            return {
                'regime': 'sideways',
                'strength': 0.5,
                'confidence': 0.0
            }
        
        # Normalize columns
        if isinstance(market_data.columns, pd.MultiIndex):
            market_data.columns = market_data.columns.get_level_values(0)
        
        close = market_data['Close']
        
        # 1. Trend indicator (MA200 crossover)
        ma200 = close.rolling(200).mean()
        ma50 = close.rolling(50).mean()
        
        price_vs_ma200 = (close.iloc[-1] / ma200.iloc[-1] - 1)
        ma50_vs_ma200 = (ma50.iloc[-1] / ma200.iloc[-1] - 1)
        
        # 2. Momentum (6-month returns)
        returns_6m = close.pct_change(126).iloc[-1]
        
        # 3. Volatility regime
        returns = close.pct_change().dropna()
        recent_vol = returns.tail(63).std() * np.sqrt(252)
        long_vol = returns.tail(252).std() * np.sqrt(252)
        vol_ratio = recent_vol / long_vol if long_vol > 0 else 1
        
        # 4. Trend consistency (R-squared of price)
        if len(close) >= 50:
            x = np.arange(len(close.tail(50)))
            y = close.tail(50).values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            r_squared = r_value ** 2
        else:
            r_squared = 0.5
        
        # Combine signals
        bullish_signals = 0
        bearish_signals = 0
        
        # Price above MA200
        if price_vs_ma200 > 0.05:
            bullish_signals += 2
        elif price_vs_ma200 > 0:
            bullish_signals += 1
        elif price_vs_ma200 < -0.05:
            bearish_signals += 2
        elif price_vs_ma200 < 0:
            bearish_signals += 1
        
        # MA50 above MA200
        if ma50_vs_ma200 > 0.03:
            bullish_signals += 1
        elif ma50_vs_ma200 < -0.03:
            bearish_signals += 1
        
        # Positive 6-month momentum
        if returns_6m > 0.15:
            bullish_signals += 2
        elif returns_6m > 0:
            bullish_signals += 1
        elif returns_6m < -0.15:
            bearish_signals += 2
        elif returns_6m < 0:
            bearish_signals += 1
        
        # Low volatility = complacency (bull market)
        if vol_ratio < 0.8:
            bullish_signals += 1
        elif vol_ratio > 1.2:
            bearish_signals += 1
        
        # Strong trend (high R-squared)
        if r_squared > 0.7:
            if price_vs_ma200 > 0:
                bullish_signals += 1
            else:
                bearish_signals += 1
        
        # Classify regime
        net_signal = bullish_signals - bearish_signals
        
        if net_signal >= 3:
            regime = 'bull'
            strength = min(1.0, net_signal / 5.0)
        elif net_signal <= -3:
            regime = 'bear'
            strength = min(1.0, abs(net_signal) / 5.0)
        else:
            regime = 'sideways'
            strength = 1 - (abs(net_signal) / 3.0)
        
        # Confidence based on signal clarity
        total_signals = bullish_signals + bearish_signals
        confidence = min(1.0, total_signals / 8.0)
        
        return {
            'regime': regime,
            'strength': strength,
            'confidence': confidence,
            'bullish_signals': bullish_signals,
            'bearish_signals': bearish_signals,
            'details': {
                'price_vs_ma200': price_vs_ma200,
                'ma50_vs_ma200': ma50_vs_ma200,
                'returns_6m': returns_6m,
                'vol_ratio': vol_ratio,
                'r_squared': r_squared
            }
        }


class AdaptiveStrategyParameters:
    """
    Adjusts strategy parameters based on market regime.
    
    Bull Market:
    - Lower confidence threshold (more trades)
    - Higher position sizes
    - Longer holding periods
    - Less defensive cash
    
    Bear Market:
    - Higher confidence threshold (fewer, safer trades)
    - Smaller position sizes
    - Faster exits
    - More defensive cash
    
    Sideways:
    - Neutral parameters
    """
    
    def __init__(self):
        # Base parameters
        self.base_confidence_threshold = 0.50
        self.base_min_position = 0.10
        self.base_max_position = 0.60
        self.base_kelly_fraction = 0.5
        
    def get_parameters(self, regime: str, regime_strength: float) -> Dict:
        """
        Get adaptive parameters based on market regime.
        
        Args:
            regime: 'bull', 'bear', or 'sideways'
            regime_strength: 0-1 (how strong the regime signal is)
        
        Returns:
            Dict with adjusted strategy parameters
        """
        if regime == 'bull':
            # Aggressive in bull markets
            return {
                'confidence_threshold': max(0.35, self.base_confidence_threshold - 0.15 * regime_strength),
                'min_position_size': min(0.25, self.base_min_position + 0.15 * regime_strength),
                'max_position_size': min(0.80, self.base_max_position + 0.20 * regime_strength),
                'kelly_fraction': min(0.75, self.base_kelly_fraction + 0.25 * regime_strength),
                'sell_threshold': 0.40,  # Sell only on strong downtrend
                'max_cash_pct': 0.20,    # Keep max 20% cash
                'regime_multiplier': 1.0 + (0.3 * regime_strength)  # Boost positions
            }
        
        elif regime == 'bear':
            # Defensive in bear markets
            return {
                'confidence_threshold': min(0.70, self.base_confidence_threshold + 0.20 * regime_strength),
                'min_position_size': max(0.05, self.base_min_position - 0.05 * regime_strength),
                'max_position_size': max(0.30, self.base_max_position - 0.30 * regime_strength),
                'kelly_fraction': max(0.25, self.base_kelly_fraction - 0.25 * regime_strength),
                'sell_threshold': 0.55,  # Sell on moderate weakness
                'max_cash_pct': 0.50,    # Keep up to 50% cash
                'regime_multiplier': max(0.5, 1.0 - (0.5 * regime_strength))  # Reduce positions
            }
        
        else:  # sideways
            # Neutral in sideways markets
            return {
                'confidence_threshold': self.base_confidence_threshold,
                'min_position_size': self.base_min_position,
                'max_position_size': self.base_max_position,
                'kelly_fraction': self.base_kelly_fraction,
                'sell_threshold': 0.50,
                'max_cash_pct': 0.30,
                'regime_multiplier': 1.0
            }
    
    def adjust_position_size(self, base_position: float, regime: str, 
                            regime_strength: float, params: Dict) -> float:
        """
        Adjust position size based on market regime.
        """
        multiplier = params['regime_multiplier']
        adjusted = base_position * multiplier
        
        # Enforce bounds
        return max(
            params['min_position_size'],
            min(params['max_position_size'], adjusted)
        )


def calculate_price_based_sentiment(data: pd.DataFrame, lookback: int = 10) -> float:
    """
    Calculate sentiment proxy based on price action.
    
    This simulates what news sentiment WOULD have been using only
    data available at that time (no look-ahead bias).
    """
    if len(data) < lookback:
        return 0.0
    
    # Normalize columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    recent = data.tail(lookback)
    
    # 1. Price momentum sentiment
    returns = recent['Close'].pct_change().dropna()
    if len(returns) == 0:
        momentum_sentiment = 0.0
    else:
        avg_return = returns.mean()
        momentum_sentiment = np.clip(avg_return * 50, -1, 1)
    
    # 2. Volume sentiment
    if 'Volume' in data.columns and data['Volume'].notna().sum() > lookback:
        volume = recent['Volume']
        avg_volume = data['Volume'].tail(lookback * 3).mean()
        
        if avg_volume > 0:
            volume_ratio = volume.mean() / avg_volume
            volume_factor = np.clip(volume_ratio - 1, 0, 2)
        else:
            volume_factor = 0
    else:
        volume_factor = 0
    
    # 3. Volatility dampener
    if len(returns) > 2:
        volatility = returns.std()
        vol_dampener = max(0.5, 1 - volatility * 25)
    else:
        vol_dampener = 1.0
    
    # Combine
    base_sentiment = momentum_sentiment * (1 + volume_factor * 0.3) * vol_dampener
    noise = np.random.normal(0, 0.05)
    
    return np.clip(base_sentiment + noise, -1, 1)


# Singleton instances
_regime_detector = None
_adaptive_params = None


def get_regime_detector() -> MarketRegimeDetector:
    """Get or create global regime detector."""
    global _regime_detector
    if _regime_detector is None:
        _regime_detector = MarketRegimeDetector()
    return _regime_detector


def get_adaptive_params() -> AdaptiveStrategyParameters:
    """Get or create global adaptive parameters."""
    global _adaptive_params
    if _adaptive_params is None:
        _adaptive_params = AdaptiveStrategyParameters()
    return _adaptive_params
