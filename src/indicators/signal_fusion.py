"""
Signal Fusion - Combines all indicators into unified trading signal.

Implements weighted ensemble approach:
1. Collect signals from all indicator categories
2. Apply regime-aware weighting
3. Calculate confidence based on signal agreement
4. Generate final BUY/SELL/HOLD decision

SOTA approach: Multi-factor model with adaptive weighting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class IndicatorSignal:
    """Container for individual indicator signal."""
    name: str
    category: str
    signal: str  # 'bullish', 'bearish', 'neutral', etc.
    strength: float  # 0.0 to 1.0
    weight: float  # importance in ensemble
    interpretation: str = ""


class SignalFusion:
    """
    Fuses signals from multiple technical indicators.
    
    Architecture:
    1. Momentum Layer: RSI, MACD, Stochastic, ROC
    2. Volatility Layer: ATR, Bollinger, Keltner
    3. Volume Layer: OBV, VWAP, MFI
    4. Trend Layer: MA, ADX, SuperTrend
    
    Each layer contributes to final signal with regime-adjusted weights.
    """
    
    # Base weights for each category (sum to 1.0)
    CATEGORY_WEIGHTS = {
        'momentum': 0.25,
        'volatility': 0.20,
        'volume': 0.25,
        'trend': 0.30  # Trend is most important
    }
    
    # Regime-specific weight adjustments
    REGIME_ADJUSTMENTS = {
        'bull': {
            'momentum': 1.2,   # Momentum more important in bull
            'volatility': 0.8, # Volatility less important
            'volume': 1.1,     # Volume confirms trends
            'trend': 1.1
        },
        'bear': {
            'momentum': 0.9,
            'volatility': 1.3,  # Volatility critical in bear
            'volume': 1.2,      # Volume for exit signals
            'trend': 0.8
        },
        'sideways': {
            'momentum': 1.1,   # RSI/Stoch more useful
            'volatility': 1.0,
            'volume': 0.9,
            'trend': 0.8       # Trend less reliable in sideways
        }
    }
    
    # Signal direction mappings
    BULLISH_SIGNALS = ['bullish', 'buy', 'oversold', 'accumulation', 
                       'bullish_divergence', 'above_vwap', 'above_upper',
                       'above_channel', 'golden_cross', 'bullish_crossover']
    
    BEARISH_SIGNALS = ['bearish', 'sell', 'overbought', 'distribution',
                       'bearish_divergence', 'below_vwap', 'below_lower',
                       'below_channel', 'death_cross', 'bearish_crossover']
    
    def __init__(self):
        self.signals: List[IndicatorSignal] = []
        self.regime = 'sideways'
        self.regime_strength = 0.5
    
    def set_regime(self, regime: str, strength: float = 0.5):
        """
        Set current market regime for weight adjustments.
        
        Args:
            regime: 'bull', 'bear', or 'sideways'
            strength: Regime signal strength (0-1)
        """
        self.regime = regime
        self.regime_strength = strength
    
    def add_signal(self, name: str, category: str, signal: str, 
                   strength: float, interpretation: str = ""):
        """
        Add an indicator signal to the ensemble.
        
        Args:
            name: Indicator name (e.g., 'RSI', 'MACD')
            category: Category ('momentum', 'volatility', 'volume', 'trend')
            signal: Signal direction
            strength: Signal strength (0-1)
            interpretation: Human-readable interpretation
        """
        base_weight = self._get_indicator_weight(name, category)
        
        self.signals.append(IndicatorSignal(
            name=name,
            category=category,
            signal=signal,
            strength=min(1.0, max(0.0, strength)),
            weight=base_weight,
            interpretation=interpretation
        ))
    
    def _get_indicator_weight(self, name: str, category: str) -> float:
        """
        Get weight for individual indicator within its category.
        
        Some indicators are more reliable than others.
        """
        indicator_weights = {
            # Momentum
            'RSI': 0.30,
            'MACD': 0.35,
            'Stochastic': 0.20,
            'Williams_R': 0.10,
            'ROC': 0.05,
            
            # Volatility
            'ATR': 0.25,
            'Bollinger': 0.40,
            'Keltner': 0.35,
            
            # Volume
            'OBV': 0.35,
            'VWAP': 0.30,
            'MFI': 0.20,
            'Volume_Momentum': 0.15,
            
            # Trend
            'MA_Alignment': 0.30,
            'ADX': 0.35,
            'SuperTrend': 0.35
        }
        
        return indicator_weights.get(name, 0.10)
    
    def _get_adjusted_weights(self) -> Dict[str, float]:
        """
        Calculate regime-adjusted category weights.
        
        Returns:
            Dict of category -> adjusted weight
        """
        adjustments = self.REGIME_ADJUSTMENTS.get(self.regime, {})
        
        adjusted = {}
        for category, base_weight in self.CATEGORY_WEIGHTS.items():
            adj = adjustments.get(category, 1.0)
            # Blend adjustment with regime strength
            blended_adj = 1.0 + (adj - 1.0) * self.regime_strength
            adjusted[category] = base_weight * blended_adj
        
        # Normalize to sum to 1.0
        total = sum(adjusted.values())
        return {k: v / total for k, v in adjusted.items()}
    
    def _classify_signal(self, signal: str) -> int:
        """
        Classify signal direction as numeric.
        
        Returns:
            1 for bullish, -1 for bearish, 0 for neutral
        """
        signal_lower = signal.lower()
        
        if signal_lower in [s.lower() for s in self.BULLISH_SIGNALS]:
            return 1
        elif signal_lower in [s.lower() for s in self.BEARISH_SIGNALS]:
            return -1
        else:
            return 0
    
    def calculate_fused_signal(self) -> Dict:
        """
        Calculate the fused signal from all indicators.
        
        Returns:
            Dict with final signal, confidence, and breakdown
        """
        if not self.signals:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'reason': 'No signals available'
            }
        
        # Get regime-adjusted weights
        category_weights = self._get_adjusted_weights()
        
        # Calculate weighted signals per category
        category_scores = {}
        category_details = {}
        
        for category in self.CATEGORY_WEIGHTS.keys():
            cat_signals = [s for s in self.signals if s.category == category]
            
            if not cat_signals:
                continue
            
            # Weight signals within category
            total_weight = sum(s.weight for s in cat_signals)
            
            bullish_score = 0.0
            bearish_score = 0.0
            
            for sig in cat_signals:
                direction = self._classify_signal(sig.signal)
                normalized_weight = sig.weight / total_weight if total_weight > 0 else 0
                
                if direction > 0:
                    bullish_score += sig.strength * normalized_weight
                elif direction < 0:
                    bearish_score += sig.strength * normalized_weight
            
            # Net score for category
            net_score = bullish_score - bearish_score
            category_scores[category] = net_score * category_weights.get(category, 0.25)
            
            category_details[category] = {
                'bullish': bullish_score,
                'bearish': bearish_score,
                'net': net_score,
                'signals': [(s.name, s.signal, s.strength) for s in cat_signals]
            }
        
        # Calculate overall fused signal
        total_score = sum(category_scores.values())
        
        # Determine final signal
        if total_score > 0.15:
            final_signal = 'BUY'
        elif total_score < -0.15:
            final_signal = 'SELL'
        else:
            final_signal = 'HOLD'
        
        # Calculate confidence
        # Higher confidence when:
        # 1. More signals agree
        # 2. Signal strength is higher
        # 3. Multiple categories agree
        
        signal_count = len(self.signals)
        agreeing_signals = sum(1 for s in self.signals 
                              if self._classify_signal(s.signal) == np.sign(total_score))
        agreement_rate = agreeing_signals / signal_count if signal_count > 0 else 0
        
        # Strength of consensus
        strength_factor = min(1.0, abs(total_score) * 2)
        
        # Category agreement
        agreeing_categories = sum(1 for score in category_scores.values() 
                                  if np.sign(score) == np.sign(total_score))
        category_factor = agreeing_categories / len(category_scores) if category_scores else 0
        
        confidence = (agreement_rate * 0.4 + strength_factor * 0.4 + category_factor * 0.2)
        confidence = min(1.0, confidence)
        
        # Generate interpretation
        interpretation = self._generate_interpretation(
            final_signal, total_score, category_details, confidence
        )
        
        return {
            'signal': final_signal,
            'confidence': confidence,
            'score': total_score,
            'category_scores': category_scores,
            'category_details': category_details,
            'interpretation': interpretation,
            'regime': self.regime,
            'signals_count': signal_count
        }
    
    def _generate_interpretation(self, signal: str, score: float, 
                                  details: Dict, confidence: float) -> str:
        """
        Generate human-readable interpretation of the fused signal.
        """
        lines = []
        
        # Header
        signal_emoji = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '⚪'}
        lines.append(f"\n{signal_emoji.get(signal, '⚪')} **{signal}** (confidence: {confidence:.0%})")
        lines.append(f"   Score: {score:+.2f} | Regime: {self.regime.upper()}")
        lines.append("")
        
        # Category breakdown
        lines.append("   📊 Category Breakdown:")
        for cat, score in details.items():
            cat_signal = 'BULL' if score['net'] > 0.1 else ('BEAR' if score['net'] < -0.1 else 'NEUTRAL')
            lines.append(f"      {cat.upper()}: {cat_signal} ({score['net']:+.2f})")
        
        # Top confirming signals
        lines.append("")
        lines.append("   🔑 Key Signals:")
        
        sorted_signals = sorted(
            self.signals, 
            key=lambda s: abs(s.strength) * s.weight, 
            reverse=True
        )[:5]
        
        for sig in sorted_signals:
            direction = '↑' if self._classify_signal(sig.signal) > 0 else ('↓' if self._classify_signal(sig.signal) < 0 else '→')
            lines.append(f"      {direction} {sig.name}: {sig.signal} ({sig.strength:.0%})")
        
        return "\n".join(lines)
    
    def clear(self):
        """Clear all signals for new analysis."""
        self.signals = []


def fuse_all_signals(data: pd.DataFrame, regime: str = 'sideways', 
                     regime_strength: float = 0.5) -> Dict:
    """
    Convenience function to analyze all indicators and fuse signals.
    
    Args:
        data: Price data DataFrame
        regime: Current market regime
        regime_strength: Regime signal strength
    
    Returns:
        Fused signal dict
    """
    from .momentum import MomentumIndicators
    from .volatility import VolatilityIndicators
    from .volume import VolumeIndicators
    from .trend import TrendIndicators
    
    fusion = SignalFusion()
    fusion.set_regime(regime, regime_strength)
    
    # Momentum signals
    try:
        mom = MomentumIndicators.get_all_momentum_signals(data)
        
        fusion.add_signal('RSI', 'momentum', mom['rsi']['signal'], 
                         mom['rsi'].get('strength', 0.5))
        fusion.add_signal('MACD', 'momentum', mom['macd']['signal'],
                         mom['macd'].get('strength', 0.5))
        fusion.add_signal('Stochastic', 'momentum', mom['stochastic']['signal'],
                         mom['stochastic'].get('strength', 0.5))
    except Exception:
        pass
    
    # Volatility signals
    try:
        vol = VolatilityIndicators.get_all_volatility_signals(data)
        
        fusion.add_signal('ATR', 'volatility', vol['atr'].get('regime', 'normal'),
                         0.5 if vol['atr'].get('regime') == 'high' else 0.3)
        fusion.add_signal('Bollinger', 'volatility', vol['bollinger']['signal'],
                         vol['bollinger'].get('strength', 0.5))
        fusion.add_signal('Keltner', 'volatility', vol['keltner']['signal'],
                         vol['keltner'].get('strength', 0.5))
    except Exception:
        pass
    
    # Volume signals
    try:
        vol_ind = VolumeIndicators.get_all_volume_signals(data)
        
        fusion.add_signal('OBV', 'volume', vol_ind['obv']['signal'],
                         vol_ind['obv'].get('strength', 0.5))
        fusion.add_signal('VWAP', 'volume', vol_ind['vwap']['signal'],
                         vol_ind['vwap'].get('strength', 0.5))
        fusion.add_signal('MFI', 'volume', vol_ind['mfi']['signal'],
                         0.3)
    except Exception:
        pass
    
    # Trend signals
    try:
        trend = TrendIndicators.get_all_trend_signals(data)
        
        fusion.add_signal('MA_Alignment', 'trend', trend['moving_averages']['signal'],
                         trend['moving_averages'].get('strength', 0.5))
        fusion.add_signal('ADX', 'trend', trend['adx']['signal'],
                         trend['adx'].get('strength', 0.5))
        fusion.add_signal('SuperTrend', 'trend', trend['supertrend']['signal'],
                         trend['supertrend'].get('strength', 0.5))
    except Exception:
        pass
    
    return fusion.calculate_fused_signal()
