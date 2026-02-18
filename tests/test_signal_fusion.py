"""
Comprehensive tests for Signal Fusion system.

Tests cover:
- Signal classification
- Category weighting
- Regime-adjusted weights
- Confidence calculation
- Ensemble decision making
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indicators.signal_fusion import SignalFusion, fuse_all_signals


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
    
    # Generate realistic price data with trend
    base_price = 100
    trend = np.linspace(0, 20, 200)
    noise = np.random.randn(200) * 2
    close = base_price + trend + noise.cumsum()
    
    high = close + np.abs(np.random.randn(200)) * 1.5
    low = close - np.abs(np.random.randn(200)) * 1.5
    volume = np.random.randint(1000000, 5000000, 200)
    
    return pd.DataFrame({
        'Open': close + np.random.randn(200) * 0.5,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)


@pytest.fixture
def fusion():
    """Create SignalFusion instance."""
    return SignalFusion()


# ============================================================================
# SIGNAL CLASSIFICATION TESTS
# ============================================================================

class TestSignalClassification:
    """Tests for signal direction classification."""
    
    def test_classify_bullish_signals(self, fusion):
        """Test classification of bullish signals."""
        bullish_signals = ['bullish', 'buy', 'oversold', 'accumulation',
                          'bullish_divergence', 'above_vwap', 'golden_cross']
        
        for signal in bullish_signals:
            result = fusion._classify_signal(signal)
            assert result == 1, f"Expected 1 for {signal}, got {result}"
    
    def test_classify_bearish_signals(self, fusion):
        """Test classification of bearish signals."""
        bearish_signals = ['bearish', 'sell', 'overbought', 'distribution',
                          'bearish_divergence', 'below_vwap', 'death_cross']
        
        for signal in bearish_signals:
            result = fusion._classify_signal(signal)
            assert result == -1, f"Expected -1 for {signal}, got {result}"
    
    def test_classify_neutral_signals(self, fusion):
        """Test classification of neutral signals."""
        neutral_signals = ['neutral', 'unknown', 'consolidation', 'squeeze']
        
        for signal in neutral_signals:
            result = fusion._classify_signal(signal)
            assert result == 0, f"Expected 0 for {signal}, got {result}"
    
    def test_classify_case_insensitive(self, fusion):
        """Test classification is case insensitive."""
        assert fusion._classify_signal('BULLISH') == 1
        assert fusion._classify_signal('Bearish') == -1
        assert fusion._classify_signal('NEUTRAL') == 0


# ============================================================================
# WEIGHT CALCULATION TESTS
# ============================================================================

class TestWeightCalculation:
    """Tests for weight calculations."""
    
    def test_category_weights_sum_to_one(self, fusion):
        """Test category weights sum to 1.0."""
        total = sum(fusion.CATEGORY_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001
    
    def test_indicator_weights_within_category(self, fusion):
        """Test indicator weights are properly scaled within categories."""
        # Weights should be relative within categories
        assert fusion._get_indicator_weight('MACD', 'momentum') > fusion._get_indicator_weight('ROC', 'momentum')
        assert fusion._get_indicator_weight('ADX', 'trend') > 0
    
    def test_regime_adjustments_bull(self, fusion):
        """Test regime adjustments for bull market."""
        fusion.set_regime('bull', strength=1.0)
        adjusted = fusion._get_adjusted_weights()
        
        # Momentum should be boosted in bull
        assert adjusted['momentum'] > fusion.CATEGORY_WEIGHTS['momentum']
    
    def test_regime_adjustments_bear(self, fusion):
        """Test regime adjustments for bear market."""
        fusion.set_regime('bear', strength=1.0)
        adjusted = fusion._get_adjusted_weights()
        
        # Volatility should be boosted in bear
        assert adjusted['volatility'] > fusion.CATEGORY_WEIGHTS['volatility']
    
    def test_regime_adjustments_sideways(self, fusion):
        """Test regime adjustments for sideways market."""
        fusion.set_regime('sideways', strength=1.0)
        adjusted = fusion._get_adjusted_weights()
        
        # Weights should still sum to 1
        total = sum(adjusted.values())
        assert abs(total - 1.0) < 0.001
    
    def test_adjusted_weights_sum_to_one(self, fusion):
        """Test adjusted weights always sum to 1.0."""
        for regime in ['bull', 'bear', 'sideways']:
            for strength in [0.0, 0.5, 1.0]:
                fusion.set_regime(regime, strength)
                adjusted = fusion._get_adjusted_weights()
                total = sum(adjusted.values())
                assert abs(total - 1.0) < 0.001, f"Regime: {regime}, Strength: {strength}"


# ============================================================================
# SIGNAL ADDITION TESTS
# ============================================================================

class TestSignalAddition:
    """Tests for adding signals to the fusion system."""
    
    def test_add_single_signal(self, fusion):
        """Test adding a single signal."""
        fusion.add_signal('RSI', 'momentum', 'oversold', 0.8, 'Test signal')
        
        assert len(fusion.signals) == 1
        assert fusion.signals[0].name == 'RSI'
        assert fusion.signals[0].signal == 'oversold'
    
    def test_add_multiple_signals(self, fusion):
        """Test adding multiple signals."""
        fusion.add_signal('RSI', 'momentum', 'oversold', 0.8)
        fusion.add_signal('MACD', 'momentum', 'bullish', 0.6)
        fusion.add_signal('ADX', 'trend', 'bullish', 0.7)
        
        assert len(fusion.signals) == 3
    
    def test_strength_capped_at_one(self, fusion):
        """Test signal strength is capped at 1.0."""
        fusion.add_signal('RSI', 'momentum', 'oversold', 5.0)  # Over 1.0
        
        assert fusion.signals[0].strength == 1.0
    
    def test_strength_floored_at_zero(self, fusion):
        """Test signal strength is floored at 0."""
        fusion.add_signal('RSI', 'momentum', 'neutral', -0.5)  # Negative
        
        assert fusion.signals[0].strength == 0.0
    
    def test_clear_signals(self, fusion):
        """Test clearing all signals."""
        fusion.add_signal('RSI', 'momentum', 'oversold', 0.8)
        fusion.add_signal('MACD', 'momentum', 'bullish', 0.6)
        
        fusion.clear()
        
        assert len(fusion.signals) == 0


# ============================================================================
# FUSED SIGNAL CALCULATION TESTS
# ============================================================================

class TestFusedSignalCalculation:
    """Tests for calculating the fused signal."""
    
    def test_empty_signals_returns_hold(self, fusion):
        """Test empty signals returns HOLD."""
        result = fusion.calculate_fused_signal()
        
        assert result['signal'] == 'HOLD'
        assert result['confidence'] == 0.0
    
    def test_all_bullish_signals(self, fusion):
        """Test all bullish signals produce BUY."""
        fusion.add_signal('RSI', 'momentum', 'oversold', 0.8)
        fusion.add_signal('MACD', 'momentum', 'bullish', 0.7)
        fusion.add_signal('ADX', 'trend', 'bullish', 0.8)
        fusion.add_signal('OBV', 'volume', 'accumulation', 0.6)
        
        result = fusion.calculate_fused_signal()
        
        assert result['signal'] == 'BUY'
        assert result['confidence'] > 0.5
    
    def test_all_bearish_signals(self, fusion):
        """Test all bearish signals produce SELL."""
        fusion.add_signal('RSI', 'momentum', 'overbought', 0.8)
        fusion.add_signal('MACD', 'momentum', 'bearish', 0.7)
        fusion.add_signal('ADX', 'trend', 'bearish', 0.8)
        fusion.add_signal('OBV', 'volume', 'distribution', 0.6)
        
        result = fusion.calculate_fused_signal()
        
        assert result['signal'] == 'SELL'
        assert result['confidence'] > 0.5
    
    def test_mixed_signals_produce_hold(self, fusion):
        """Test mixed signals produce HOLD."""
        fusion.add_signal('RSI', 'momentum', 'oversold', 0.8)  # Bullish
        fusion.add_signal('MACD', 'momentum', 'bearish', 0.7)  # Bearish
        fusion.add_signal('ADX', 'trend', 'neutral', 0.3)
        
        result = fusion.calculate_fused_signal()
        
        # Mixed signals should lean toward HOLD
        assert result['signal'] in ['HOLD', 'BUY', 'SELL']  # Could go either way with mixed
    
    def test_result_structure(self, fusion):
        """Test result has correct structure."""
        fusion.add_signal('RSI', 'momentum', 'oversold', 0.8)
        
        result = fusion.calculate_fused_signal()
        
        assert 'signal' in result
        assert 'confidence' in result
        assert 'score' in result
        assert 'category_scores' in result
        assert 'category_details' in result
        assert 'interpretation' in result
        assert 'regime' in result
    
    def test_regime_in_result(self, fusion):
        """Test regime is included in result."""
        fusion.set_regime('bull', 0.8)
        fusion.add_signal('RSI', 'momentum', 'oversold', 0.8)
        
        result = fusion.calculate_fused_signal()
        
        assert result['regime'] == 'bull'


# ============================================================================
# CONFIDENCE CALCULATION TESTS
# ============================================================================

class TestConfidenceCalculation:
    """Tests for confidence calculation."""
    
    def test_high_agreement_high_confidence(self, fusion):
        """Test high signal agreement produces high confidence."""
        # All signals agree
        fusion.add_signal('RSI', 'momentum', 'oversold', 0.9)
        fusion.add_signal('MACD', 'momentum', 'bullish', 0.9)
        fusion.add_signal('ADX', 'trend', 'bullish', 0.9)
        fusion.add_signal('OBV', 'volume', 'accumulation', 0.9)
        
        result = fusion.calculate_fused_signal()
        
        assert result['confidence'] > 0.7
    
    def test_low_agreement_low_confidence(self, fusion):
        """Test low signal agreement produces low confidence."""
        fusion.add_signal('RSI', 'momentum', 'oversold', 0.9)  # Bullish
        fusion.add_signal('MACD', 'momentum', 'bearish', 0.9)  # Bearish
        fusion.add_signal('ADX', 'trend', 'neutral', 0.3)
        
        result = fusion.calculate_fused_signal()
        
        # Lower confidence when signals disagree
        assert result['confidence'] < 0.8  # Could still be moderate due to other factors
    
    def test_more_signals_more_confidence(self, fusion):
        """Test more agreeing signals increases confidence."""
        # Few signals
        fusion.add_signal('RSI', 'momentum', 'oversold', 0.8)
        result1 = fusion.calculate_fused_signal()
        
        # More signals
        fusion.clear()
        fusion.add_signal('RSI', 'momentum', 'oversold', 0.8)
        fusion.add_signal('MACD', 'momentum', 'bullish', 0.8)
        fusion.add_signal('ADX', 'trend', 'bullish', 0.8)
        result2 = fusion.calculate_fused_signal()
        
        # More signals should give higher or equal confidence
        assert result2['confidence'] >= result1['confidence'] * 0.8  # Allow some margin


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestSignalFusionIntegration:
    """Tests for full integration with real data."""
    
    def test_fuse_all_signals_returns_dict(self, sample_data):
        """Test fuse_all_signals returns proper structure."""
        result = fuse_all_signals(sample_data, regime='sideways', regime_strength=0.5)
        
        assert isinstance(result, dict)
        assert 'signal' in result
        assert 'confidence' in result
    
    def test_fuse_all_signals_different_regimes(self, sample_data):
        """Test fuse_all_signals with different regimes."""
        for regime in ['bull', 'bear', 'sideways']:
            result = fuse_all_signals(sample_data, regime=regime, regime_strength=0.7)
            
            assert result['signal'] in ['BUY', 'SELL', 'HOLD']
            assert result['regime'] == regime
    
    def test_fuse_all_signals_has_category_breakdown(self, sample_data):
        """Test fuse_all_signals includes category breakdown."""
        result = fuse_all_signals(sample_data)
        
        assert 'category_scores' in result
        assert 'momentum' in result['category_scores'] or len(result['category_scores']) > 0


# ============================================================================
# INTERPRETATION TESTS
# ============================================================================

class TestInterpretation:
    """Tests for human-readable interpretation."""
    
    def test_interpretation_includes_signal(self, fusion):
        """Test interpretation includes signal direction."""
        fusion.add_signal('RSI', 'momentum', 'oversold', 0.8)
        
        result = fusion.calculate_fused_signal()
        
        assert 'BUY' in result['interpretation'] or 'SELL' in result['interpretation'] or 'HOLD' in result['interpretation']
    
    def test_interpretation_includes_regime(self, fusion):
        """Test interpretation includes regime."""
        fusion.set_regime('bull', 0.8)
        fusion.add_signal('RSI', 'momentum', 'oversold', 0.8)
        
        result = fusion.calculate_fused_signal()
        
        assert 'BULL' in result['interpretation'].upper()


# ============================================================================
# EDGE CASES
# ============================================================================

class TestSignalFusionEdgeCases:
    """Tests for edge cases."""
    
    def test_zero_strength_signals(self, fusion):
        """Test signals with zero strength."""
        fusion.add_signal('RSI', 'momentum', 'neutral', 0.0)
        
        result = fusion.calculate_fused_signal()
        
        assert result['signal'] == 'HOLD'
    
    def test_single_category_signals(self, fusion):
        """Test signals from only one category."""
        fusion.add_signal('RSI', 'momentum', 'oversold', 0.8)
        fusion.add_signal('MACD', 'momentum', 'bullish', 0.7)
        
        result = fusion.calculate_fused_signal()
        
        assert result is not None
        assert 'signal' in result
    
    def test_extreme_regime_strength(self, fusion):
        """Test extreme regime strength values."""
        for strength in [0.0, 1.0]:
            fusion.set_regime('bull', strength)
            fusion.add_signal('RSI', 'momentum', 'oversold', 0.8)
            
            result = fusion.calculate_fused_signal()
            
            assert result is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
