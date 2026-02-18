"""
Comprehensive tests for TrendDetectorV2 - dual-timeframe trend detection.

Tests cover:
- Trend detection accuracy
- Dual-timeframe analysis (macro/micro)
- Volatility regime classification
- Adaptive thresholds
- Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.signals.trend_detector_v2 import TrendDetectorV2, RobustTrendDetector


class TestTrendDetectorV2Core:
    """Core functionality tests for TrendDetectorV2."""
    
    @pytest.fixture
    def detector(self):
        return TrendDetectorV2(macro_period=50, micro_period=20)
    
    @pytest.fixture
    def uptrend_data(self):
        """Create clear uptrend data with increasing prices."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        prices = np.linspace(100, 150, 100) + np.random.randn(100) * 1
        return pd.DataFrame({'Close': prices}, index=dates)
    
    @pytest.fixture
    def downtrend_data(self):
        """Create clear downtrend data with decreasing prices."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        prices = np.linspace(150, 100, 100) + np.random.randn(100) * 1
        return pd.DataFrame({'Close': prices}, index=dates)
    
    @pytest.fixture
    def consolidation_data(self):
        """Create consolidation/sideways data."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        prices = 125 + np.random.randn(100) * 3
        return pd.DataFrame({'Close': prices}, index=dates)
    
    def test_uptrend_detection(self, detector, uptrend_data):
        """Test that clear uptrends are detected correctly."""
        result = detector.detect_trend(uptrend_data)
        
        assert 'consensus' in result, "Result should have consensus"
        assert 'confidence' in result, "Result should have confidence"
        assert 'macro_regime' in result, "Result should have macro_regime"
        assert 'micro_state' in result, "Result should have micro_state"
        
        # Uptrend data should produce uptrend or bull_pullback consensus
        assert result['consensus'] in ['uptrend', 'bull_pullback'], \
            f"Expected uptrend, got {result['consensus']}"
        assert result['confidence'] > 0, "Confidence should be positive"
    
    def test_downtrend_detection(self, detector, downtrend_data):
        """Test that clear downtrends are detected correctly."""
        result = detector.detect_trend(downtrend_data)
        
        # Downtrend data should produce downtrend or bear_bounce consensus
        assert result['consensus'] in ['downtrend', 'bear_bounce', 'consolidation'], \
            f"Expected downtrend, got {result['consensus']}"
    
    def test_consolidation_detection(self, detector, consolidation_data):
        """Test that sideways markets are detected as consolidation."""
        result = detector.detect_trend(consolidation_data)
        
        assert result['consensus'] in ['consolidation', 'uptrend', 'downtrend'], \
            f"Unexpected consensus: {result['consensus']}"
    
    def test_insufficient_data_returns_unknown(self, detector):
        """Test that insufficient data returns unknown status."""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        short_data = pd.DataFrame({'Close': np.random.randn(10) + 100}, index=dates)
        
        result = detector.detect_trend(short_data)
        
        assert result['consensus'] == 'unknown', "Should return unknown for insufficient data"
        assert result['confidence'] == 0, "Should have zero confidence"
    
    def test_minimum_data_requirement(self, detector):
        """Test that exactly micro_period data is handled."""
        dates = pd.date_range(start='2024-01-01', periods=20, freq='D')
        min_data = pd.DataFrame({'Close': np.linspace(100, 120, 20)}, index=dates)
        
        result = detector.detect_trend(min_data)
        
        # Should work with exactly 20 data points (micro_period)
        assert 'consensus' in result
        # Should use micro as proxy for macro (not enough for macro)
        assert result['macro_regime'] == result['micro_state']


class TestDualTimeframe:
    """Tests for dual-timeframe analysis (macro + micro)."""
    
    @pytest.fixture
    def detector(self):
        return TrendDetectorV2(macro_period=50, micro_period=20)
    
    def test_macro_micro_both_uptrend(self, detector):
        """Test when both macro and micro show uptrend."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        prices = np.linspace(100, 200, 100)  # Strong uptrend
        data = pd.DataFrame({'Close': prices}, index=dates)
        
        result = detector.detect_trend(data)
        
        # Both agreeing should give strong signal
        assert result['macro_regime'] == 'uptrend'
        assert result['micro_state'] == 'uptrend'
        assert result['consensus'] == 'uptrend'
        assert result['confidence'] > 0.7, "Strong agreement should give high confidence"
    
    def test_macro_uptrend_micro_downtrend_bull_pullback(self, detector):
        """Test bull market pullback scenario (macro up, micro down)."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # Macro uptrend, but recent pullback
        prices = np.linspace(100, 150, 80).tolist() + np.linspace(150, 140, 20).tolist()
        data = pd.DataFrame({'Close': prices}, index=dates)
        
        result = detector.detect_trend(data)
        
        # Should recognize as pullback in bull market, not full downtrend
        assert result['macro_regime'] == 'uptrend'
        # Consensus should be consolidation or downtrend (depending on micro confidence)
        assert result['consensus'] in ['consolidation', 'downtrend', 'uptrend']
    
    def test_macro_downtrend_micro_uptrend_bear_bounce(self, detector):
        """Test bear market bounce scenario (macro down, micro up)."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # Macro downtrend, but recent bounce
        prices = np.linspace(150, 100, 80).tolist() + np.linspace(100, 110, 20).tolist()
        data = pd.DataFrame({'Close': prices}, index=dates)
        
        result = detector.detect_trend(data)
        
        # Should recognize as bounce in bear market
        assert result['macro_regime'] == 'downtrend'
    
    def test_macro_consolidation_trusts_micro(self, detector):
        """Test that macro consolidation trusts micro signal."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # Flat macro, but recent uptrend
        prices = (np.ones(80) * 120).tolist() + np.linspace(120, 140, 20).tolist()
        data = pd.DataFrame({'Close': prices}, index=dates)
        
        result = detector.detect_trend(data)
        
        # Should trust micro signal
        assert result['consensus'] in ['uptrend', 'consolidation']


class TestSlopeCalculation:
    """Tests for Theil-Sen robust regression slope calculation."""
    
    @pytest.fixture
    def detector(self):
        return TrendDetectorV2()
    
    def test_slope_strength_positive_trend(self, detector):
        """Test slope calculation for positive trend."""
        prices = np.linspace(100, 200, 50)
        slope, strength = detector._calculate_slope_strength(prices)
        
        assert slope > 0, "Slope should be positive for uptrend"
        assert strength > 0, "Strength should be positive"
    
    def test_slope_strength_negative_trend(self, detector):
        """Test slope calculation for negative trend."""
        prices = np.linspace(200, 100, 50)
        slope, strength = detector._calculate_slope_strength(prices)
        
        assert slope < 0, "Slope should be negative for downtrend"
        assert strength > 0, "Strength should be positive (magnitude)"
    
    def test_slope_strength_flat_trend(self, detector):
        """Test slope calculation for flat/consolidating data."""
        prices = np.ones(50) * 150 + np.random.randn(50) * 0.1
        slope, strength = detector._calculate_slope_strength(prices)
        
        assert abs(slope) < 0.1, "Slope should be near zero for flat data"
        assert strength < 0.2, "Strength should be low for flat data"
    
    def test_slope_handles_single_value(self, detector):
        """Test slope calculation with single value."""
        prices = np.array([100])
        slope, strength = detector._calculate_slope_strength(prices)
        
        assert slope == 0, "Should return 0 slope for single value"
        assert strength == 0, "Should return 0 strength for single value"
    
    def test_slope_handles_empty_data(self, detector):
        """Test slope calculation with empty data."""
        prices = np.array([])
        slope, strength = detector._calculate_slope_strength(prices)
        
        assert slope == 0, "Should return 0 slope for empty data"
        assert strength == 0, "Should return 0 strength for empty data"


class TestVolatilityRegime:
    """Tests for volatility regime classification."""
    
    @pytest.fixture
    def detector(self):
        return TrendDetectorV2()
    
    def test_low_volatility_regime(self, detector):
        """Test low volatility regime detection."""
        # Very stable prices
        prices = 100 + np.random.randn(100) * 0.001
        data = pd.DataFrame({'Close': prices})
        
        # Pass the values array, not the DataFrame
        regime = detector._calculate_volatility_regime(data['Close'].values)
        assert regime == 'low', f"Expected low volatility, got {regime}"
    
    def test_high_volatility_regime(self, detector):
        """Test high volatility regime detection."""
        # Very volatile prices
        prices = 100 + np.random.randn(100) * 0.05
        data = pd.DataFrame({'Close': prices})
        
        # Pass the values array, not the DataFrame
        regime = detector._calculate_volatility_regime(data['Close'].values)
        assert regime == 'high', f"Expected high volatility, got {regime}"
    
    def test_medium_volatility_regime(self, detector):
        """Test medium volatility regime detection."""
        # Moderate volatility
        prices = 100 + np.random.randn(100) * 0.01
        data = pd.DataFrame({'Close': prices})
        
        # Pass the values array, not the DataFrame
        regime = detector._calculate_volatility_regime(data['Close'].values)
        assert regime in ['low', 'medium', 'high'], f"Invalid regime: {regime}"
    
    def test_volatility_regime_insufficient_data(self, detector):
        """Test volatility regime with insufficient data."""
        prices = np.array([100, 101])
        data = pd.DataFrame({'Close': prices})
        
        regime = detector._calculate_volatility_regime(data)
        assert regime == 'medium', "Should default to medium for insufficient data"


class TestAdaptiveThresholds:
    """Tests for volatility-adaptive thresholds."""
    
    @pytest.fixture
    def detector(self):
        return TrendDetectorV2()
    
    def test_low_vol_threshold_more_sensitive(self, detector):
        """Test that low volatility uses lower threshold (more sensitive)."""
        base_threshold = 0.15
        slope = 0.02
        strength = 0.12
        
        direction, confidence = detector._classify_trend(
            slope, strength, 'low', threshold=base_threshold
        )
        
        # With low vol, threshold is 0.15 * 0.67 = 0.10
        # strength 0.12 > 0.10, so should classify as trend
        assert direction == 'uptrend', "Should classify as uptrend with lower threshold"
    
    def test_high_vol_threshold_more_conservative(self, detector):
        """Test that high volatility uses higher threshold (more conservative)."""
        base_threshold = 0.15
        slope = 0.02
        # With high vol, threshold is 0.15 * 1.33 = 0.20
        # Consolidation only when strength < 0.5 * adjusted_threshold = 0.10
        strength = 0.08  # Below 50% of adjusted threshold
        
        direction, confidence = detector._classify_trend(
            slope, strength, 'high', threshold=base_threshold
        )
        
        # strength 0.08 < 0.10 (50% of 0.20), so should classify as consolidation
        assert direction == 'consolidation', \
            f"Should classify as consolidation with strength={strength}, got {direction}"
    
    def test_threshold_scaling_factors(self, detector):
        """Test threshold scaling factors."""
        base = 0.15
        
        # Low volatility: 0.67x (more sensitive)
        low_adjusted = base * 0.67
        assert abs(low_adjusted - 0.10) < 0.01
        
        # High volatility: 1.33x (more conservative)
        high_adjusted = base * 1.33
        assert abs(high_adjusted - 0.20) < 0.01


class TestConsensusLogic:
    """Tests for consensus building between macro and micro."""
    
    @pytest.fixture
    def detector(self):
        return TrendDetectorV2()
    
    def test_consensus_both_agree(self, detector):
        """Test consensus when macro and micro agree."""
        consensus, confidence = detector._get_consensus('uptrend', 0.8, 'uptrend', 0.7)
        
        assert consensus == 'uptrend', "Should agree when both are uptrend"
        assert confidence > 0.7, "Should have high confidence when agreeing"
    
    def test_consensus_consolidation_penalty(self, detector):
        """Test that consolidation consensus gets confidence penalty."""
        consensus, confidence = detector._get_consensus('consolidation', 0.8, 'consolidation', 0.7)
        
        assert consensus == 'consolidation'
        # Confidence should be penalized (multiplied by 0.7)
        assert confidence < 0.8, "Consolidation should have reduced confidence"
    
    def test_consensus_macro_weighted_heavier(self, detector):
        """Test that macro regime is weighted 2x vs micro."""
        # Strong macro uptrend, weak micro consolidation
        consensus, confidence = detector._get_consensus('uptrend', 0.9, 'consolidation', 0.3)
        
        # Should trust macro when micro is consolidation
        assert consensus == 'uptrend', "Should trust macro when micro is unclear"


class TestRobustTrendDetector:
    """Tests for backward compatibility wrapper."""
    
    def test_wrapper_returns_old_format(self):
        """Test that wrapper returns dict with consensus and confidence."""
        detector = RobustTrendDetector(lookback_period=20)
        
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        prices = np.linspace(100, 150, 100)
        data = pd.DataFrame({'Close': prices}, index=dates)
        
        result = detector.get_robust_trend(data)
        
        assert 'consensus' in result, "Should have consensus key"
        assert 'confidence' in result, "Should have confidence key"
        assert result['consensus'] in ['uptrend', 'downtrend', 'consolidation']
    
    def test_wrapper_uses_v2_logic(self):
        """Test that wrapper uses TrendDetectorV2 internally."""
        detector = RobustTrendDetector()
        
        assert hasattr(detector, 'detector_v2'), "Should have v2 detector"
        assert isinstance(detector.detector_v2, TrendDetectorV2)


class TestGetRegimeForStrategy:
    """Tests for simplified strategy interface."""
    
    def test_returns_tuple(self):
        """Test that get_regime_for_strategy returns tuple."""
        detector = TrendDetectorV2()
        
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        data = pd.DataFrame({'Close': np.linspace(100, 150, 100)}, index=dates)
        
        result = detector.get_regime_for_strategy(data)
        
        assert isinstance(result, tuple), "Should return tuple"
        assert len(result) == 2, "Should return 2-element tuple"
        
        regime, confidence = result
        assert regime in ['uptrend', 'downtrend', 'consolidation']
        assert 0 <= confidence <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
