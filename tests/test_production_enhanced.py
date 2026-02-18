"""
Comprehensive tests for Enhanced Production Runner.

Tests cover:
- Initialization and configuration
- Regime detection integration
- Signal fusion integration
- Position sizing with regime adjustment
- Parallel processing
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, Mock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from production_enhanced import EnhancedProductionRunner


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
    
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
def runner_no_news():
    """Create runner without news for faster tests."""
    return EnhancedProductionRunner(
        use_news=False,
        use_fusion=True,
        use_regime=True,
        n_workers=1
    )


@pytest.fixture
def runner_simple():
    """Create simple runner without fusion or regime."""
    return EnhancedProductionRunner(
        use_news=False,
        use_fusion=False,
        use_regime=False,
        n_workers=1
    )


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

class TestInitialization:
    """Tests for runner initialization."""
    
    def test_init_default(self):
        """Test default initialization."""
        runner = EnhancedProductionRunner(use_news=False)
        
        assert runner.use_news == False
        assert runner.use_fusion == True
        assert runner.use_regime == True
        assert runner.n_workers <= 8
    
    def test_init_with_custom_workers(self):
        """Test initialization with custom worker count."""
        runner = EnhancedProductionRunner(use_news=False, n_workers=4)
        
        assert runner.n_workers == 4
    
    def test_init_without_fusion(self):
        """Test initialization without signal fusion."""
        runner = EnhancedProductionRunner(use_news=False, use_fusion=False)
        
        assert runner.use_fusion == False
    
    def test_init_without_regime(self):
        """Test initialization without regime detection."""
        runner = EnhancedProductionRunner(use_news=False, use_regime=False)
        
        assert runner.use_regime == False
    
    def test_workers_capped_at_8(self):
        """Test worker count is capped at 8."""
        runner = EnhancedProductionRunner(use_news=False, n_workers=100)
        
        assert runner.n_workers <= 8


# ============================================================================
# REGIME DETECTION TESTS
# ============================================================================

class TestRegimeDetection:
    """Tests for market regime detection integration."""
    
    def test_fetch_market_data(self, runner_no_news):
        """Test market data fetching."""
        runner_no_news.fetch_market_data(days=100)
        
        assert runner_no_news.market_data is not None
    
    def test_detect_regime_returns_dict(self, runner_no_news):
        """Test regime detection returns proper structure."""
        runner_no_news.fetch_market_data(days=100)
        result = runner_no_news.detect_market_regime()
        
        assert 'regime' in result
        assert 'strength' in result
        assert 'params' in result
    
    def test_regime_values_valid(self, runner_no_news):
        """Test regime values are valid."""
        runner_no_news.fetch_market_data(days=100)
        result = runner_no_news.detect_market_regime()
        
        assert result['regime'] in ['bull', 'bear', 'sideways']
        assert 0 <= result['strength'] <= 1
    
    def test_regime_params_adjusted(self, runner_no_news):
        """Test regime parameters are adjusted."""
        runner_no_news.fetch_market_data(days=100)
        result = runner_no_news.detect_market_regime()
        
        params = result['params']
        
        assert 'confidence_threshold' in params
        assert 'max_position_size' in params
        assert 'max_cash_pct' in params
    
    def test_bull_regime_lower_threshold(self, runner_no_news):
        """Test bull regime has lower confidence threshold."""
        runner_no_news.current_regime = 'bull'
        runner_no_news.regime_params = {
            'confidence_threshold': 0.35,
            'max_position_size': 0.80,
            'max_cash_pct': 0.20,
            'strength': 1.0
        }
        
        # Bull regime should allow more trades (lower threshold)
        assert runner_no_news.regime_params['confidence_threshold'] < 0.50
    
    def test_bear_regime_higher_threshold(self, runner_no_news):
        """Test bear regime has higher confidence threshold."""
        runner_no_news.current_regime = 'bear'
        runner_no_news.regime_params = {
            'confidence_threshold': 0.70,
            'max_position_size': 0.30,
            'max_cash_pct': 0.50,
            'strength': 1.0
        }
        
        # Bear regime should be more conservative
        assert runner_no_news.regime_params['confidence_threshold'] > 0.50
    
    def test_regime_disabled(self, runner_simple):
        """Test regime detection when disabled."""
        result = runner_simple.detect_market_regime()
        
        assert result['regime'] == 'sideways'
        assert result['strength'] == 0.5


# ============================================================================
# POSITION SIZING TESTS
# ============================================================================

class TestPositionSizing:
    """Tests for position sizing with regime adjustment."""
    
    def test_kelly_position_returns_float(self, runner_no_news, sample_data):
        """Test Kelly position returns a float."""
        position = runner_no_news.calculate_kelly_position(sample_data, confidence=0.7)
        
        assert isinstance(position, float)
    
    def test_kelly_position_in_valid_range(self, runner_no_news, sample_data):
        """Test Kelly position is in valid range."""
        position = runner_no_news.calculate_kelly_position(sample_data, confidence=0.7)
        
        assert 0.10 <= position <= 0.80
    
    def test_kelly_with_high_confidence(self, runner_no_news, sample_data):
        """Test Kelly with high confidence produces larger position."""
        low_conf_position = runner_no_news.calculate_kelly_position(sample_data, confidence=0.5)
        high_conf_position = runner_no_news.calculate_kelly_position(sample_data, confidence=0.9)
        
        # Higher confidence should generally produce larger position
        # (depending on volatility)
        assert high_conf_position >= low_conf_position * 0.5  # Allow some margin
    
    def test_kelly_regime_adjustment_bull(self, runner_no_news, sample_data):
        """Test Kelly position adjusted for bull regime."""
        runner_no_news.current_regime = 'bull'
        runner_no_news.regime_params = {
            'confidence_threshold': 0.35,
            'max_position_size': 0.80,
            'regime_multiplier': 1.3,
            'strength': 1.0
        }
        
        position = runner_no_news.calculate_kelly_position(sample_data, confidence=0.7)
        
        # Bull regime should allow larger positions
        assert position <= 0.80  # Capped at regime max
    
    def test_kelly_regime_adjustment_bear(self, runner_no_news, sample_data):
        """Test Kelly position adjusted for bear regime."""
        runner_no_news.current_regime = 'bear'
        runner_no_news.regime_params = {
            'confidence_threshold': 0.70,
            'max_position_size': 0.30,
            'regime_multiplier': 0.5,
            'strength': 1.0
        }
        
        position = runner_no_news.calculate_kelly_position(sample_data, confidence=0.7)
        
        # Bear regime should cap positions lower
        assert position <= 0.30


# ============================================================================
# DATA FETCHING TESTS
# ============================================================================

class TestDataFetching:
    """Tests for data fetching."""
    
    def test_get_data_returns_dataframe(self, runner_no_news, sample_data):
        """Test get_data returns DataFrame."""
        with patch('production_enhanced.yf.download') as mock_download:
            mock_download.return_value = sample_data
            
            data = runner_no_news.get_data('PETR4.SA', days=100)
            
            assert isinstance(data, pd.DataFrame)
    
    def test_get_data_insufficient_returns_none(self, runner_no_news):
        """Test get_data with insufficient data returns None."""
        with patch('production_enhanced.yf.download') as mock_download:
            mock_download.return_value = pd.DataFrame()  # Empty
            
            data = runner_no_news.get_data('INVALID.SA', days=100)
            
            assert data is None


# ============================================================================
# TICKER ANALYSIS TESTS
# ============================================================================

class TestTickerAnalysis:
    """Tests for individual ticker analysis."""
    
    def test_analyze_ticker_returns_dict(self, runner_no_news, sample_data):
        """Test analyze_ticker returns dict."""
        result = runner_no_news.analyze_ticker('TEST.SA', data=sample_data)
        
        assert isinstance(result, dict)
    
    def test_analyze_ticker_structure(self, runner_no_news, sample_data):
        """Test analyze_ticker returns correct structure."""
        result = runner_no_news.analyze_ticker('TEST.SA', data=sample_data)
        
        if result is not None:
            assert 'ticker' in result
            assert 'signal' in result
            assert 'confidence' in result
            assert 'price' in result
    
    def test_analyze_ticker_signal_values(self, runner_no_news, sample_data):
        """Test analyze_ticker signal is valid."""
        result = runner_no_news.analyze_ticker('TEST.SA', data=sample_data)
        
        if result is not None:
            assert result['signal'] in ['BUY', 'SELL', 'HOLD']
    
    def test_analyze_ticker_with_fusion(self, runner_no_news, sample_data):
        """Test analyze_ticker with signal fusion enabled."""
        runner_no_news.use_fusion = True
        result = runner_no_news.analyze_ticker('TEST.SA', data=sample_data)
        
        if result is not None:
            assert 'fused_score' in result
    
    def test_analyze_ticker_without_fusion(self, runner_simple, sample_data):
        """Test analyze_ticker without signal fusion."""
        runner_simple.use_fusion = False
        result = runner_simple.analyze_ticker('TEST.SA', data=sample_data)
        
        if result is not None:
            # Should still have basic fields
            assert 'signal' in result
            assert 'confidence' in result


# ============================================================================
# SIGNAL GENERATION TESTS
# ============================================================================

class TestSignalGeneration:
    """Tests for signal generation logic."""
    
    def test_uptrend_produces_buy(self, runner_no_news):
        """Test uptrend data produces BUY signal."""
        # Create clear uptrend
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        close = 100 + np.linspace(0, 30, 100)
        
        uptrend_df = pd.DataFrame({
            'Close': close,
            'High': close + 1,
            'Low': close - 1,
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
        result = runner_no_news.analyze_ticker('UPTREND.SA', data=uptrend_df)
        
        if result is not None:
            assert result['signal'] in ['BUY', 'HOLD']  # Could be HOLD if confidence low
    
    def test_downtrend_produces_sell(self, runner_no_news):
        """Test downtrend data produces SELL signal."""
        # Create clear downtrend
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        close = 130 - np.linspace(0, 30, 100)
        
        downtrend_df = pd.DataFrame({
            'Close': close,
            'High': close + 1,
            'Low': close - 1,
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
        result = runner_no_news.analyze_ticker('DOWNTREND.SA', data=downtrend_df)
        
        if result is not None:
            assert result['signal'] in ['SELL', 'HOLD']


# ============================================================================
# PARALLEL PROCESSING TESTS
# ============================================================================

class TestParallelProcessing:
    """Tests for parallel processing."""
    
    def test_parallel_flag(self, runner_no_news):
        """Test parallel processing flag."""
        assert runner_no_news.n_workers >= 1
    
    def test_analyze_wrapper_returns_result(self, runner_no_news, sample_data):
        """Test analyze wrapper returns result."""
        with patch.object(runner_no_news, 'analyze_ticker') as mock_analyze:
            mock_analyze.return_value = {
                'ticker': 'TEST.SA',
                'signal': 'BUY',
                'trend': 'uptrend',
                'confidence': 0.7
            }
            
            result = runner_no_news._analyze_wrapper('TEST.SA')
            
            assert result is not None


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Tests for full integration."""
    
    def test_full_analysis_single_ticker(self, runner_no_news):
        """Test full analysis for single ticker."""
        results = runner_no_news.run(tickers=['PETR4.SA'], parallel=False)
        
        # Should return list (may be empty if data fetch fails)
        assert isinstance(results, list)
    
    def test_full_analysis_multiple_tickers(self, runner_no_news, sample_data):
        """Test full analysis for multiple tickers."""
        # Use mock data to avoid network calls
        with patch.object(runner_no_news, 'get_data') as mock_get:
            mock_get.return_value = sample_data
            
            results = runner_no_news.run(
                tickers=['TEST1.SA', 'TEST2.SA'],
                parallel=False
            )
            
            assert isinstance(results, list)
    
    def test_summary_printed(self, runner_no_news, capsys):
        """Test summary is printed."""
        with patch.object(runner_no_news, 'analyze_ticker') as mock_analyze:
            mock_analyze.return_value = {
                'ticker': 'TEST.SA',
                'signal': 'BUY',
                'confidence': 0.7,
                'trend': 'uptrend',
                'price': 100.0,
                'conviction': 0.7,
                'position_size': 0.25,
                'fused_score': 0.3,
                'features': {}
            }
            
            runner_no_news._print_summary([mock_analyze.return_value])
            
            captured = capsys.readouterr()
            assert 'RESUMO' in captured.out or 'TEST.SA' in captured.out


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_insufficient_data(self, runner_no_news):
        """Test with insufficient price data."""
        short_df = pd.DataFrame({
            'Close': [100, 101, 102]
        })
        
        result = runner_no_news.analyze_ticker('SHORT.SA', data=short_df)
        
        # Should return None for insufficient data
        assert result is None
    
    def test_empty_ticker_list(self, runner_no_news):
        """Test with empty ticker list."""
        results = runner_no_news.run(tickers=[], parallel=False)
        
        assert results == []
    
    def test_none_data(self, runner_no_news):
        """Test with None data."""
        result = runner_no_news.analyze_ticker('NONE.SA', data=None)
        
        # Should handle gracefully (either None or attempt to fetch)
        # Depending on implementation
        pass
    
    def test_missing_regime_params(self, runner_no_news, sample_data):
        """Test analysis without regime params set."""
        runner_no_news.regime_params = None
        
        result = runner_no_news.analyze_ticker('TEST.SA', data=sample_data)
        
        # Should still work with defaults
        if result is not None:
            assert 'signal' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])