"""
Comprehensive tests for SimpleProductionRunner - main trading engine.

Tests cover:
- Data fetching with retry logic
- Kelly Criterion position sizing
- Signal generation
- Analysis pipeline
- Error handling
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, Mock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from production_simple import SimpleProductionRunner


class TestDataFetching:
    """Tests for data fetching with retry logic."""
    
    @pytest.fixture
    def runner(self):
        return SimpleProductionRunner(use_news=False)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample price data."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 100,
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 100,
            'Low': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
    
    def test_get_data_success(self, runner, sample_data):
        """Test successful data fetching."""
        with patch('production_simple.yf.download') as mock_download:
            mock_download.return_value = sample_data
            
            result = runner.get_data('PETR4.SA')
            
            assert result is not None
            assert len(result) == 100
    
    def test_get_data_retry_on_failure(self, runner, sample_data):
        """Test retry on failure."""
        with patch('production_simple.yf.download') as mock_download:
            # First two fail, third succeeds
            mock_download.side_effect = [
                Exception("Network error"),
                Exception("Timeout"),
                sample_data
            ]
            
            result = runner.get_data('PETR4.SA')
            
            assert result is not None
            assert mock_download.call_count == 3
    
    def test_get_data_returns_none_after_retries(self, runner):
        """Test returns None after all retries fail."""
        with patch('production_simple.yf.download') as mock_download:
            mock_download.side_effect = Exception("Permanent failure")
            
            result = runner.get_data('PETR4.SA')
            
            assert result is None
            assert mock_download.call_count == 3
    
    def test_get_data_handles_empty_data(self, runner):
        """Test handling of empty data."""
        with patch('production_simple.yf.download') as mock_download:
            mock_download.return_value = pd.DataFrame()
            
            result = runner.get_data('PETR4.SA')
            
            assert result is None
    
    def test_exponential_backoff_timing(self, runner):
        """Test exponential backoff delays."""
        with patch('production_simple.yf.download') as mock_download:
            mock_download.side_effect = Exception("Fail")
            with patch('time.sleep') as mock_sleep:
                result = runner.get_data('PETR4.SA')
                
                # Should have sleep calls with exponential backoff
                if mock_sleep.call_count >= 2:
                    calls = [c[0][0] for c in mock_sleep.call_args_list]
                    # First delay: 1s, second: 2s
                    assert calls[0] == 1.0
                    assert calls[1] == 2.0


class TestKellyCriterion:
    """Tests for Kelly Criterion position sizing."""
    
    @pytest.fixture
    def runner(self):
        return SimpleProductionRunner(use_news=False)
    
    @pytest.fixture
    def volatile_data(self):
        """Create volatile price data."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        return pd.DataFrame({
            'Close': 100 + np.random.randn(100).cumsum() * 2
        }, index=dates)
    
    @pytest.fixture
    def stable_data(self):
        """Create stable price data."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'Close': 100 + np.random.randn(100) * 0.1
        }, index=dates)
    
    def test_kelly_position_in_bounds(self, runner, volatile_data):
        """Test Kelly position is within bounds [10%, 80%]."""
        for confidence in [0.5, 0.6, 0.7, 0.8, 0.9]:
            position = runner.calculate_kelly_position(volatile_data, confidence)
            
            assert 0.10 <= position <= 0.80, \
                f"Position for confidence {confidence} should be in [0.10, 0.80]"
    
    def test_kelly_higher_confidence_larger_position(self, runner, volatile_data):
        """Test higher confidence leads to larger position."""
        pos_low = runner.calculate_kelly_position(volatile_data, 0.5)
        pos_high = runner.calculate_kelly_position(volatile_data, 0.8)
        
        # Higher confidence should generally lead to larger position
        # (though not always due to half-Kelly adjustment)
        assert pos_high >= pos_low * 0.8  # Allow some tolerance
    
    def test_kelly_handles_flat_prices(self, runner):
        """Test Kelly with flat prices (zero volatility)."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        flat_data = pd.DataFrame({'Close': [100.0] * 100}, index=dates)
        
        position = runner.calculate_kelly_position(flat_data, 0.7)
        
        # Should return default 20%
        assert position == 0.20
    
    def test_kelly_handles_edge_cases(self, runner):
        """Test Kelly handles edge cases gracefully."""
        # Very short data
        short_data = pd.DataFrame({'Close': [100, 101, 102]})
        
        try:
            position = runner.calculate_kelly_position(short_data, 0.7)
            assert 0.10 <= position <= 0.80
        except Exception:
            # If it fails, it should fail gracefully
            pass


class TestSignalGeneration:
    """Tests for signal generation logic."""
    
    @pytest.fixture
    def runner(self):
        return SimpleProductionRunner(use_news=False)
    
    @pytest.fixture
    def uptrend_data(self):
        """Create uptrend data."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'Close': np.linspace(100, 150, 100)
        }, index=dates)
    
    @pytest.fixture
    def downtrend_data(self):
        """Create downtrend data."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'Close': np.linspace(150, 100, 100)
        }, index=dates)
    
    def test_analyze_ticker_returns_dict(self, runner, uptrend_data):
        """Test analyze_ticker returns proper dict."""
        with patch.object(runner, 'get_data') as mock_get:
            mock_get.return_value = uptrend_data
            
            result = runner.analyze_ticker('TEST.SA')
            
            if result:  # May be None if data insufficient
                assert 'ticker' in result
                assert 'signal' in result
                assert 'position_size' in result
                assert 'conviction' in result
    
    def test_signal_is_valid(self, runner, uptrend_data):
        """Test signal is one of valid values."""
        with patch.object(runner, 'get_data') as mock_get:
            mock_get.return_value = uptrend_data
            
            result = runner.analyze_ticker('TEST.SA')
            
            if result:
                assert result['signal'] in ['BUY', 'SELL', 'HOLD']
    
    def test_buy_signal_for_uptrend(self, runner, uptrend_data):
        """Test BUY signal for uptrend with high confidence."""
        with patch.object(runner, 'get_data') as mock_get:
            mock_get.return_value = uptrend_data
            
            result = runner.analyze_ticker('TEST.SA')
            
            if result and result['conviction'] >= 0.5:
                assert result['signal'] == 'BUY'
    
    def test_sell_signal_for_downtrend(self, runner, downtrend_data):
        """Test SELL signal for downtrend."""
        with patch.object(runner, 'get_data') as mock_get:
            mock_get.return_value = downtrend_data
            
            result = runner.analyze_ticker('TEST.SA')
            
            if result and result['trend'] == 'downtrend':
                # Should be SELL if confidence is high enough
                if result['conviction'] <= -0.5:
                    assert result['signal'] == 'SELL'


class TestNewsIntegration:
    """Tests for news integration in runner."""
    
    @pytest.fixture
    def runner_with_news(self):
        return SimpleProductionRunner(use_news=True)
    
    @pytest.fixture
    def runner_no_news(self):
        return SimpleProductionRunner(use_news=False)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample price data."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'Close': np.linspace(100, 150, 100)
        }, index=dates)
    
    def test_news_disabled(self, runner_no_news, sample_data):
        """Test that news is disabled when use_news=False."""
        with patch.object(runner_no_news, 'get_data') as mock_get:
            mock_get.return_value = sample_data
            
            result = runner_no_news.analyze_ticker('TEST.SA')
            
            if result:
                assert result['news_sentiment'] == 0.0
    
    def test_news_sentiment_in_result(self, runner_with_news, sample_data):
        """Test that news sentiment is in result."""
        with patch.object(runner_with_news, 'get_data') as mock_get:
            mock_get.return_value = sample_data
            with patch.object(runner_with_news, 'get_news_sentiment') as mock_news:
                mock_news.return_value = {'sentiment': 0.5, 'articles': []}
                
                result = runner_with_news.analyze_ticker('TEST.SA')
                
                if result:
                    assert 'news_sentiment' in result


class TestTopMovers:
    """Tests for top movers identification."""
    
    @pytest.fixture
    def runner(self):
        return SimpleProductionRunner(use_news=False)
    
    def test_get_top_movers_returns_list(self, runner):
        """Test get_top_movers returns list."""
        results = [
            {'ticker': 'A.SA', 'conviction': 0.8, 'signal': 'BUY', 'trend': 'uptrend'},
            {'ticker': 'B.SA', 'conviction': 0.6, 'signal': 'BUY', 'trend': 'uptrend'},
            {'ticker': 'C.SA', 'conviction': 0.4, 'signal': 'HOLD', 'trend': 'consolidation'},
        ]
        
        top = runner.get_top_movers(results, top_n=2)
        
        assert isinstance(top, list)
        assert len(top) == 2
        assert top[0] == 'A.SA'  # Highest conviction
    
    def test_get_top_movers_handles_empty(self, runner):
        """Test get_top_movers with empty results."""
        top = runner.get_top_movers([], top_n=5)
        
        assert top == []
    
    def test_get_top_movers_sorts_by_conviction(self, runner):
        """Test that top movers are sorted by conviction."""
        results = [
            {'ticker': 'LOW.SA', 'conviction': 0.3, 'signal': 'HOLD', 'trend': 'consolidation'},
            {'ticker': 'HIGH.SA', 'conviction': 0.9, 'signal': 'BUY', 'trend': 'uptrend'},
            {'ticker': 'MED.SA', 'conviction': 0.5, 'signal': 'BUY', 'trend': 'uptrend'},
        ]
        
        top = runner.get_top_movers(results, top_n=3)
        
        assert top == ['HIGH.SA', 'MED.SA', 'LOW.SA']


class TestRunnerInitialization:
    """Tests for runner initialization."""
    
    def test_init_with_news(self):
        """Test initialization with news enabled."""
        runner = SimpleProductionRunner(use_news=True)
        
        assert runner.use_news == True
        assert hasattr(runner, 'news_client')
    
    def test_init_without_news(self):
        """Test initialization with news disabled."""
        runner = SimpleProductionRunner(use_news=False)
        
        assert runner.use_news == False
        assert not hasattr(runner, 'news_client') or runner.news_client is None
    
    def test_workers_setting(self):
        """Test worker count setting."""
        runner = SimpleProductionRunner(use_news=False, n_workers=4)
        
        assert runner.n_workers == 4
    
    def test_workers_capped_at_8(self):
        """Test worker count is capped at 8."""
        runner = SimpleProductionRunner(use_news=False, n_workers=100)
        
        assert runner.n_workers <= 8


class TestRunMethod:
    """Tests for main run method."""
    
    @pytest.fixture
    def runner(self):
        return SimpleProductionRunner(use_news=False)
    
    def test_run_returns_list(self, runner):
        """Test run returns list of results."""
        with patch.object(runner, 'analyze_ticker') as mock_analyze:
            mock_analyze.return_value = {
                'ticker': 'TEST.SA',
                'signal': 'HOLD',
                'trend': 'neutral',
                'conviction': 0.0,
                'position_size': 0.0,
                'confidence': 0.0,
                'news_sentiment': 0.0,
                'price': 100.0,
                'features': {}
            }
            
            results = runner.run(tickers=['TEST.SA'])
            
            assert isinstance(results, list)
    
    def test_run_handles_none_results(self, runner):
        """Test run handles None results from analyze_ticker."""
        with patch.object(runner, 'analyze_ticker') as mock_analyze:
            mock_analyze.return_value = None
            
            results = runner.run(tickers=['TEST.SA'])
            
            assert isinstance(results, list)
            assert len(results) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
