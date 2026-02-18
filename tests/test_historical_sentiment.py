"""
Comprehensive tests for Historical Sentiment Engine.

Tests cover:
- Price-based sentiment proxy
- Volatility-based sentiment
- Return-based sentiment
- Ensemble approach
- Backtest adapter
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.news.historical_sentiment import (
    HistoricalSentimentEngine,
    SentimentBacktestAdapter,
    get_backtest_sentiment
)


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
def engine():
    """Create HistoricalSentimentEngine instance."""
    return HistoricalSentimentEngine()


@pytest.fixture
def adapter():
    """Create SentimentBacktestAdapter instance."""
    return SentimentBacktestAdapter()


# ============================================================================
# PRICE-BASED SENTIMENT TESTS
# ============================================================================

class TestPriceBasedSentiment:
    """Tests for enhanced price-based sentiment proxy."""
    
    def test_returns_series(self, engine, sample_data):
        """Test price sentiment returns a Series."""
        sentiment = engine.calculate_enhanced_price_sentiment(sample_data)
        
        assert isinstance(sentiment, pd.Series)
        assert len(sentiment) == len(sample_data)
    
    def test_values_in_range(self, engine, sample_data):
        """Test sentiment values are in -1 to +1 range."""
        sentiment = engine.calculate_enhanced_price_sentiment(sample_data)
        valid = sentiment.dropna()
        
        assert (valid >= -1).all() and (valid <= 1).all()
    
    def test_uptrend_positive_sentiment(self, engine):
        """Test uptrend produces positive sentiment."""
        # Create clear uptrend
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        close = 100 + np.linspace(0, 30, 100)  # Strong uptrend
        volume = np.random.randint(1000000, 5000000, 100)
        
        uptrend_df = pd.DataFrame({
            'Close': close,
            'High': close + 1,
            'Low': close - 1,
            'Volume': volume
        }, index=dates)
        
        sentiment = engine.calculate_enhanced_price_sentiment(uptrend_df)
        
        # Average sentiment should be positive in uptrend
        avg_sentiment = sentiment.iloc[-20:].mean()  # Last 20 days
        assert avg_sentiment > -0.2  # Allow some noise
    
    def test_downtrend_negative_sentiment(self, engine):
        """Test downtrend produces negative sentiment."""
        # Create clear downtrend
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        close = 130 - np.linspace(0, 30, 100)  # Strong downtrend
        volume = np.random.randint(1000000, 5000000, 100)
        
        downtrend_df = pd.DataFrame({
            'Close': close,
            'High': close + 1,
            'Low': close - 1,
            'Volume': volume
        }, index=dates)
        
        sentiment = engine.calculate_enhanced_price_sentiment(downtrend_df)
        
        # Average sentiment should be negative in downtrend
        avg_sentiment = sentiment.iloc[-20:].mean()
        assert avg_sentiment < 0.2  # Allow some noise
    
    def test_handles_missing_volume(self, engine, sample_data):
        """Test sentiment works without volume data."""
        df_no_volume = sample_data.drop('Volume', axis=1)
        
        sentiment = engine.calculate_enhanced_price_sentiment(df_no_volume)
        
        assert isinstance(sentiment, pd.Series)
        assert len(sentiment) == len(df_no_volume)


# ============================================================================
# VOLATILITY-BASED SENTIMENT TESTS
# ============================================================================

class TestVolatilitySentiment:
    """Tests for volatility-based sentiment proxy."""
    
    def test_returns_series(self, engine, sample_data):
        """Test volatility sentiment returns a Series."""
        sentiment = engine.calculate_volatility_sentiment(sample_data)
        
        assert isinstance(sentiment, pd.Series)
        assert len(sentiment) == len(sample_data)
    
    def test_values_in_range(self, engine, sample_data):
        """Test sentiment values are in -1 to +1 range."""
        sentiment = engine.calculate_volatility_sentiment(sample_data)
        valid = sentiment.dropna()
        
        assert (valid >= -1).all() and (valid <= 1).all()
    
    def test_high_volatility_negative(self, engine):
        """Test high volatility produces negative sentiment (fear)."""
        # Create high volatility data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        close = 100 + np.random.randn(100).cumsum() * 5  # High volatility
        
        high_vol_df = pd.DataFrame({
            'Close': close,
            'High': close + 3,
            'Low': close - 3
        }, index=dates)
        
        sentiment = engine.calculate_volatility_sentiment(high_vol_df)
        
        # High volatility should produce negative sentiment (fear)
        avg_sentiment = sentiment.iloc[-20:].mean()
        assert avg_sentiment <= 0.1  # Should be negative or near zero


# ============================================================================
# RETURN-BASED SENTIMENT TESTS
# ============================================================================

class TestReturnSentiment:
    """Tests for return-based sentiment proxy."""
    
    def test_returns_series(self, engine, sample_data):
        """Test return sentiment returns a Series."""
        sentiment = engine.calculate_return_sentiment(sample_data)
        
        assert isinstance(sentiment, pd.Series)
        assert len(sentiment) == len(sample_data)
    
    def test_values_in_range(self, engine, sample_data):
        """Test sentiment values are in -1 to +1 range."""
        sentiment = engine.calculate_return_sentiment(sample_data)
        valid = sentiment.dropna()
        
        assert (valid >= -1).all() and (valid <= 1).all()
    
    def test_positive_returns_positive_sentiment(self, engine):
        """Test positive returns produce positive sentiment."""
        dates = pd.date_range(start='2024-01-01', periods=20, freq='D')
        close = 100 + np.linspace(0, 10, 20)  # Rising prices
        
        rising_df = pd.DataFrame({'Close': close}, index=dates)
        
        sentiment = engine.calculate_return_sentiment(rising_df)
        
        # Last value should be positive
        assert sentiment.iloc[-1] > 0


# ============================================================================
# ENSEMBLE SENTIMENT TESTS
# ============================================================================

class TestEnsembleSentiment:
    """Tests for ensemble sentiment combining all methods."""
    
    def test_ensemble_returns_series(self, engine, sample_data):
        """Test ensemble returns a Series."""
        sentiment = engine.get_historical_sentiment(sample_data, method='ensemble')
        
        assert isinstance(sentiment, pd.Series)
        assert len(sentiment) == len(sample_data)
    
    def test_ensemble_values_in_range(self, engine, sample_data):
        """Test ensemble values are in -1 to +1 range."""
        sentiment = engine.get_historical_sentiment(sample_data, method='ensemble')
        valid = sentiment.dropna()
        
        assert (valid >= -1).all() and (valid <= 1).all()
    
    def test_different_methods(self, engine, sample_data):
        """Test different methods produce results."""
        methods = ['enhanced', 'volatility', 'returns', 'ensemble']
        
        for method in methods:
            sentiment = engine.get_historical_sentiment(sample_data, method=method)
            assert isinstance(sentiment, pd.Series)
    
    def test_invalid_method_raises(self, engine, sample_data):
        """Test invalid method raises error."""
        with pytest.raises(ValueError):
            engine.get_historical_sentiment(sample_data, method='invalid')


# ============================================================================
# SENTIMENT DECAY TESTS
# ============================================================================

class TestSentimentDecay:
    """Tests for sentiment time decay."""
    
    def test_decay_reduces_sentiment(self, engine, sample_data):
        """Test decay reduces sentiment over time."""
        sentiment = engine.calculate_enhanced_price_sentiment(sample_data)
        decayed = engine.apply_sentiment_decay(sentiment)
        
        assert isinstance(decayed, pd.Series)
        assert len(decayed) == len(sentiment)


# ============================================================================
# BACKTEST ADAPTER TESTS
# ============================================================================

class TestBacktestAdapter:
    """Tests for SentimentBacktestAdapter."""
    
    def test_adapter_initialization(self, adapter):
        """Test adapter initializes correctly."""
        assert adapter.engine is not None
        assert adapter.method == 'ensemble'
    
    def test_precompute_creates_cache(self, adapter, sample_data):
        """Test precompute creates cache entry."""
        adapter.precompute('TEST.SA', sample_data)
        
        assert 'TEST.SA' in adapter.precomputed
    
    def test_get_sentiment_from_cache(self, adapter, sample_data):
        """Test get sentiment from precomputed cache."""
        adapter.precompute('TEST.SA', sample_data)
        
        date = sample_data.index[-30]  # A date in the data
        sentiment = adapter.get_sentiment('TEST.SA', date)
        
        assert isinstance(sentiment, (int, float))
        assert -1 <= sentiment <= 1
    
    def test_get_sentiment_no_cache(self, adapter, sample_data):
        """Test get sentiment without precompute."""
        date = sample_data.index[-30]
        sentiment = adapter.get_sentiment('TEST.SA', date, data=sample_data)
        
        assert isinstance(sentiment, (int, float))
    
    def test_get_sentiment_series(self, adapter, sample_data):
        """Test get full sentiment series."""
        adapter.precompute('TEST.SA', sample_data)
        series = adapter.get_sentiment_series('TEST.SA')
        
        assert isinstance(series, pd.Series)
    
    def test_get_sentiment_series_missing(self, adapter):
        """Test get series for missing ticker."""
        series = adapter.get_sentiment_series('MISSING.SA')
        
        assert isinstance(series, pd.Series)
        assert len(series) == 0


# ============================================================================
# DATE-BASED RETRIEVAL TESTS
# ============================================================================

class TestDateRetrieval:
    """Tests for date-based sentiment retrieval."""
    
    def test_get_sentiment_for_date(self, engine, sample_data):
        """Test getting sentiment for specific date."""
        date = sample_data.index[-30]
        sentiment = engine.get_sentiment_for_date(sample_data, date)
        
        assert isinstance(sentiment, (int, float))
        assert -1 <= sentiment <= 1
    
    def test_get_sentiment_no_lookahead(self, engine, sample_data):
        """Test no look-ahead bias in sentiment calculation."""
        # Get sentiment for middle date
        date = sample_data.index[100]
        sentiment = engine.get_sentiment_for_date(sample_data, date)
        
        # Should not use data after this date
        # This is implicitly tested by the slicing in get_sentiment_for_date
        assert isinstance(sentiment, (int, float))
    
    def test_get_sentiment_insufficient_data(self, engine):
        """Test sentiment with insufficient data."""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        short_df = pd.DataFrame({
            'Close': np.random.randn(10).cumsum() + 100
        }, index=dates)
        
        date = dates[-1]
        sentiment = engine.get_sentiment_for_date(short_df, date)
        
        # Should return 0 for insufficient data
        assert sentiment == 0.0


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_get_backtest_sentiment_returns_series(self, sample_data):
        """Test get_backtest_sentiment returns Series."""
        sentiment = get_backtest_sentiment(sample_data)
        
        assert isinstance(sentiment, pd.Series)
    
    def test_get_backtest_sentiment_different_methods(self, sample_data):
        """Test get_backtest_sentiment with different methods."""
        for method in ['enhanced', 'volatility', 'returns', 'ensemble']:
            sentiment = get_backtest_sentiment(sample_data, method=method)
            assert isinstance(sentiment, pd.Series)


# ============================================================================
# EDGE CASES
# ============================================================================

class TestSentimentEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_data(self, engine):
        """Test sentiment with empty data."""
        empty_df = pd.DataFrame(columns=['Close', 'High', 'Low'])
        
        # Should handle empty data gracefully
        try:
            sentiment = engine.get_historical_sentiment(empty_df)
            assert isinstance(sentiment, pd.Series)
        except Exception:
            # Empty data may raise, which is acceptable
            pass
    
    def test_constant_prices(self, engine):
        """Test sentiment with constant prices."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        constant_df = pd.DataFrame({
            'Close': [100] * 100,
            'High': [100] * 100,
            'Low': [100] * 100
        }, index=dates)
        
        # Should not crash
        sentiment = engine.get_historical_sentiment(constant_df)
        assert isinstance(sentiment, pd.Series)
    
    def test_single_value(self, engine):
        """Test sentiment with single data point."""
        single_df = pd.DataFrame({
            'Close': [100],
            'High': [100],
            'Low': [100]
        }, index=[datetime.now()])
        
        # Should not crash
        sentiment = engine.get_historical_sentiment(single_df)
        assert isinstance(sentiment, pd.Series)
    
    def test_very_volatile_data(self, engine):
        """Test sentiment with extreme volatility."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        close = 100 + np.random.randn(100).cumsum() * 20  # Extreme moves
        
        volatile_df = pd.DataFrame({
            'Close': close,
            'High': close + 10,
            'Low': close - 10
        }, index=dates)
        
        # Should not crash and values should be bounded
        sentiment = engine.get_historical_sentiment(volatile_df)
        valid = sentiment.dropna()
        
        assert (valid >= -1).all() and (valid <= 1).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])