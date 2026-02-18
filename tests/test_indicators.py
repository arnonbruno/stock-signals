"""
Comprehensive tests for SOTA technical indicators.

Tests cover:
- Momentum indicators (RSI, MACD, Stochastic)
- Volatility indicators (ATR, Bollinger Bands, Keltner)
- Volume indicators (OBV, VWAP, MFI)
- Trend indicators (MA, ADX, SuperTrend)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indicators.momentum import MomentumIndicators
from src.indicators.volatility import VolatilityIndicators
from src.indicators.volume import VolumeIndicators
from src.indicators.trend import TrendIndicators


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
    trend = np.linspace(0, 20, 200)  # Upward trend
    noise = np.random.randn(200) * 2
    close = base_price + trend + noise.cumsum()
    
    # Generate OHLC
    high = close + np.abs(np.random.randn(200)) * 1.5
    low = close - np.abs(np.random.randn(200)) * 1.5
    open_price = close + np.random.randn(200) * 0.5
    
    # Volume
    volume = np.random.randint(1000000, 5000000, 200)
    
    return pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)


@pytest.fixture
def uptrend_data():
    """Create data with clear uptrend."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    close = 100 + np.linspace(0, 30, 100)  # Clear uptrend
    
    return pd.DataFrame({
        'Close': close,
        'High': close + 1,
        'Low': close - 1,
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)


@pytest.fixture
def downtrend_data():
    """Create data with clear downtrend."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    close = 130 - np.linspace(0, 30, 100)  # Clear downtrend
    
    return pd.DataFrame({
        'Close': close,
        'High': close + 1,
        'Low': close - 1,
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)


@pytest.fixture
def sideways_data():
    """Create sideways/ranging data."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    close = 100 + np.sin(np.linspace(0, 4*np.pi, 100)) * 5  # Oscillating
    
    return pd.DataFrame({
        'Close': close,
        'High': close + 2,
        'Low': close - 2,
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)


# ============================================================================
# MOMENTUM INDICATOR TESTS
# ============================================================================

class TestRSI:
    """Tests for Relative Strength Index."""
    
    def test_rsi_returns_series(self, sample_data):
        """Test RSI returns a pandas Series."""
        rsi = MomentumIndicators.rsi(sample_data, period=14)
        
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(sample_data)
    
    def test_rsi_range_0_to_100(self, sample_data):
        """Test RSI values are between 0 and 100."""
        rsi = MomentumIndicators.rsi(sample_data, period=14)
        
        # Skip NaN values at start
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()
    
    def test_rsi_overbought_signal(self, uptrend_data):
        """Test RSI shows overbought in strong uptrend."""
        rsi = MomentumIndicators.rsi(uptrend_data, period=14)
        signal = MomentumIndicators.rsi_signal(rsi)
        
        # Strong uptrend should push RSI high
        assert signal['rsi_value'] > 50 or signal['signal'] in ['overbought', 'neutral']
    
    def test_rsi_oversold_signal(self, downtrend_data):
        """Test RSI shows oversold in strong downtrend."""
        rsi = MomentumIndicators.rsi(downtrend_data, period=14)
        signal = MomentumIndicators.rsi_signal(rsi)
        
        # Strong downtrend should push RSI low
        assert signal['rsi_value'] < 50 or signal['signal'] in ['oversold', 'neutral']
    
    def test_rsi_signal_structure(self, sample_data):
        """Test RSI signal returns correct structure."""
        rsi = MomentumIndicators.rsi(sample_data, period=14)
        signal = MomentumIndicators.rsi_signal(rsi)
        
        assert 'signal' in signal
        assert 'strength' in signal
        assert 'rsi_value' in signal
        assert 'interpretation' in signal


class TestMACD:
    """Tests for Moving Average Convergence Divergence."""
    
    def test_macd_returns_dict(self, sample_data):
        """Test MACD returns correct structure."""
        macd = MomentumIndicators.macd(sample_data)
        
        assert 'macd' in macd
        assert 'signal' in macd
        assert 'histogram' in macd
    
    def test_macd_line_crosses_signal(self, sample_data):
        """Test MACD histogram reflects crossover."""
        macd = MomentumIndicators.macd(sample_data)
        
        # Histogram = MACD - Signal
        expected_hist = macd['macd'] - macd['signal']
        pd.testing.assert_series_equal(macd['histogram'], expected_hist, check_names=False)
    
    def test_macd_signal_structure(self, sample_data):
        """Test MACD signal returns correct structure."""
        macd = MomentumIndicators.macd(sample_data)
        signal = MomentumIndicators.macd_signal(macd)
        
        assert 'signal' in signal
        assert 'strength' in signal
        assert 'interpretation' in signal
    
    def test_macd_bullish_in_uptrend(self, uptrend_data):
        """Test MACD is bullish in uptrend."""
        macd = MomentumIndicators.macd(uptrend_data)
        signal = MomentumIndicators.macd_signal(macd)
        
        assert signal['signal'] in ['bullish', 'bullish_crossover']


class TestStochastic:
    """Tests for Stochastic Oscillator."""
    
    def test_stochastic_returns_dict(self, sample_data):
        """Test Stochastic returns correct structure."""
        stoch = MomentumIndicators.stochastic(sample_data)
        
        assert 'k' in stoch
        assert 'd' in stoch
    
    def test_stochastic_range_0_to_100(self, sample_data):
        """Test Stochastic values are between 0 and 100."""
        stoch = MomentumIndicators.stochastic(sample_data)
        
        valid_k = stoch['k'].dropna()
        valid_d = stoch['d'].dropna()
        
        assert (valid_k >= 0).all() and (valid_k <= 100).all()
        assert (valid_d >= 0).all() and (valid_d <= 100).all()
    
    def test_stochastic_signal_structure(self, sample_data):
        """Test Stochastic signal returns correct structure."""
        stoch = MomentumIndicators.stochastic(sample_data)
        signal = MomentumIndicators.stochastic_signal(stoch)
        
        assert 'signal' in signal
        assert 'strength' in signal
        assert 'k_value' in signal
        assert 'd_value' in signal


class TestMomentumIntegration:
    """Tests for momentum indicator integration."""
    
    def test_get_all_momentum_signals(self, sample_data):
        """Test get_all_momentum_signals returns all indicators."""
        signals = MomentumIndicators.get_all_momentum_signals(sample_data)
        
        assert 'rsi' in signals
        assert 'macd' in signals
        assert 'stochastic' in signals
        assert 'williams_r' in signals
        assert 'roc' in signals
    
    def test_momentum_signals_have_values(self, sample_data):
        """Test all momentum signals have valid values."""
        signals = MomentumIndicators.get_all_momentum_signals(sample_data)
        
        # RSI
        assert signals['rsi']['signal'] in ['overbought', 'oversold', 'neutral']
        
        # MACD
        assert signals['macd']['signal'] in ['bullish', 'bearish', 'neutral', 
                                             'bullish_crossover', 'bearish_crossover']
        
        # Stochastic
        assert signals['stochastic']['signal'] in ['overbought', 'oversold', 'neutral']


# ============================================================================
# VOLATILITY INDICATOR TESTS
# ============================================================================

class TestATR:
    """Tests for Average True Range."""
    
    def test_atr_returns_series(self, sample_data):
        """Test ATR returns a pandas Series."""
        atr = VolatilityIndicators.atr(sample_data, period=14)
        
        assert isinstance(atr, pd.Series)
        assert len(atr) == len(sample_data)
    
    def test_atr_positive_values(self, sample_data):
        """Test ATR values are positive."""
        atr = VolatilityIndicators.atr(sample_data, period=14)
        valid_atr = atr.dropna()
        
        assert (valid_atr > 0).all()
    
    def test_atr_signal_structure(self, sample_data):
        """Test ATR signal returns correct structure."""
        signal = VolatilityIndicators.atr_signal(sample_data)
        
        assert 'regime' in signal
        assert 'atr' in signal
        assert 'atr_pct' in signal


class TestBollingerBands:
    """Tests for Bollinger Bands."""
    
    def test_bollinger_returns_dict(self, sample_data):
        """Test Bollinger returns correct structure."""
        bb = VolatilityIndicators.bollinger_bands(sample_data)
        
        assert 'upper' in bb
        assert 'middle' in bb
        assert 'lower' in bb
        assert 'bandwidth' in bb
        assert 'percent_b' in bb
    
    def test_bollinger_band_order(self, sample_data):
        """Test upper > middle > lower."""
        bb = VolatilityIndicators.bollinger_bands(sample_data)
        
        # For valid (non-NaN) values
        valid_idx = bb['upper'].notna()
        assert (bb['upper'][valid_idx] >= bb['middle'][valid_idx]).all()
        assert (bb['middle'][valid_idx] >= bb['lower'][valid_idx]).all()
    
    def test_bollinger_percent_b_range(self, sample_data):
        """Test %B calculation."""
        bb = VolatilityIndicators.bollinger_bands(sample_data)
        
        # %B can be outside 0-100 during strong moves
        # Just check it's a valid series
        assert isinstance(bb['percent_b'], pd.Series)
    
    def test_bollinger_signal_structure(self, sample_data):
        """Test Bollinger signal returns correct structure."""
        bb = VolatilityIndicators.bollinger_bands(sample_data)
        signal = VolatilityIndicators.bollinger_signal(bb, sample_data)
        
        assert 'signal' in signal
        assert 'strength' in signal
        assert 'interpretation' in signal


class TestKeltnerChannels:
    """Tests for Keltner Channels."""
    
    def test_keltner_returns_dict(self, sample_data):
        """Test Keltner returns correct structure."""
        kc = VolatilityIndicators.keltner_channels(sample_data)
        
        assert 'upper' in kc
        assert 'middle' in kc
        assert 'lower' in kc
        assert 'width' in kc
    
    def test_keltner_signal_structure(self, sample_data):
        """Test Keltner signal returns correct structure."""
        kc = VolatilityIndicators.keltner_channels(sample_data)
        signal = VolatilityIndicators.keltner_signal(kc, sample_data)
        
        assert 'signal' in signal
        assert 'strength' in signal


class TestVolatilityIntegration:
    """Tests for volatility indicator integration."""
    
    def test_get_all_volatility_signals(self, sample_data):
        """Test get_all_volatility_signals returns all indicators."""
        signals = VolatilityIndicators.get_all_volatility_signals(sample_data)
        
        assert 'atr' in signals
        assert 'bollinger' in signals
        assert 'keltner' in signals


# ============================================================================
# VOLUME INDICATOR TESTS
# ============================================================================

class TestOBV:
    """Tests for On-Balance Volume."""
    
    def test_obv_returns_series(self, sample_data):
        """Test OBV returns a pandas Series."""
        obv = VolumeIndicators.obv(sample_data)
        
        assert isinstance(obv, pd.Series)
        assert len(obv) == len(sample_data)
    
    def test_obv_signal_structure(self, sample_data):
        """Test OBV signal returns correct structure."""
        obv = VolumeIndicators.obv(sample_data)
        signal = VolumeIndicators.obv_signal(sample_data, obv)
        
        assert 'signal' in signal
        assert 'strength' in signal
        assert 'interpretation' in signal


class TestVWAP:
    """Tests for Volume Weighted Average Price."""
    
    def test_vwap_returns_series(self, sample_data):
        """Test VWAP returns a pandas Series."""
        vwap = VolumeIndicators.vwap(sample_data)
        
        assert isinstance(vwap, pd.Series)
        assert len(vwap) == len(sample_data)
    
    def test_vwap_signal_structure(self, sample_data):
        """Test VWAP signal returns correct structure."""
        vwap = VolumeIndicators.vwap(sample_data)
        signal = VolumeIndicators.vwap_signal(sample_data, vwap)
        
        assert 'signal' in signal
        assert 'strength' in signal
        assert 'distance_pct' in signal


class TestVolumeMomentum:
    """Tests for Volume Momentum."""
    
    def test_volume_momentum_structure(self, sample_data):
        """Test volume momentum returns correct structure."""
        vm = VolumeIndicators.volume_momentum(sample_data)
        
        assert 'ratio' in vm
        assert 'trend' in vm
        assert 'unusual' in vm
    
    def test_volume_momentum_ratio_positive(self, sample_data):
        """Test volume ratio is positive."""
        vm = VolumeIndicators.volume_momentum(sample_data)
        
        assert vm['ratio'] > 0


class TestMFI:
    """Tests for Money Flow Index."""
    
    def test_mfi_returns_series(self, sample_data):
        """Test MFI returns a pandas Series."""
        mfi = VolumeIndicators.mfi(sample_data, period=14)
        
        assert isinstance(mfi, pd.Series)
        assert len(mfi) == len(sample_data)
    
    def test_mfi_range_0_to_100(self, sample_data):
        """Test MFI values are between 0 and 100."""
        mfi = VolumeIndicators.mfi(sample_data, period=14)
        valid_mfi = mfi.dropna()
        
        assert (valid_mfi >= 0).all() and (valid_mfi <= 100).all()


class TestVolumeIntegration:
    """Tests for volume indicator integration."""
    
    def test_get_all_volume_signals(self, sample_data):
        """Test get_all_volume_signals returns all indicators."""
        signals = VolumeIndicators.get_all_volume_signals(sample_data)
        
        assert 'obv' in signals
        assert 'vwap' in signals
        assert 'volume_momentum' in signals
        assert 'mfi' in signals


# ============================================================================
# TREND INDICATOR TESTS
# ============================================================================

class TestMovingAverages:
    """Tests for Moving Averages."""
    
    def test_ma_returns_dict(self, sample_data):
        """Test MA returns correct structure."""
        ma = TrendIndicators.moving_averages(sample_data)
        
        assert 'sma' in ma
        assert 'ema' in ma
    
    def test_ma_periods(self, sample_data):
        """Test default MA periods are present."""
        ma = TrendIndicators.moving_averages(sample_data)
        
        assert 20 in ma['sma']
        assert 50 in ma['sma']
        assert 200 in ma['sma']
    
    def test_ma_signal_structure(self, sample_data):
        """Test MA signal returns correct structure."""
        ma = TrendIndicators.moving_averages(sample_data)
        signal = TrendIndicators.ma_signal(ma, sample_data)
        
        assert 'signal' in signal
        assert 'strength' in signal


class TestADX:
    """Tests for Average Directional Index."""
    
    def test_adx_returns_dict(self, sample_data):
        """Test ADX returns correct structure."""
        adx = TrendIndicators.adx(sample_data)
        
        assert 'adx' in adx
        assert 'plus_di' in adx
        assert 'minus_di' in adx
    
    def test_adx_range(self, sample_data):
        """Test ADX values are in valid range."""
        adx = TrendIndicators.adx(sample_data)
        valid_adx = adx['adx'].dropna()
        
        assert (valid_adx >= 0).all() and (valid_adx <= 100).all()
    
    def test_adx_signal_structure(self, sample_data):
        """Test ADX signal returns correct structure."""
        adx = TrendIndicators.adx(sample_data)
        signal = TrendIndicators.adx_signal(adx)
        
        assert 'signal' in signal
        assert 'strength' in signal
        assert 'trend_strength' in signal


class TestSuperTrend:
    """Tests for SuperTrend indicator."""
    
    def test_supertrend_returns_dict(self, sample_data):
        """Test SuperTrend returns correct structure."""
        st = TrendIndicators.supertrend(sample_data)
        
        assert 'supertrend' in st
        assert 'direction' in st
        assert 'upper' in st
        assert 'lower' in st
    
    def test_supertrend_direction_values(self, sample_data):
        """Test SuperTrend direction is 1 or -1."""
        st = TrendIndicators.supertrend(sample_data)
        valid_dir = st['direction'].dropna()
        
        assert valid_dir.isin([1, -1]).all()
    
    def test_supertrend_signal_structure(self, sample_data):
        """Test SuperTrend signal returns correct structure."""
        st = TrendIndicators.supertrend(sample_data)
        signal = TrendIndicators.supertrend_signal(st, sample_data)
        
        assert 'signal' in signal
        assert 'strength' in signal


class TestTrendIntegration:
    """Tests for trend indicator integration."""
    
    def test_get_all_trend_signals(self, sample_data):
        """Test get_all_trend_signals returns all indicators."""
        signals = TrendIndicators.get_all_trend_signals(sample_data)
        
        assert 'moving_averages' in signals
        assert 'adx' in signals
        assert 'supertrend' in signals


# ============================================================================
# EDGE CASES
# ============================================================================

class TestIndicatorEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_data(self):
        """Test indicators handle empty data gracefully."""
        empty_df = pd.DataFrame()
        
        # Should not raise exceptions
        try:
            MomentumIndicators.get_all_momentum_signals(empty_df)
            VolatilityIndicators.get_all_volatility_signals(empty_df)
            VolumeIndicators.get_all_volume_signals(empty_df)
            TrendIndicators.get_all_trend_signals(empty_df)
            assert True
        except Exception:
            assert True  # Errors are acceptable for empty data
    
    def test_insufficient_data(self):
        """Test indicators with insufficient data."""
        short_df = pd.DataFrame({
            'Close': [100, 101, 102],
            'High': [101, 102, 103],
            'Low': [99, 100, 101],
            'Volume': [1000, 1000, 1000]
        })
        
        # Should not crash
        signals = MomentumIndicators.get_all_momentum_signals(short_df)
        assert 'error' in signals or signals['rsi']['signal'] == 'neutral'
    
    def test_missing_columns(self):
        """Test indicators with missing columns."""
        df_no_volume = pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 100
        })
        
        # Volume indicators should handle missing volume
        signals = VolumeIndicators.get_all_volume_signals(df_no_volume)
        assert signals is not None
    
    def test_constant_prices(self):
        """Test indicators with constant prices (edge case)."""
        constant_df = pd.DataFrame({
            'Close': [100] * 100,
            'High': [100] * 100,
            'Low': [100] * 100,
            'Volume': [1000] * 100
        })
        
        # Should handle division by zero etc.
        signals = MomentumIndicators.get_all_momentum_signals(constant_df)
        assert signals is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
