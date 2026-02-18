"""
Volume Indicators - OBV, VWAP, Volume Profile

Volume analysis for trend confirmation and divergence detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class VolumeIndicators:
    """
    Volume-based technical indicators.
    
    Volume confirms price movements:
    - Price up + Volume up: Strong bullish
    - Price up + Volume down: Weak bullish (divergence)
    - Price down + Volume up: Strong bearish
    - Price down + Volume down: Weak bearish (divergence)
    """
    
    @staticmethod
    def obv(data: pd.DataFrame) -> pd.Series:
        """
        On-Balance Volume (OBV)
        
        Cumulative volume indicator that adds volume on up days
        and subtracts on down days.
        
        Args:
            data: DataFrame with 'Close', 'Volume' columns
        
        Returns:
            OBV values
        """
        close = data['Close'].squeeze()
        volume = data['Volume'].squeeze() if 'Volume' in data.columns else pd.Series(0, index=data.index)
        
        # Direction: +1 if close > prev close, -1 if close < prev close, 0 otherwise
        direction = np.sign(close.diff())
        
        # OBV = cumulative sum of (direction * volume)
        obv = (direction * volume).cumsum()
        
        return obv
    
    @staticmethod
    def obv_signal(data: pd.DataFrame, obv: pd.Series = None) -> Dict:
        """
        Generate signal from OBV.
        
        Returns:
            Dict with signal, strength, and interpretation
        """
        if obv is None:
            obv = VolumeIndicators.obv(data)
        
        if len(obv) < 20 or pd.isna(obv.iloc[-1]):
            return {'signal': 'neutral', 'strength': 0}
        
        close = data['Close'].squeeze()
        
        # OBV trend
        obv_ma = obv.rolling(20).mean()
        obv_trend = (obv.iloc[-1] - obv_ma.iloc[-1]) / abs(obv_ma.iloc[-1]) if obv_ma.iloc[-1] != 0 else 0
        
        # Price trend
        price_trend = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20] if close.iloc[-20] != 0 else 0
        
        # Divergence detection
        if price_trend > 0.05 and obv_trend < -0.1:
            signal = 'bearish_divergence'
            strength = abs(obv_trend)
            interpretation = f'Bearish divergence: price up, OBV down'
        
        elif price_trend < -0.05 and obv_trend > 0.1:
            signal = 'bullish_divergence'
            strength = abs(obv_trend)
            interpretation = f'Bullish divergence: price down, OBV up'
        
        elif obv_trend > 0.1:
            signal = 'accumulation'
            strength = obv_trend
            interpretation = f'OBV rising - accumulation in progress'
        
        elif obv_trend < -0.1:
            signal = 'distribution'
            strength = abs(obv_trend)
            interpretation = f'OBV falling - distribution in progress'
        
        else:
            signal = 'neutral'
            strength = 0
            interpretation = 'No clear volume trend'
        
        return {
            'signal': signal,
            'strength': min(1.0, abs(obv_trend)),
            'obv_trend': obv_trend,
            'price_trend': price_trend,
            'interpretation': interpretation
        }
    
    @staticmethod
    def vwap(data: pd.DataFrame, period: int = None) -> pd.Series:
        """
        Volume Weighted Average Price (VWAP)
        
        Average price weighted by volume. Used as a benchmark
        for institutional trading.
        
        - Price > VWAP: Bullish
        - Price < VWAP: Bearish
        
        Args:
            data: DataFrame with 'High', 'Low', 'Close', 'Volume' columns
            period: Lookback period (None = from start, typical = intraday)
        
        Returns:
            VWAP values
        """
        high = data['High'].squeeze() if 'High' in data.columns else data['Close'].squeeze()
        low = data['Low'].squeeze() if 'Low' in data.columns else data['Close'].squeeze()
        close = data['Close'].squeeze()
        volume = data['Volume'].squeeze() if 'Volume' in data.columns else pd.Series(1, index=data.index)
        
        # Typical price
        typical_price = (high + low + close) / 3
        
        # VWAP = cumulative(TP * Volume) / cumulative(Volume)
        if period:
            tp_vol = typical_price * volume
            vwap = tp_vol.rolling(window=period).sum() / volume.rolling(window=period).sum()
        else:
            vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        return vwap
    
    @staticmethod
    def vwap_signal(data: pd.DataFrame, vwap: pd.Series = None) -> Dict:
        """
        Generate signal from VWAP.
        
        Returns:
            Dict with signal, strength, and interpretation
        """
        if vwap is None:
            vwap = VolumeIndicators.vwap(data)
        
        if len(vwap) == 0 or pd.isna(vwap.iloc[-1]):
            return {'signal': 'neutral', 'strength': 0}
        
        close = data['Close'].squeeze()
        current_price = close.iloc[-1]
        current_vwap = vwap.iloc[-1]
        
        # Distance from VWAP
        distance_pct = (current_price - current_vwap) / current_vwap * 100
        
        if current_price > current_vwap:
            signal = 'above_vwap'
            strength = min(1.0, distance_pct / 5)  # 5% = full strength
            interpretation = f'Above VWAP by {distance_pct:.2f}% - bullish'
        else:
            signal = 'below_vwap'
            strength = min(1.0, abs(distance_pct) / 5)
            interpretation = f'Below VWAP by {distance_pct:.2f}% - bearish'
        
        return {
            'signal': signal,
            'strength': strength,
            'distance_pct': distance_pct,
            'vwap_value': current_vwap,
            'interpretation': interpretation
        }
    
    @staticmethod
    def volume_momentum(data: pd.DataFrame, short_period: int = 5, 
                        long_period: int = 20) -> Dict:
        """
        Volume Momentum Analysis
        
        Compares recent volume to historical average.
        
        Args:
            data: DataFrame with 'Volume' column
            short_period: Short-term volume period
            long_period: Long-term volume period
        
        Returns:
            Dict with volume momentum metrics
        """
        volume = data['Volume'].squeeze() if 'Volume' in data.columns else pd.Series(1, index=data.index)
        
        if len(volume) < long_period:
            return {'ratio': 1.0, 'trend': 'unknown', 'unusual': False}
        
        short_avg = volume.rolling(short_period).mean().iloc[-1]
        long_avg = volume.rolling(long_period).mean().iloc[-1]
        
        ratio = short_avg / long_avg if long_avg > 0 else 1
        
        # Volume trend
        vol_slope = np.polyfit(range(10), volume.iloc[-10:].values, 1)[0]
        trend = 'increasing' if vol_slope > 0 else 'decreasing'
        
        # Unusual volume detection
        vol_std = volume.rolling(long_period).std().iloc[-1]
        current_vol = volume.iloc[-1]
        unusual = current_vol > (long_avg + 2 * vol_std)
        
        return {
            'ratio': ratio,
            'trend': trend,
            'unusual': unusual,
            'current_volume': current_vol,
            'avg_volume': long_avg
        }
    
    @staticmethod
    def mfi(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Money Flow Index (MFI)
        
        Volume-weighted RSI. Uses price and volume to identify
        overbought/oversold conditions.
        
        MFI > 80: Overbought
        MFI < 20: Oversold
        
        Args:
            data: DataFrame with 'High', 'Low', 'Close', 'Volume' columns
            period: Lookback period
        
        Returns:
            MFI values (0-100)
        """
        high = data['High'].squeeze() if 'High' in data.columns else data['Close'].squeeze()
        low = data['Low'].squeeze() if 'Low' in data.columns else data['Close'].squeeze()
        close = data['Close'].squeeze()
        volume = data['Volume'].squeeze() if 'Volume' in data.columns else pd.Series(1, index=data.index)
        
        # Typical price
        typical_price = (high + low + close) / 3
        
        # Money flow
        money_flow = typical_price * volume
        
        # Positive and negative money flow
        positive_mf = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_mf = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        # Money flow ratio
        positive_sum = positive_mf.rolling(window=period).sum()
        negative_sum = negative_mf.rolling(window=period).sum()
        
        mf_ratio = positive_sum / negative_sum
        
        # MFI
        mfi = 100 - (100 / (1 + mf_ratio))
        
        return mfi
    
    @staticmethod
    def get_all_volume_signals(data: pd.DataFrame) -> Dict:
        """
        Generate all volume signals for a stock.
        
        Returns:
            Dict with all volume indicators and signals
        """
        try:
            # OBV
            obv = VolumeIndicators.obv(data)
            obv_sig = VolumeIndicators.obv_signal(data, obv)
            
            # VWAP
            vwap = VolumeIndicators.vwap(data)
            vwap_sig = VolumeIndicators.vwap_signal(data, vwap)
            
            # Volume momentum
            vol_mom = VolumeIndicators.volume_momentum(data)
            
            # MFI
            mfi = VolumeIndicators.mfi(data)
            mfi_value = mfi.iloc[-1] if len(mfi) > 0 and not pd.isna(mfi.iloc[-1]) else 50
            
            return {
                'obv': obv_sig,
                'vwap': vwap_sig,
                'volume_momentum': vol_mom,
                'mfi': {
                    'value': mfi_value,
                    'signal': 'overbought' if mfi_value > 80 else ('oversold' if mfi_value < 20 else 'neutral')
                }
            }
        
        except Exception as e:
            return {
                'obv': {'signal': 'neutral', 'strength': 0},
                'vwap': {'signal': 'neutral', 'strength': 0},
                'volume_momentum': {'ratio': 1.0, 'trend': 'unknown', 'unusual': False},
                'mfi': {'value': 50, 'signal': 'neutral'},
                'error': str(e)
            }
