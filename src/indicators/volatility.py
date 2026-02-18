"""
Volatility Indicators - ATR, Bollinger Bands, Keltner Channels

Essential for risk management and position sizing.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


class VolatilityIndicators:
    """
    Volatility-based technical indicators.
    
    Used for:
    - Position sizing (ATR-based)
    - Breakout detection (Bollinger Bands)
    - Trend strength (Keltner Channels)
    - Risk management
    """
    
    @staticmethod
    def atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Average True Range (ATR)
        
        Measures market volatility by decomposing the entire range
        of an asset price for that period.
        
        Args:
            data: DataFrame with 'High', 'Low', 'Close' columns
            period: Lookback period (default 14)
        
        Returns:
            ATR values
        """
        high = data['High'].squeeze() if 'High' in data.columns else data['Close'].squeeze()
        low = data['Low'].squeeze() if 'Low' in data.columns else data['Close'].squeeze()
        close = data['Close'].squeeze()
        
        # True Range = max(H-L, |H-PrevC|, |L-PrevC|)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR is exponential moving average of TR
        atr = true_range.ewm(alpha=1/period, min_periods=period).mean()
        
        return atr
    
    @staticmethod
    def atr_signal(data: pd.DataFrame, atr: pd.Series = None) -> Dict:
        """
        Generate volatility regime signal from ATR.
        
        Returns:
            Dict with volatility regime and ATR-based metrics
        """
        if atr is None:
            atr = VolatilityIndicators.atr(data)
        
        if len(atr) < 20 or pd.isna(atr.iloc[-1]):
            return {'regime': 'unknown', 'atr': 0, 'atr_pct': 0}
        
        close = data['Close'].squeeze()
        current_atr = atr.iloc[-1]
        current_price = close.iloc[-1]
        
        # ATR as percentage of price
        atr_pct = (current_atr / current_price) * 100
        
        # Compare to historical ATR
        avg_atr = atr.rolling(50).mean().iloc[-1]
        atr_ratio = current_atr / avg_atr if avg_atr > 0 else 1
        
        # Regime classification
        if atr_ratio > 1.5:
            regime = 'extreme_high'
            interpretation = 'Extreme volatility - reduce positions'
        elif atr_ratio > 1.2:
            regime = 'high'
            interpretation = 'High volatility - cautious trading'
        elif atr_ratio < 0.7:
            regime = 'low'
            interpretation = 'Low volatility - potential breakout coming'
        elif atr_ratio < 0.85:
            regime = 'below_normal'
            interpretation = 'Below normal volatility'
        else:
            regime = 'normal'
            interpretation = 'Normal volatility conditions'
        
        return {
            'regime': regime,
            'atr': current_atr,
            'atr_pct': atr_pct,
            'atr_ratio': atr_ratio,
            'interpretation': interpretation
        }
    
    @staticmethod
    def bollinger_bands(data: pd.DataFrame, period: int = 20, 
                        std_dev: float = 2.0) -> Dict:
        """
        Bollinger Bands
        
        Volatility bands placed above and below a moving average.
        - Price near upper band: Overbought / strong uptrend
        - Price near lower band: Oversold / strong downtrend
        - Band squeeze: Low volatility, potential breakout
        
        Args:
            data: DataFrame with 'Close' column
            period: Moving average period (default 20)
            std_dev: Standard deviation multiplier (default 2.0)
        
        Returns:
            Dict with upper, middle, lower bands and bandwidth
        """
        close = data['Close'].squeeze()
        
        middle = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        # Bandwidth (measure of volatility)
        bandwidth = (upper - lower) / middle * 100
        
        # %B (where price is relative to bands)
        percent_b = (close - lower) / (upper - lower) * 100
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'bandwidth': bandwidth,
            'percent_b': percent_b
        }
    
    @staticmethod
    def bollinger_signal(bb_data: Dict, data: pd.DataFrame) -> Dict:
        """
        Generate signal from Bollinger Bands.
        
        Returns:
            Dict with signal, strength, and interpretation
        """
        percent_b = bb_data['percent_b']
        bandwidth = bb_data['bandwidth']
        
        if len(percent_b) == 0 or pd.isna(percent_b.iloc[-1]):
            return {'signal': 'neutral', 'strength': 0}
        
        current_pb = percent_b.iloc[-1]
        current_bw = bandwidth.iloc[-1]
        prev_bw = bandwidth.iloc[-2] if len(bandwidth) > 1 else current_bw
        
        # Squeeze detection (low bandwidth)
        avg_bw = bandwidth.rolling(50).mean().iloc[-1] if len(bandwidth) >= 50 else current_bw
        
        if current_bw < avg_bw * 0.5:
            signal = 'squeeze'
            strength = 0.6
            interpretation = f'Band squeeze - potential breakout imminent (BW: {current_bw:.1f}%)'
        
        # Position relative to bands
        elif current_pb > 100:
            signal = 'above_upper'
            strength = min(1.0, (current_pb - 100) / 20)
            interpretation = f'Above upper band - strong momentum or overbought (%B: {current_pb:.1f})'
        
        elif current_pb < 0:
            signal = 'below_lower'
            strength = min(1.0, abs(current_pb) / 20)
            interpretation = f'Below lower band - strong decline or oversold (%B: {current_pb:.1f})'
        
        elif current_pb > 80:
            signal = 'near_upper'
            strength = (current_pb - 80) / 20
            interpretation = f'Near upper band (%B: {current_pb:.1f})'
        
        elif current_pb < 20:
            signal = 'near_lower'
            strength = (20 - current_pb) / 20
            interpretation = f'Near lower band (%B: {current_pb:.1f})'
        
        else:
            signal = 'neutral'
            strength = 0
            interpretation = f'Within bands (%B: {current_pb:.1f})'
        
        # Bandwidth expansion/contraction
        bw_change = (current_bw - prev_bw) / prev_bw if prev_bw > 0 else 0
        if bw_change > 0.2:
            interpretation += ' | Bands expanding'
        elif bw_change < -0.2:
            interpretation += ' | Bands contracting'
        
        return {
            'signal': signal,
            'strength': min(1.0, strength),
            'percent_b': current_pb,
            'bandwidth': current_bw,
            'interpretation': interpretation
        }
    
    @staticmethod
    def keltner_channels(data: pd.DataFrame, period: int = 20, 
                         atr_mult: float = 2.0) -> Dict:
        """
        Keltner Channels
        
        Similar to Bollinger Bands but uses ATR instead of std dev.
        More stable in trending markets.
        
        Args:
            data: DataFrame with 'High', 'Low', 'Close' columns
            period: EMA period (default 20)
            atr_mult: ATR multiplier (default 2.0)
        
        Returns:
            Dict with upper, middle, lower channels
        """
        high = data['High'].squeeze() if 'High' in data.columns else data['Close'].squeeze()
        low = data['Low'].squeeze() if 'Low' in data.columns else data['Close'].squeeze()
        close = data['Close'].squeeze()
        
        # Middle line is EMA of typical price
        typical_price = (high + low + close) / 3
        middle = typical_price.ewm(span=period).mean()
        
        # ATR for channel width
        atr = VolatilityIndicators.atr(data)
        
        upper = middle + (atr * atr_mult)
        lower = middle - (atr * atr_mult)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'width': (upper - lower) / middle * 100
        }
    
    @staticmethod
    def keltner_signal(kc_data: Dict, data: pd.DataFrame) -> Dict:
        """
        Generate signal from Keltner Channels.
        
        Returns:
            Dict with signal, strength, and interpretation
        """
        close = data['Close'].squeeze()
        upper = kc_data['upper']
        lower = kc_data['lower']
        middle = kc_data['middle']
        
        if len(close) == 0 or pd.isna(close.iloc[-1]):
            return {'signal': 'neutral', 'strength': 0}
        
        current_price = close.iloc[-1]
        current_upper = upper.iloc[-1]
        current_lower = lower.iloc[-1]
        current_middle = middle.iloc[-1]
        
        # Channel position
        if current_price > current_upper:
            signal = 'above_channel'
            strength = min(1.0, (current_price - current_upper) / (current_upper - current_middle))
            interpretation = 'Above channel - strong uptrend'
        
        elif current_price < current_lower:
            signal = 'below_channel'
            strength = min(1.0, (current_lower - current_price) / (current_middle - current_lower))
            interpretation = 'Below channel - strong downtrend'
        
        else:
            # Within channel
            channel_pos = (current_price - current_lower) / (current_upper - current_lower)
            signal = 'within_channel'
            strength = 0
            interpretation = f'Within channel ({channel_pos:.0%} from lower)'
        
        return {
            'signal': signal,
            'strength': strength,
            'channel_position': channel_pos if signal == 'within_channel' else None,
            'interpretation': interpretation
        }
    
    @staticmethod
    def get_all_volatility_signals(data: pd.DataFrame) -> Dict:
        """
        Generate all volatility signals for a stock.
        
        Returns:
            Dict with all volatility indicators and signals
        """
        try:
            # ATR
            atr = VolatilityIndicators.atr(data)
            atr_sig = VolatilityIndicators.atr_signal(data, atr)
            
            # Bollinger Bands
            bb_data = VolatilityIndicators.bollinger_bands(data)
            bb_sig = VolatilityIndicators.bollinger_signal(bb_data, data)
            
            # Keltner Channels
            kc_data = VolatilityIndicators.keltner_channels(data)
            kc_sig = VolatilityIndicators.keltner_signal(kc_data, data)
            
            return {
                'atr': atr_sig,
                'bollinger': bb_sig,
                'keltner': kc_sig,
                'current_atr': atr.iloc[-1] if len(atr) > 0 else 0
            }
        
        except Exception as e:
            return {
                'atr': {'regime': 'unknown'},
                'bollinger': {'signal': 'neutral', 'strength': 0},
                'keltner': {'signal': 'neutral', 'strength': 0},
                'error': str(e)
            }
