"""
Trend Indicators - Moving Averages, ADX, Parabolic SAR

Trend strength and direction analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


class TrendIndicators:
    """
    Trend-based technical indicators.
    
    Used for:
    - Trend direction identification
    - Trend strength measurement
    - Support/resistance levels
    """
    
    @staticmethod
    def moving_averages(data: pd.DataFrame, periods: list = [20, 50, 200]) -> Dict:
        """
        Simple and Exponential Moving Averages.
        
        Args:
            data: DataFrame with 'Close' column
            periods: List of MA periods
        
        Returns:
            Dict with SMA and EMA values
        """
        close = data['Close'].squeeze()
        
        result = {'sma': {}, 'ema': {}}
        
        for period in periods:
            result['sma'][period] = close.rolling(window=period).mean()
            result['ema'][period] = close.ewm(span=period, min_periods=period).mean()
        
        return result
    
    @staticmethod
    def ma_signal(ma_data: Dict, data: pd.DataFrame) -> Dict:
        """
        Generate signal from moving average analysis.
        
        Returns:
            Dict with signal, strength, and interpretation
        """
        close = data['Close'].squeeze()
        current_price = close.iloc[-1]
        
        signals = []
        
        # Price vs MAs
        for period, sma in ma_data['sma'].items():
            if len(sma) > 0 and not pd.isna(sma.iloc[-1]):
                ma_value = sma.iloc[-1]
                if current_price > ma_value:
                    signals.append(('bullish', period))
                else:
                    signals.append(('bearish', period))
        
        # MA crossovers
        if 20 in ma_data['sma'] and 50 in ma_data['sma']:
            sma20 = ma_data['sma'][20]
            sma50 = ma_data['sma'][50]
            
            if len(sma20) > 1 and len(sma50) > 1:
                # Golden Cross (20 > 50)
                if sma20.iloc[-1] > sma50.iloc[-1] and sma20.iloc[-2] <= sma50.iloc[-2]:
                    signals.append(('golden_cross', 50))
                # Death Cross (20 < 50)
                elif sma20.iloc[-1] < sma50.iloc[-1] and sma20.iloc[-2] >= sma50.iloc[-2]:
                    signals.append(('death_cross', 50))
        
        # Count bullish vs bearish
        bullish = sum(1 for s in signals if s[0] in ['bullish', 'golden_cross'])
        bearish = sum(1 for s in signals if s[0] in ['bearish', 'death_cross'])
        
        if bullish > bearish:
            signal = 'bullish'
            strength = bullish / len(signals) if signals else 0
            interpretation = f'Bullish MA alignment ({bullish}/{len(signals)})'
        elif bearish > bullish:
            signal = 'bearish'
            strength = bearish / len(signals) if signals else 0
            interpretation = f'Bearish MA alignment ({bearish}/{len(signals)})'
        else:
            signal = 'neutral'
            strength = 0
            interpretation = 'Mixed MA signals'
        
        # Add crossover info
        cross_types = [s[0] for s in signals if 'cross' in s[0]]
        if cross_types:
            interpretation += f' | {cross_types[0].replace("_", " ").title()}'
        
        return {
            'signal': signal,
            'strength': strength,
            'bullish_count': bullish,
            'bearish_count': bearish,
            'interpretation': interpretation
        }
    
    @staticmethod
    def adx(data: pd.DataFrame, period: int = 14) -> Dict:
        """
        Average Directional Index (ADX)
        
        Measures trend strength (not direction).
        ADX > 25: Strong trend
        ADX < 20: Weak/no trend
        
        Args:
            data: DataFrame with 'High', 'Low', 'Close' columns
            period: Lookback period
        
        Returns:
            Dict with ADX, +DI, -DI values
        """
        high = data['High'].squeeze() if 'High' in data.columns else data['Close'].squeeze()
        low = data['Low'].squeeze() if 'Low' in data.columns else data['Close'].squeeze()
        close = data['Close'].squeeze()
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        # Smoothed values
        atr = tr.ewm(alpha=1/period, min_periods=period).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
        
        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/period, min_periods=period).mean()
        
        return {
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        }
    
    @staticmethod
    def adx_signal(adx_data: Dict) -> Dict:
        """
        Generate signal from ADX data.
        
        Returns:
            Dict with signal, strength, and interpretation
        """
        adx = adx_data['adx']
        plus_di = adx_data['plus_di']
        minus_di = adx_data['minus_di']
        
        if len(adx) == 0 or pd.isna(adx.iloc[-1]):
            return {'signal': 'neutral', 'strength': 0, 'trend_strength': 0}
        
        current_adx = adx.iloc[-1]
        current_plus_di = plus_di.iloc[-1]
        current_minus_di = minus_di.iloc[-1]
        
        # Trend strength
        if current_adx >= 50:
            strength_label = 'very_strong'
            strength = 1.0
        elif current_adx >= 25:
            strength_label = 'strong'
            strength = current_adx / 50
        elif current_adx >= 20:
            strength_label = 'developing'
            strength = current_adx / 50
        else:
            strength_label = 'weak'
            strength = 0.3
        
        # Trend direction
        if current_plus_di > current_minus_di:
            direction = 'bullish'
            interpretation = f'Bullish trend ({strength_label}, ADX: {current_adx:.1f})'
        elif current_minus_di > current_plus_di:
            direction = 'bearish'
            interpretation = f'Bearish trend ({strength_label}, ADX: {current_adx:.1f})'
        else:
            direction = 'neutral'
            interpretation = f'No clear direction (ADX: {current_adx:.1f})'
        
        return {
            'signal': direction,
            'strength': strength,
            'trend_strength': current_adx,
            'strength_label': strength_label,
            'plus_di': current_plus_di,
            'minus_di': current_minus_di,
            'interpretation': interpretation
        }
    
    @staticmethod
    def supertrend(data: pd.DataFrame, period: int = 10, 
                   multiplier: float = 3.0) -> Dict:
        """
        SuperTrend Indicator
        
        Trend-following indicator based on ATR.
        Uses ATR to determine dynamic support/resistance levels.
        
        Args:
            data: DataFrame with 'High', 'Low', 'Close' columns
            period: ATR period
            multiplier: ATR multiplier
        
        Returns:
            Dict with SuperTrend values and direction
        """
        high = data['High'].squeeze() if 'High' in data.columns else data['Close'].squeeze()
        low = data['Low'].squeeze() if 'Low' in data.columns else data['Close'].squeeze()
        close = data['Close'].squeeze()
        
        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, min_periods=period).mean()
        
        # Basic bands
        hl2 = (high + low) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        # SuperTrend calculation
        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=int)
        
        supertrend.iloc[0] = upper_band.iloc[0]
        direction.iloc[0] = 1
        
        for i in range(1, len(close)):
            if close.iloc[i] > supertrend.iloc[i-1]:
                supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i-1])
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i-1])
                direction.iloc[i] = -1
        
        return {
            'supertrend': supertrend,
            'direction': direction,
            'upper': upper_band,
            'lower': lower_band
        }
    
    @staticmethod
    def supertrend_signal(st_data: Dict, data: pd.DataFrame) -> Dict:
        """
        Generate signal from SuperTrend.
        
        Returns:
            Dict with signal, strength, and interpretation
        """
        close = data['Close'].squeeze()
        supertrend = st_data['supertrend']
        direction = st_data['direction']
        
        if len(supertrend) == 0 or pd.isna(supertrend.iloc[-1]):
            return {'signal': 'neutral', 'strength': 0}
        
        current_st = supertrend.iloc[-1]
        current_price = close.iloc[-1]
        current_dir = direction.iloc[-1]
        prev_dir = direction.iloc[-2] if len(direction) > 1 else current_dir
        
        # Trend change detection
        if current_dir == 1 and prev_dir == -1:
            signal = 'buy'
            strength = 0.8
            interpretation = 'SuperTrend bullish reversal'
        elif current_dir == -1 and prev_dir == 1:
            signal = 'sell'
            strength = 0.8
            interpretation = 'SuperTrend bearish reversal'
        elif current_dir == 1:
            signal = 'bullish'
            strength = min(1.0, (current_price - current_st) / current_st * 10)
            interpretation = f'SuperTrend bullish (support: {current_st:.2f})'
        else:
            signal = 'bearish'
            strength = min(1.0, (current_st - current_price) / current_price * 10)
            interpretation = f'SuperTrend bearish (resistance: {current_st:.2f})'
        
        return {
            'signal': signal,
            'strength': strength,
            'supertrend_value': current_st,
            'direction': current_dir,
            'interpretation': interpretation
        }
    
    @staticmethod
    def get_all_trend_signals(data: pd.DataFrame) -> Dict:
        """
        Generate all trend signals for a stock.
        
        Returns:
            Dict with all trend indicators and signals
        """
        try:
            # Moving Averages
            ma_data = TrendIndicators.moving_averages(data)
            ma_sig = TrendIndicators.ma_signal(ma_data, data)
            
            # ADX
            adx_data = TrendIndicators.adx(data)
            adx_sig = TrendIndicators.adx_signal(adx_data)
            
            # SuperTrend
            st_data = TrendIndicators.supertrend(data)
            st_sig = TrendIndicators.supertrend_signal(st_data, data)
            
            return {
                'moving_averages': ma_sig,
                'adx': adx_sig,
                'supertrend': st_sig
            }
        
        except Exception as e:
            return {
                'moving_averages': {'signal': 'neutral', 'strength': 0},
                'adx': {'signal': 'neutral', 'strength': 0, 'trend_strength': 0},
                'supertrend': {'signal': 'neutral', 'strength': 0},
                'error': str(e)
            }
