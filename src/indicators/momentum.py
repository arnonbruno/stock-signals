"""
Momentum Indicators - RSI, MACD, Stochastic, ROC, Williams %R

SOTA momentum analysis for trend strength and reversal detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


class MomentumIndicators:
    """
    Collection of momentum-based technical indicators.
    
    Used for:
    - Trend strength confirmation
    - Overbought/oversold detection
    - Divergence analysis
    - Signal confidence boosting
    """
    
    @staticmethod
    def rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Relative Strength Index (RSI)
        
        Measures speed and magnitude of price movements.
        RSI > 70: Overbought (potential reversal down)
        RSI < 30: Oversold (potential reversal up)
        
        Args:
            data: DataFrame with 'Close' column
            period: Lookback period (default 14)
        
        Returns:
            RSI values (0-100)
        """
        close = data['Close'].squeeze()
        
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        
        # Use exponential moving average for smoother results
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def rsi_signal(rsi: pd.Series) -> Dict:
        """
        Generate signal from RSI values.
        
        Returns:
            Dict with signal, strength, and interpretation
        """
        if len(rsi) == 0 or pd.isna(rsi.iloc[-1]):
            return {'signal': 'neutral', 'strength': 0, 'interpretation': 'No data'}
        
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2] if len(rsi) > 1 else current_rsi
        
        # Determine signal
        if current_rsi > 70:
            signal = 'overbought'
            strength = (current_rsi - 70) / 30  # 0-1 scale
            interpretation = f'Overbought ({current_rsi:.1f}) - potential reversal'
        elif current_rsi < 30:
            signal = 'oversold'
            strength = (30 - current_rsi) / 30  # 0-1 scale
            interpretation = f'Oversold ({current_rsi:.1f}) - potential bounce'
        else:
            signal = 'neutral'
            strength = 0
            interpretation = f'Neutral ({current_rsi:.1f})'
        
        # Check for divergence
        if len(rsi) >= 14:
            rsi_trend = rsi.iloc[-14:].values
            if current_rsi < prev_rsi and signal == 'overbought':
                interpretation += ' | RSI declining from overbought'
            elif current_rsi > prev_rsi and signal == 'oversold':
                interpretation += ' | RSI rising from oversold'
        
        return {
            'signal': signal,
            'strength': min(1.0, strength),
            'rsi_value': current_rsi,
            'interpretation': interpretation
        }
    
    @staticmethod
    def macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, 
             signal: int = 9) -> Dict:
        """
        Moving Average Convergence Divergence (MACD)
        
        Trend-following momentum indicator.
        - MACD > Signal: Bullish
        - MACD < Signal: Bearish
        - Histogram growing: Momentum increasing
        
        Args:
            data: DataFrame with 'Close' column
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line period (default 9)
        
        Returns:
            Dict with MACD line, signal line, histogram
        """
        close = data['Close'].squeeze()
        
        ema_fast = close.ewm(span=fast, min_periods=fast).mean()
        ema_slow = close.ewm(span=slow, min_periods=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def macd_signal(macd_data: Dict) -> Dict:
        """
        Generate signal from MACD data.
        
        Returns:
            Dict with signal, strength, and interpretation
        """
        macd = macd_data['macd']
        signal_line = macd_data['signal']
        histogram = macd_data['histogram']
        
        if len(macd) == 0 or pd.isna(macd.iloc[-1]):
            return {'signal': 'neutral', 'strength': 0}
        
        current_macd = macd.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_hist = histogram.iloc[-1]
        prev_hist = histogram.iloc[-2] if len(histogram) > 1 else current_hist
        
        # Crossover detection
        if current_macd > current_signal:
            if prev_hist <= 0 and current_hist > 0:
                signal = 'bullish_crossover'
                strength = 0.8
                interpretation = 'Bullish MACD crossover - buy signal'
            else:
                signal = 'bullish'
                strength = min(1.0, abs(current_hist) / abs(macd.std()) * 2)
                interpretation = f'Bullish MACD ({current_hist:.4f})'
        else:
            if prev_hist >= 0 and current_hist < 0:
                signal = 'bearish_crossover'
                strength = 0.8
                interpretation = 'Bearish MACD crossover - sell signal'
            else:
                signal = 'bearish'
                strength = min(1.0, abs(current_hist) / abs(macd.std()) * 2)
                interpretation = f'Bearish MACD ({current_hist:.4f})'
        
        # Check for zero line crossover
        if current_macd > 0 and macd.iloc[-2] <= 0:
            interpretation += ' | Zero line crossover (bullish)'
            strength = min(1.0, strength + 0.2)
        elif current_macd < 0 and macd.iloc[-2] >= 0:
            interpretation += ' | Zero line crossover (bearish)'
            strength = min(1.0, strength + 0.2)
        
        return {
            'signal': signal,
            'strength': strength,
            'macd_value': current_macd,
            'histogram': current_hist,
            'interpretation': interpretation
        }
    
    @staticmethod
    def stochastic(data: pd.DataFrame, k_period: int = 14, 
                   d_period: int = 3) -> Dict:
        """
        Stochastic Oscillator
        
        Compares closing price to price range over period.
        %K > 80: Overbought
        %K < 20: Oversold
        
        Args:
            data: DataFrame with 'Close', 'High', 'Low' columns
            k_period: %K period (default 14)
            d_period: %D smoothing period (default 3)
        
        Returns:
            Dict with %K and %D values
        """
        close = data['Close'].squeeze()
        high = data['High'].squeeze() if 'High' in data.columns else close
        low = data['Low'].squeeze() if 'Low' in data.columns else close
        
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period).mean()
        
        return {
            'k': k,
            'd': d
        }
    
    @staticmethod
    def stochastic_signal(stoch_data: Dict) -> Dict:
        """
        Generate signal from Stochastic data.
        
        Returns:
            Dict with signal, strength, and interpretation
        """
        k = stoch_data['k']
        d = stoch_data['d']
        
        if len(k) == 0 or pd.isna(k.iloc[-1]):
            return {'signal': 'neutral', 'strength': 0}
        
        current_k = k.iloc[-1]
        current_d = d.iloc[-1]
        prev_k = k.iloc[-2] if len(k) > 1 else current_k
        prev_d = d.iloc[-2] if len(d) > 1 else current_d
        
        # Determine signal
        if current_k > 80:
            signal = 'overbought'
            strength = (current_k - 80) / 20
            interpretation = f'Overbought (%K: {current_k:.1f})'
        elif current_k < 20:
            signal = 'oversold'
            strength = (20 - current_k) / 20
            interpretation = f'Oversold (%K: {current_k:.1f})'
        else:
            signal = 'neutral'
            strength = 0
            interpretation = f'Neutral (%K: {current_k:.1f})'
        
        # Crossover detection
        if prev_k <= prev_d and current_k > current_d:
            interpretation += ' | Bullish %K/%D crossover'
            if signal == 'oversold':
                strength = min(1.0, strength + 0.3)
        elif prev_k >= prev_d and current_k < current_d:
            interpretation += ' | Bearish %K/%D crossover'
            if signal == 'overbought':
                strength = min(1.0, strength + 0.3)
        
        return {
            'signal': signal,
            'strength': min(1.0, strength),
            'k_value': current_k,
            'd_value': current_d,
            'interpretation': interpretation
        }
    
    @staticmethod
    def williams_r(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Williams %R
        
        Similar to Stochastic but inverted scale.
        %R > -20: Overbought
        %R < -80: Oversold
        
        Args:
            data: DataFrame with 'Close', 'High', 'Low' columns
            period: Lookback period
        
        Returns:
            Williams %R values (-100 to 0)
        """
        close = data['Close'].squeeze()
        high = data['High'].squeeze() if 'High' in data.columns else close
        low = data['Low'].squeeze() if 'Low' in data.columns else close
        
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        
        return wr
    
    @staticmethod
    def rate_of_change(data: pd.DataFrame, period: int = 10) -> pd.Series:
        """
        Rate of Change (ROC)
        
        Measures percentage price change over period.
        ROC > 0: Price increasing
        ROC < 0: Price decreasing
        
        Args:
            data: DataFrame with 'Close' column
            period: Lookback period
        
        Returns:
            ROC values (percentage)
        """
        close = data['Close'].squeeze()
        roc = ((close - close.shift(period)) / close.shift(period)) * 100
        
        return roc
    
    @staticmethod
    def get_all_momentum_signals(data: pd.DataFrame) -> Dict:
        """
        Generate all momentum signals for a stock.
        
        Returns:
            Dict with all momentum indicators and signals
        """
        try:
            # RSI
            rsi = MomentumIndicators.rsi(data)
            rsi_sig = MomentumIndicators.rsi_signal(rsi)
            
            # MACD
            macd_data = MomentumIndicators.macd(data)
            macd_sig = MomentumIndicators.macd_signal(macd_data)
            
            # Stochastic
            stoch_data = MomentumIndicators.stochastic(data)
            stoch_sig = MomentumIndicators.stochastic_signal(stoch_data)
            
            # Williams %R
            wr = MomentumIndicators.williams_r(data)
            wr_value = wr.iloc[-1] if len(wr) > 0 and not pd.isna(wr.iloc[-1]) else -50
            
            # ROC
            roc = MomentumIndicators.rate_of_change(data)
            roc_value = roc.iloc[-1] if len(roc) > 0 and not pd.isna(roc.iloc[-1]) else 0
            
            return {
                'rsi': rsi_sig,
                'macd': macd_sig,
                'stochastic': stoch_sig,
                'williams_r': {
                    'value': wr_value,
                    'signal': 'overbought' if wr_value > -20 else ('oversold' if wr_value < -80 else 'neutral')
                },
                'roc': {
                    'value': roc_value,
                    'signal': 'bullish' if roc_value > 0 else 'bearish'
                }
            }
        
        except Exception as e:
            return {
                'rsi': {'signal': 'neutral', 'strength': 0},
                'macd': {'signal': 'neutral', 'strength': 0},
                'stochastic': {'signal': 'neutral', 'strength': 0},
                'williams_r': {'value': -50, 'signal': 'neutral'},
                'roc': {'value': 0, 'signal': 'neutral'},
                'error': str(e)
            }
