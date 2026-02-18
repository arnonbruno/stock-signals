"""
Historical Sentiment for Backtesting

Since newsdata.io doesn't provide historical news, we need alternatives:
1. Price-based sentiment proxy (enhanced)
2. Web scraping from financial archives
3. Sentiment from volatility and returns
4. Caching for backtest efficiency

This module provides multiple approaches for obtaining historical sentiment.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class HistoricalSentimentEngine:
    """
    Generate historical sentiment scores for backtesting.
    
    Approaches (in order of preference):
    1. Cached news data (if available)
    2. Enhanced price-based proxy
    3. Volatility-adjusted sentiment
    4. Return-based sentiment
    
    All methods are designed to work with only data available at that time
    (no look-ahead bias).
    """
    
    def __init__(self):
        self.cache = {}  # Ticker -> {date: sentiment}
        
        # Sentiment persistence factor
        # News sentiment tends to persist for a few days
        self.decay_factor = 0.85
        self.decay_days = 3
    
    def calculate_enhanced_price_sentiment(self, data: pd.DataFrame, 
                                            lookback: int = 10) -> pd.Series:
        """
        Enhanced price-based sentiment proxy.
        
        Uses multiple price-derived factors to estimate what news sentiment
        would have been at each point in time.
        
        Factors:
        1. Price momentum (returns)
        2. Volume changes
        3. Volatility regime
        4. Price vs moving averages
        5. Trend strength (ADX-like)
        
        Args:
            data: DataFrame with OHLCV data
            lookback: Days to look back for calculations
        
        Returns:
            Series of sentiment scores (-1 to +1)
        """
        close = data['Close'].squeeze()
        volume = data['Volume'].squeeze() if 'Volume' in data.columns else None
        high = data['High'].squeeze() if 'High' in data.columns else close
        low = data['Low'].squeeze() if 'Low' in data.columns else close
        
        sentiment = pd.Series(index=data.index, dtype=float)
        
        for i in range(max(lookback, 20), len(data)):
            # 1. Price Momentum Sentiment (40% weight)
            returns_5d = (close.iloc[i] / close.iloc[i-5] - 1)
            returns_10d = (close.iloc[i] / close.iloc[i-10] - 1)
            
            # Normalize returns to sentiment
            momentum_sent = np.clip((returns_5d * 3 + returns_10d * 2) / 0.1, -1, 1)
            
            # 2. Volume Sentiment (20% weight)
            if volume is not None:
                vol_recent = volume.iloc[i-5:i].mean()
                vol_avg = volume.iloc[i-lookback:i].mean()
                vol_ratio = vol_recent / vol_avg if vol_avg > 0 else 1
                
                # High volume on up days = bullish
                # High volume on down days = bearish
                vol_direction = 1 if returns_5d > 0 else -1
                volume_sent = np.clip((vol_ratio - 1) * vol_direction * 2, -1, 1)
            else:
                volume_sent = 0
            
            # 3. Volatility Sentiment (15% weight)
            returns = close.iloc[i-lookback:i].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            
            # Low volatility = complacency (bullish in bull market)
            # High volatility = fear/uncertainty (bearish)
            vol_regime = 'low' if volatility < 0.15 else ('high' if volatility > 0.35 else 'normal')
            
            # Infer market regime from longer-term trend
            ma50 = close.iloc[i-50:i].mean() if i >= 50 else close.iloc[i-lookback:i].mean()
            current_price = close.iloc[i]
            market_trend = 1 if current_price > ma50 else -1
            
            if vol_regime == 'low':
                vol_sent = 0.3 * market_trend  # Complacency
            elif vol_regime == 'high':
                vol_sent = -0.5  # Fear
            else:
                vol_sent = 0
            
            # 4. Price vs MA Sentiment (15% weight)
            ma20 = close.iloc[i-20:i].mean()
            ma50 = close.iloc[i-50:i].mean() if i >= 50 else ma20
            
            price_vs_ma20 = (current_price / ma20 - 1)
            price_vs_ma50 = (current_price / ma50 - 1) if i >= 50 else 0
            
            ma_sent = np.clip((price_vs_ma20 * 2 + price_vs_ma50) * 5, -1, 1)
            
            # 5. Trend Strength (10% weight)
            if i >= 14:
                # Simple ADX-like calculation
                tr = []
                for j in range(i-13, i+1):
                    tr.append(max(
                        high.iloc[j] - low.iloc[j],
                        abs(high.iloc[j] - close.iloc[j-1]),
                        abs(low.iloc[j] - close.iloc[j-1])
                    ))
                avg_tr = np.mean(tr)
                
                # Trend direction
                up_moves = [high.iloc[j] - high.iloc[j-1] for j in range(i-13, i+1)]
                down_moves = [low.iloc[j-1] - low.iloc[j] for j in range(i-13, i+1)]
                
                plus_dm = sum(max(0, u) for u in up_moves)
                minus_dm = sum(max(0, d) for d in down_moves)
                
                if plus_dm > minus_dm:
                    trend_sent = min(1, plus_dm / (plus_dm + minus_dm + 1e-10))
                else:
                    trend_sent = -min(1, minus_dm / (plus_dm + minus_dm + 1e-10))
            else:
                trend_sent = 0
            
            # Combine with weights
            combined_sentiment = (
                momentum_sent * 0.40 +
                volume_sent * 0.20 +
                vol_sent * 0.15 +
                ma_sent * 0.15 +
                trend_sent * 0.10
            )
            
            # Add small random noise for realism
            noise = np.random.normal(0, 0.05)
            
            sentiment.iloc[i] = np.clip(combined_sentiment + noise, -1, 1)
        
        # Forward fill early values
        sentiment.iloc[:max(lookback, 20)] = 0
        
        return sentiment
    
    def calculate_volatility_sentiment(self, data: pd.DataFrame,
                                        lookback: int = 20) -> pd.Series:
        """
        Volatility-based sentiment proxy.
        
        VIX-like approach: High volatility = fear, low volatility = complacency.
        
        Args:
            data: DataFrame with OHLCV data
            lookback: Lookback period
        
        Returns:
            Series of sentiment scores
        """
        close = data['Close'].squeeze()
        high = data['High'].squeeze() if 'High' in data.columns else close
        low = data['Low'].squeeze() if 'Low' in data.columns else close
        
        # Calculate True Range
        tr = pd.Series(index=data.index, dtype=float)
        for i in range(1, len(data)):
            tr.iloc[i] = max(
                high.iloc[i] - low.iloc[i],
                abs(high.iloc[i] - close.iloc[i-1]),
                abs(low.iloc[i] - close.iloc[i-1])
            )
        
        # ATR
        atr = tr.rolling(lookback).mean()
        
        # VIX-like measure (ATR as % of price)
        vix_like = (atr / close) * 100
        
        # Invert: high VIX = bearish sentiment
        # VIX < 15: Complacency (mildly bullish)
        # VIX 15-25: Normal
        # VIX > 25: Fear (bearish)
        # VIX > 35: Panic (very bearish, but potential contrarian buy)
        
        sentiment = pd.Series(index=data.index, dtype=float)
        
        for i in range(lookback, len(data)):
            vix_val = vix_like.iloc[i]
            
            if vix_val < 1:  # Very low volatility
                sent = 0.3  # Mildly bullish
            elif vix_val < 2:  # Low volatility
                sent = 0.2
            elif vix_val < 3:  # Normal
                sent = 0
            elif vix_val < 4:  # Elevated
                sent = -0.2
            elif vix_val < 5:  # High
                sent = -0.4
            else:  # Very high / panic
                # Contrarian: extreme fear can be bullish
                sent = -0.3 if vix_val < 6 else 0.1
            
            sentiment.iloc[i] = sent
        
        sentiment.iloc[:lookback] = 0
        
        return sentiment
    
    def calculate_return_sentiment(self, data: pd.DataFrame,
                                    lookback: int = 5) -> pd.Series:
        """
        Return-based sentiment proxy.
        
        Simple approach: positive returns = bullish sentiment.
        
        Args:
            data: DataFrame with 'Close' column
            lookback: Return calculation period
        
        Returns:
            Series of sentiment scores
        """
        close = data['Close'].squeeze()
        returns = close.pct_change(lookback)
        
        # Scale returns to sentiment
        # 5% return in lookback period = full bullish
        sentiment = np.clip(returns / 0.05, -1, 1)
        
        return sentiment.fillna(0)
    
    def get_historical_sentiment(self, data: pd.DataFrame, 
                                  method: str = 'enhanced') -> pd.Series:
        """
        Get historical sentiment using specified method.
        
        Args:
            data: DataFrame with OHLCV data
            method: 'enhanced', 'volatility', 'returns', or 'ensemble'
        
        Returns:
            Series of sentiment scores (-1 to +1)
        """
        if method == 'enhanced':
            return self.calculate_enhanced_price_sentiment(data)
        elif method == 'volatility':
            return self.calculate_volatility_sentiment(data)
        elif method == 'returns':
            return self.calculate_return_sentiment(data)
        elif method == 'ensemble':
            # Combine all methods
            enhanced = self.calculate_enhanced_price_sentiment(data)
            volatility = self.calculate_volatility_sentiment(data)
            returns = self.calculate_return_sentiment(data)
            
            # Weighted average
            return (
                enhanced * 0.50 +
                volatility * 0.30 +
                returns * 0.20
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def apply_sentiment_decay(self, sentiment: pd.Series) -> pd.Series:
        """
        Apply time decay to sentiment.
        
        News impact fades over time. This applies exponential decay
        to simulate realistic sentiment persistence.
        
        Args:
            sentiment: Raw sentiment scores
        
        Returns:
            Decayed sentiment scores
        """
        decayed = sentiment.copy()
        
        for i in range(self.decay_days, len(sentiment)):
            # If current sentiment is near zero, carry forward previous
            if abs(sentiment.iloc[i]) < 0.1:
                prev_sentiment = decayed.iloc[i-1]
                decayed.iloc[i] = prev_sentiment * self.decay_factor
        
        return decayed
    
    def get_sentiment_for_date(self, data: pd.DataFrame, 
                                date: datetime,
                                method: str = 'enhanced') -> float:
        """
        Get sentiment for a specific date (for backtesting).
        
        This is the main entry point for backtesting - it returns
        the sentiment that would have been available at that date.
        
        Args:
            data: Full price history (will be sliced to date)
            date: Target date
            method: Sentiment calculation method
        
        Returns:
            Sentiment score (-1 to +1)
        """
        # Slice data to only include data up to this date
        if isinstance(data.index, pd.DatetimeIndex):
            data_slice = data.loc[:date]
        else:
            data_slice = data.iloc[:data.index.get_loc(date) + 1]
        
        if len(data_slice) < 20:
            return 0.0
        
        # Calculate sentiment
        sentiment = self.get_historical_sentiment(data_slice, method)
        
        # Return most recent value
        return sentiment.iloc[-1]


class SentimentBacktestAdapter:
    """
    Adapter for using sentiment in backtests.
    
    Provides a unified interface that works with or without news data.
    """
    
    def __init__(self, method: str = 'ensemble'):
        """
        Initialize adapter.
        
        Args:
            method: Default sentiment method ('enhanced', 'volatility', 'returns', 'ensemble')
        """
        self.engine = HistoricalSentimentEngine()
        self.method = method
        self.precomputed = {}  # Cache for precomputed sentiments
    
    def precompute(self, ticker: str, data: pd.DataFrame):
        """
        Precompute sentiment for a ticker (speeds up backtests).
        
        Args:
            ticker: Stock ticker
            data: Full price history
        """
        self.precomputed[ticker] = self.engine.get_historical_sentiment(data, self.method)
        logger.info(f"Precomputed sentiment for {ticker}: {len(data)} days")
    
    def get_sentiment(self, ticker: str, date: datetime, 
                      data: pd.DataFrame = None) -> float:
        """
        Get sentiment for a ticker on a specific date.
        
        Args:
            ticker: Stock ticker
            date: Target date
            data: Price data (if not precomputed)
        
        Returns:
            Sentiment score (-1 to +1)
        """
        # Check precomputed cache
        if ticker in self.precomputed:
            series = self.precomputed[ticker]
            if date in series.index:
                return series.loc[date]
            # Find nearest date
            try:
                idx = series.index.get_indexer([date], method='nearest')[0]
                return series.iloc[idx]
            except:
                return 0.0
        
        # Compute on the fly
        if data is not None:
            return self.engine.get_sentiment_for_date(data, date, self.method)
        
        return 0.0
    
    def get_sentiment_series(self, ticker: str) -> pd.Series:
        """
        Get full sentiment series for a ticker.
        
        Returns:
            Series of sentiment scores
        """
        return self.precomputed.get(ticker, pd.Series())


# Convenience function
def get_backtest_sentiment(data: pd.DataFrame, method: str = 'ensemble') -> pd.Series:
    """
    Quick function to get historical sentiment for backtesting.
    
    Args:
        data: Price data DataFrame
        method: Sentiment method
    
    Returns:
        Series of sentiment scores
    """
    engine = HistoricalSentimentEngine()
    return engine.get_historical_sentiment(data, method)
