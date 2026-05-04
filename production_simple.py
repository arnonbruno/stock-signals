#!/usr/bin/env python3
"""
Production runner: technicals + optional news + fundamentals.

``SimpleProductionRunner`` is the engine behind ``scripts/quick_market_monitor.py``
(top-50 liquid list, fundamentals on, news typically off) and can also analyze
the full universe from ``data/validated_tickers.json`` when invoked from this module.

Price data: **yfinance** for historical OHLCV; **BrAPI** (``src.brapi_client``) for
spot quotes with ``price_cache.json`` caching.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from src.signals.trend_detector_v2 import TrendDetectorV2
from src.news.free_news_client import FreeNewsClient
from src.features.feature_engineering import get_feature_engineer
from src.fundamentals.integration import FundamentalIntegrator
from src.indicators.signal_fusion import fuse_all_signals
from src.brapi_client import BrAPIClient
from src.alerts.alert_generator import generate_trading_alerts
from src.strategy.regime_detection import get_regime_detector, get_adaptive_params


def load_tickers() -> List[str]:
    """Load validated tickers from JSON file."""
    config_path = Path(__file__).parent / 'data' / 'validated_tickers.json'
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            data = json.load(f)
        tickers = data.get('all_tickers', [])
        # Add .SA suffix for Yahoo Finance
        return [f"{t}.SA" if not t.endswith('.SA') else t for t in tickers]
    else:
        # Fallback to hardcoded list
        print("⚠️ validated_tickers.json not found, using fallback list")
        return IBOV_FALLBACK + SMLL_FALLBACK


# Fallback tickers (used only if JSON file missing)
IBOV_FALLBACK = [
    'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'BBAS3.SA', 'ABEV3.SA',
    'B3SA3.SA', 'SUZB3.SA', 'RENT3.SA', 'WEGE3.SA'
]

SMLL_FALLBACK = [
    'CURY3.SA', 'EZTC3.SA', 'JHSF3.SA', 'CYRE3.SA', 'MRVE3.SA'
]

from multiprocessing import Pool, cpu_count
import functools


class SimpleProductionRunner:
    """
    End-to-end pipeline: indicators + TrendDetectorV2 + optional FinBERT news +
    FundamentalIntegrator (Graham / Lynch / Greenblatt), then signals and sizing.
    """
    
    # Class-level model cache (shared across instances)
    _model_cache = {}
    MIN_SKIP_TURNOVER_BRL = 50_000
    MIN_DOWNGRADE_TURNOVER_BRL = 500_000
    
    def __init__(self, use_news: bool = True, use_fundamentals: bool = True, n_workers: int = None):
        self.trend_detector = TrendDetectorV2()
        self.use_news = use_news
        self.use_fundamentals = use_fundamentals
        # Cap workers at 8 to prevent resource exhaustion
        max_workers = min(cpu_count(), 8)
        self.n_workers = min(n_workers, max_workers) if n_workers else max_workers
        
        # BrAPI client for real-time prices (replaces yfinance stale prices)
        self.brapi_client = BrAPIClient()
        self._brapi_prices: Dict[str, float] = {}
        self._price_snapshots: Dict[str, Dict] = {}
        
        if use_news:
            # Check if model already cached
            if 'news_client' not in self._model_cache:
                self._model_cache['news_client'] = FreeNewsClient()
            self.news_client = self._model_cache['news_client']
        
        if use_fundamentals:
            self.fundamental_integrator = FundamentalIntegrator()

        # Market regime detection on IBOV (replaces simple vol→regime mapping)
        self.regime_detector = get_regime_detector()
        self.adaptive_params = get_adaptive_params()
        self.current_regime = 'sideways'
        self.regime_strength = 0.5
        self.regime_params = {}

        print(f"✅ Sistema inicializado (news={'ON' if use_news else 'OFF'}, "
              f"fundamentals={'ON' if use_fundamentals else 'OFF'}, "
              f"price_source=BrAPI->Yahoo fallback, workers={self.n_workers})")

    @staticmethod
    def _normalize_data(data: pd.DataFrame) -> pd.DataFrame:
        """Normalize yfinance/brapi frames to flat OHLCV columns."""
        if isinstance(data.columns, pd.MultiIndex):
            data = data.copy()
            data.columns = data.columns.get_level_values(0)
        return data

    def _fetch_market_data(self, as_of_date=None) -> pd.DataFrame:
        """Fetch IBOV index data for market regime detection."""
        try:
            if as_of_date:
                end = as_of_date.strftime('%Y-%m-%d') if hasattr(as_of_date, 'strftime') else str(as_of_date)
                ibov = yf.download('^BVSP', period='2y', progress=False, end=end)
            else:
                ibov = yf.download('^BVSP', period='2y', progress=False)
            if ibov is not None and len(ibov) > 0:
                if isinstance(ibov.columns, pd.MultiIndex):
                    ibov.columns = ibov.columns.get_level_values(0)
                return ibov
        except Exception as e:
            if not as_of_date:  # Only print warning on live runs
                print(f"\n⚠️ Could not fetch IBOV for regime detection: {e}")
        return None

    def detect_market_regime(self, as_of_date=None) -> Dict:
        """
        Detect overall market regime from IBOV index using the full
        MarketRegimeDetector (MA200, MA50 cross, 6mo momentum, R², vol ratio).

        Returns regime info with adaptive parameters from AdaptiveStrategyParameters.
        """
        ibov = self._fetch_market_data(as_of_date=as_of_date)
        if ibov is None or len(ibov) < 50:
            if not as_of_date:
                print("   [REGIME] No IBOV data, using fallback: sideways/neutral")
            return {
                'regime': 'sideways',
                'strength': 0.5,
                'params': {}
            }

        regime_info = self.regime_detector.detect_regime(ibov)
        params = self.adaptive_params.get_parameters(
            regime_info['regime'],
            regime_info['strength']
        )

        # Cache for live runs
        if not as_of_date:
            self.current_regime = regime_info['regime']
            self.regime_strength = regime_info['strength']
            self.regime_params = params

        return {
            'regime': regime_info['regime'],
            'strength': regime_info['strength'],
            'params': params,
            'details': regime_info.get('details', {}),
        }

    @staticmethod
    def _align_integrated_signal_with_trend(trend: str, recommendation: str, market_regime: str = "sideways") -> str:
        """
        Align fundamental recommendation with technical trend.

        In bull markets, trust the technical trend over fundamentals
        (technicals capture momentum that fundamentals lag).
        In bear/sideways, let fundamentals veto technical signals.
        """
        trend = (trend or "").upper()
        recommendation = (recommendation or "HOLD").upper()

        # Bull market: trust technicals — fundamentals lag behind rallies
        if market_regime == "bull":
            if trend == "UPTREND":
                # Only upgrade, never downgrade a strong technical BUY in bull
                if recommendation in {"BUY", "STRONG_BUY"}:
                    return recommendation
                return "BUY"  # Trust the uptrend over fundamental caution
            return recommendation

        # Bear / sideways: let fundamentals be the gatekeeper
        if trend == "UPTREND" and recommendation in {"SELL", "STRONG_SELL"}:
            return "HOLD"
        if trend == "DOWNTREND" and recommendation in {"BUY", "STRONG_BUY"}:
            return "HOLD"
        return recommendation

    def _calculate_liquidity_profile(self, data: pd.DataFrame) -> Dict:
        """
        Calculate 20-day liquidity using average traded value (price * volume).

        This is more robust than raw share volume for Brazilian equities where
        penny stocks can print high share count but still be illiquid in BRL.
        """
        data = self._normalize_data(data)

        if 'Close' not in data.columns or 'Volume' not in data.columns or len(data) < 20:
            return {
                'avg_volume_20d': 0.0,
                'avg_turnover_20d': 0.0,
                'status': 'skip',
            }

        closes = data['Close'].tail(20).astype(float)
        volumes = data['Volume'].tail(20).astype(float)
        avg_volume_20d = float(volumes.mean()) if not volumes.empty else 0.0
        avg_turnover_20d = float((closes * volumes).mean()) if not closes.empty else 0.0

        if np.isnan(avg_turnover_20d):
            avg_turnover_20d = 0.0
        if np.isnan(avg_volume_20d):
            avg_volume_20d = 0.0

        status = 'healthy'
        if avg_turnover_20d < self.MIN_SKIP_TURNOVER_BRL:
            status = 'skip'
        elif avg_turnover_20d < self.MIN_DOWNGRADE_TURNOVER_BRL:
            status = 'downgrade'

        return {
            'avg_volume_20d': avg_volume_20d,
            'avg_turnover_20d': avg_turnover_20d,
            'status': status,
        }

    @staticmethod
    def _downgrade_signal(signal: str) -> str:
        """Downgrade a signal by one level toward neutral."""
        downgrade_map = {
            'STRONG_BUY': 'BUY',
            'BUY': 'HOLD',
            'SELL': 'HOLD',
            'STRONG_SELL': 'SELL',
        }
        return downgrade_map.get(signal, signal)

    def _apply_liquidity_adjustment(
        self,
        ticker: str,
        signal: str,
        conviction: float,
        position_size: float,
        liquidity_profile: Dict,
    ) -> Tuple[Optional[str], float, float]:
        """Apply low-liquidity skip/downgrade policy to the final signal."""
        avg_turnover = liquidity_profile.get('avg_turnover_20d', 0.0)
        status = liquidity_profile.get('status', 'healthy')

        if status == 'skip':
            print(
                f"     [LIQ] Skipping {ticker}: "
                f"20d avg turnover R${avg_turnover:,.0f} < R${self.MIN_SKIP_TURNOVER_BRL:,.0f}"
            )
            return None, 0.0, 0.0

        if status == 'downgrade' and signal not in ['HOLD', 'NONE']:
            downgraded_signal = self._downgrade_signal(signal)
            if downgraded_signal != signal:
                print(
                    f"     [LIQ] Downgrading {ticker}: "
                    f"20d avg turnover R${avg_turnover:,.0f} < R${self.MIN_DOWNGRADE_TURNOVER_BRL:,.0f} "
                    f"({signal} -> {downgraded_signal})"
                )
                signal = downgraded_signal
                if signal == 'HOLD':
                    conviction = 0.0
                    position_size = 0.0
                else:
                    conviction = np.sign(conviction) * min(abs(conviction), 0.65)
                    position_size *= 0.75

        return signal, conviction, position_size

    @staticmethod
    def _extract_key_indicators(fused: Dict) -> List[Dict]:
        """Extract the strongest indicator signals from the fusion output."""
        category_details = fused.get('category_details', {})
        top_signals = []

        for category, details in category_details.items():
            for sig_name, sig_signal, sig_strength in details.get('signals', []):
                top_signals.append({
                    'name': sig_name,
                    'category': category,
                    'signal': sig_signal,
                    'strength': sig_strength,
                })

        top_signals.sort(key=lambda x: abs(x['strength']), reverse=True)
        return top_signals[:5]

    @staticmethod
    def _latest_close_from_data(data: pd.DataFrame) -> float:
        price_val = data['Close'].iloc[-1]
        return float(price_val.item()) if hasattr(price_val, 'item') else float(price_val)

    @staticmethod
    def _format_index_timestamp(index_value) -> Optional[str]:
        if hasattr(index_value, 'isoformat'):
            return index_value.isoformat()
        return str(index_value) if index_value is not None else None

    def _prefetch_spot_prices(self, tickers: List[str]) -> None:
        clean_tickers = [t.replace('.SA', '') for t in tickers]

        if self.brapi_client.is_quota_exceeded():
            print(f"💰 BrAPI quota exhausted - fetching Yahoo Finance spot fallback ({len(clean_tickers)} tickers)...")
        else:
            print(f"💰 Fetching real-time spot prices ({len(clean_tickers)} tickers)...")

        self._brapi_prices = self.brapi_client.get_batch_prices(clean_tickers)
        self._price_snapshots = {}

        # When quota exceeded, accept stale quotes (previous close)
        allow_stale = self.brapi_client.is_quota_exceeded()
        for ticker in clean_tickers:
            snapshot = self.brapi_client.get_price_snapshot(ticker, allow_stale=allow_stale)
            if snapshot is not None:
                self._price_snapshots[ticker] = snapshot

        brapi_count = sum(1 for snapshot in self._price_snapshots.values() if snapshot.get('source') == 'brapi')
        yahoo_count = sum(
            1 for snapshot in self._price_snapshots.values()
            if str(snapshot.get('source', '')).startswith('yfinance')
        )
        stale_count = sum(1 for snapshot in self._price_snapshots.values() if snapshot.get('is_stale', False))
        missing_count = len(clean_tickers) - len(self._price_snapshots)

        if self.brapi_client.is_quota_exceeded():
            print("   ⚠️ BrAPI quota exhausted; Yahoo Finance fallback active")

        print(
            f"   ✅ Spot prices: {len(self._price_snapshots)}/{len(clean_tickers)} "
            f"(BrAPI={brapi_count}, Yahoo={yahoo_count}, stale={stale_count}, missing={missing_count})\n"
        )

    def _resolve_current_price(self, ticker: str, data: pd.DataFrame) -> Tuple[Optional[float], str, Optional[str], bool]:
        """
        Resolve current price from BrAPI or fallback sources.
        
        Returns:
            Tuple of (price, source, quote_ts, is_stale)
            is_stale is True when using previous day's close during market hours
        """
        clean_ticker = ticker.replace('.SA', '')
        
        # First check if we have a stale cached price from prefetch
        if clean_ticker in self._price_snapshots:
            snapshot = self._price_snapshots[clean_ticker]
            is_stale = snapshot.get('is_stale', False)
            return (
                float(snapshot['price']),
                str(snapshot.get('source', 'unknown')),
                snapshot.get('quote_ts') or snapshot.get('updated'),
                is_stale,
            )
        
        # Otherwise fetch fresh
        snapshot = self.brapi_client.get_price_snapshot(clean_ticker)

        if snapshot is None:
            live_price = self.brapi_client.get_spot_price(clean_ticker)
            if live_price is not None:
                snapshot = self.brapi_client.get_price_snapshot(clean_ticker)
                self._brapi_prices[clean_ticker] = live_price
                if snapshot is not None:
                    self._price_snapshots[clean_ticker] = snapshot

        if snapshot is not None:
            is_stale = snapshot.get('is_stale', False)
            return (
                float(snapshot['price']),
                str(snapshot.get('source', 'unknown')),
                snapshot.get('quote_ts') or snapshot.get('updated'),
                is_stale,
            )

        if self.brapi_client.is_market_hours():
            print(f"     [PRICE] No spot quote for {ticker}; skipping during market hours")
            return None, 'missing', None, False

        latest_close = self._latest_close_from_data(data)
        close_ts = self._format_index_timestamp(data.index[-1] if len(data.index) else None)
        print(f"     [PRICE] Using latest daily close for {ticker} outside market hours")
        return latest_close, 'yfinance_close', close_ts, False

    def _build_trade_levels(
        self,
        current_price: float,
        technical_indicators: Dict,
        market_regime: str,
    ) -> Optional[Dict]:
        """Build entry, stop, and target levels for long setups."""
        from src.config import get_config

        config = get_config()
        if current_price <= 0:
            return None

        stop_pct = config.get_stop_loss(market_regime)
        base_stop = current_price * (1 - stop_pct)
        ma50 = technical_indicators.get('ma50')

        if ma50 and ma50 > 0 and ma50 < current_price:
            stop_loss = max(base_stop, ma50 * 0.995)
        else:
            stop_loss = base_stop

        risk = current_price - stop_loss
        if risk <= 0:
            return None

        return {
            'entry': current_price,
            'stop_loss': stop_loss,
            'target_1': current_price + (risk * 2),
            'target_2': current_price + (risk * 3),
        }
    
    def calculate_kelly_position(self, data: pd.DataFrame, confidence: float) -> float:
        """
        Calculate position size using TRUE Kelly Criterion with confidence adjustment.
        
        Kelly Formula: f* = (bp - q) / b
        where: b = average win / average loss (payoff ratio/odds)
               p = win probability (win rate)
               q = 1 - p (loss probability)
        
        Uses Half-Kelly for safety: position = 0.5 * kelly * confidence
        
        Args:
            data: Price data DataFrame with 'Close' column
            confidence: Signal confidence from trend detection (0-1)
        
        Returns:
            Position size as fraction of portfolio (0.10 to 0.60)
        """
        from src.config import get_config
        config = get_config()
        
        try:
            # Calculate returns from close prices
            close_prices = data['Close'].squeeze()
            returns = close_prices.pct_change().dropna()
            
            # Need minimum data points for statistical significance
            if len(returns) < config.KELLY_MIN_DATA_POINTS:
                return config.DEFAULT_POSITION_SIZE
            
            # Separate positive and negative returns
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            
            # Need both winning and losing trades to calculate Kelly
            if len(positive_returns) == 0 or len(negative_returns) == 0:
                return config.DEFAULT_POSITION_SIZE
            
            # Calculate Kelly parameters
            p = len(positive_returns) / len(returns)  # Win probability
            q = 1 - p  # Loss probability
            
            avg_win = positive_returns.mean()  # Average winning return
            avg_loss = abs(negative_returns.mean())  # Average losing return (absolute)
            
            # Payoff ratio (odds) - how much we win vs how much we lose
            if avg_loss == 0:
                return config.DEFAULT_POSITION_SIZE
            
            b = avg_win / avg_loss  # Payoff ratio
            
            # Kelly formula: f* = (bp - q) / b
            # This gives the optimal fraction of capital to risk
            kelly = (b * p - q) / b
            
            # Kelly can be negative if edge is negative (don't trade)
            # or > 1 if edge is very high (cap at reasonable levels)
            if kelly <= 0:
                return config.MIN_POSITION_SIZE
            
            # Apply Half-Kelly for safety (reduces volatility and drawdowns)
            # Also scale by confidence from signal quality and realized volatility.
            realized_vol = float(returns.std() * np.sqrt(252))
            if np.isnan(realized_vol) or realized_vol <= 0:
                vol_adjustment = 1.0
            else:
                target_vol = 0.35
                vol_adjustment = float(np.clip(target_vol / realized_vol, 0.5, 1.25))

            position = kelly * config.KELLY_FRACTION * confidence * vol_adjustment
            
            # Enforce bounds
            return max(config.MIN_POSITION_SIZE, min(config.MAX_POSITION_SIZE, position))
        
        except Exception as e:
            return config.DEFAULT_POSITION_SIZE
    
    def get_data(self, ticker: str, days: int = 120, max_retries: int = 3, 
                 end_date: datetime = None) -> pd.DataFrame:
        """
        Download recent data with error handling and retries.
        
        Implements exponential backoff for network failures.
        Returns None if all retries fail.
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days of historical data
            max_retries: Maximum retry attempts
            end_date: Optional end date for backtesting (prevents look-ahead bias)
        """
        end_date = end_date or datetime.now()
        start_date = end_date - timedelta(days=days)
        
        for attempt in range(max_retries):
            try:
                data = yf.download(
                    ticker,
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    progress=False
                )
                
                if data.empty:
                    print(f"    ⚠️ No data returned for {ticker}")
                    return None
                
                return data
            
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 1.0  # Exponential backoff: 1s, 2s, 4s
                    print(f"    ⚠️ Attempt {attempt + 1}/{max_retries} failed for {ticker}: {e}")
                    print(f"       Retrying in {wait_time}s...")
                    import time
                    time.sleep(wait_time)
                else:
                    print(f"    ❌ All retries exhausted for {ticker}: {e}")
                    return None
        
        return None
    
    def get_news_sentiment(self, ticker: str) -> dict:
        """Get sentiment for today only (newsdata.io + cache)"""
        if not self.use_news:
            return {"sentiment": 0.0, "articles": [], "dates": []}
        
        try:
            # Fetch sentiment for TODAY ONLY using FreeNewsClient
            # (which handles newsdata.io + Investing.com fallback + caching)
            date = datetime.now().strftime("%Y-%m-%d")
            sentiment = self.news_client.get_sentiment(ticker, date)
            cached_payload = self.news_client.cache.get_full(ticker) or {}
            cached_articles = cached_payload.get("articles", [])
            articles = [
                {
                    "title": article.get("title", ""),
                    "summary": article.get("summary", ""),
                    "source": article.get("source_publication", article.get("source", "Unknown")),
                    "link": article.get("link", ""),
                }
                for article in cached_articles
                if article.get("title")
            ]
            
            return {
                "sentiment": sentiment,
                "articles": articles,
                "dates": [date],
                "daily_scores": {date: sentiment}
            }
        except Exception as e:
            print(f"    ⚠️ News error: {e}")
            return {"sentiment": 0.0, "articles": [], "dates": []}
    
    def analyze_ticker(self, ticker: str, data: pd.DataFrame = None, 
                       as_of_date: datetime = None) -> Dict:
        """Analyze single ticker
        
        Args:
            ticker: Stock ticker symbol
            data: Optional pre-loaded DataFrame (for backtesting). If None, downloads fresh data.
            as_of_date: Optional date for backtesting (prevents look-ahead bias by using only historical data)
        """
        try:
            # Download data if not provided (production mode)
            if data is None:
                data = self.get_data(ticker, end_date=as_of_date)
            
            if data is None:
                return None

            data = self._normalize_data(data)

            if len(data) < 50:
                return None
            
            current_price, price_source, price_timestamp, is_price_stale = self._resolve_current_price(ticker, data)
            if current_price is None:
                return None

            liquidity_profile = self._calculate_liquidity_profile(data)
            if liquidity_profile['status'] == 'skip':
                self._apply_liquidity_adjustment(ticker, 'BUY', 0.0, 0.0, liquidity_profile)
                return None
            
            # Trend detection (core validated logic)
            trend_result = self.trend_detector.detect_trend(data)
            consensus = trend_result.get('consensus', 'unknown')
            confidence = trend_result.get('confidence', 0.0)
            
            # Feature engineering: volume, volatility, sector features
            feature_engineer = get_feature_engineer()
            features = feature_engineer.generate_all_features(ticker, data)
            
            # Adjust confidence based on feature signals
            volume_features = features.get('volume', {})
            volatility_features = features.get('volatility', {})
            
            # Volume confirmation boost
            if volume_features.get('unusual_volume', False):
                confidence *= 1.05  # 5% boost for unusual volume
                print(f"     [VOL] Unusual volume detected (+5% confidence)")
            
            # Volatility regime adjustment
            vol_regime = volatility_features.get('regime', 'medium')
            if vol_regime == 'high':
                confidence *= 0.90  # Reduce confidence in high volatility
                print(f"     [VOL] High volatility regime (-10% confidence)")

            # Market regime via IBOV index (full detector: MA200, MA50 cross, 6mo momentum, R², vol ratio)
            regime_info = self.detect_market_regime(as_of_date=as_of_date)
            market_regime = regime_info.get('regime', 'sideways')
            regime_params = regime_info.get('params', {})

            # Technical signal fusion adds a second opinion on the trend setup.
            fused = fuse_all_signals(
                data,
                regime=market_regime if market_regime in {'bull', 'bear', 'sideways'} else 'sideways',
                regime_strength=min(1.0, max(confidence, 0.25)),
            )
            fused_score = float(fused.get('score', 0.0))
            fusion_confidence = float(fused.get('confidence', 0.0))
            key_indicators = self._extract_key_indicators(fused)
            
            # Store features for reporting
            feature_summary = {
                'volume_momentum': volume_features.get('volume_momentum', 1.0),
                'unusual_volume': volume_features.get('unusual_volume', False),
                'volatility_regime': vol_regime,
                'vix_equivalent': volatility_features.get('vix_equivalent', 20.0),
                'avg_volume_20d': liquidity_profile.get('avg_volume_20d', 0.0),
                'avg_turnover_20d': liquidity_profile.get('avg_turnover_20d', 0.0),
                'liquidity_status': liquidity_profile.get('status', 'healthy'),
            }
            
            # Print trend details
            print(f"\n     [TREND] {consensus} @ {confidence:.1%} confidence")
            
            # Map consensus to simple trend (UPPERCASE to match integration.py expectations)
            if consensus in ['uptrend', 'bull_pullback']:
                trend = "UPTREND"
            elif consensus in ['downtrend', 'bear_bounce']:
                trend = "DOWNTREND"
            else:
                trend = "SIDEWAYS"

            trend_direction = 1 if trend == "UPTREND" else -1 if trend == "DOWNTREND" else 0
            fusion_alignment = fused_score * trend_direction if trend_direction else 0.0
            confidence = float(np.clip(confidence + (fusion_alignment * 0.10), 0.0, 1.0))
            
            # News sentiment (if enabled)
            if self.use_news:
                print(f"     [NEWS] newsdata.io...", end=" ", flush=True)
            
            news_data = self.get_news_sentiment(ticker)
            news_sentiment = news_data.get("sentiment", 0.0)
            news_articles = news_data.get("articles", [])
            
            if self.use_news and news_articles:
                print(f"✅ {len(news_articles)} articles, sentiment: {news_sentiment:+.2f}")
            elif self.use_news:
                print(f"⚠️  No articles found, sentiment: {news_sentiment:+.2f}")
            
            from src.config import get_config
            config = get_config()
            
            signal = "HOLD"
            position_size = 0.0
            conviction = 0.0
            technical_score = float(np.clip((confidence * 100) + (fusion_alignment * 15), 0.0, 100.0))
            
            # Regime-aware thresholds: use AdaptiveStrategyParameters from real MarketRegimeDetector
            # This analyzes IBOV via MA200, MA50 cross, 6mo momentum, R², and vol ratio
            # to set appropriate buy/sell thresholds, position sizing, and cash buffers.
            thresholds = config.get_thresholds(market_regime)

            if regime_params:
                MIN_CONFIDENCE = regime_params.get('confidence_threshold', 0.50)
                SELL_CONFIDENCE = regime_params.get('sell_threshold', 0.50)
            else:
                # Fallback if no IBOV data
                MIN_CONFIDENCE = 0.50
                SELL_CONFIDENCE = 0.50

            style = regime_info.get('details', {}).get('price_vs_ma200', 0)
            prefix = "🟢" if style > 0.05 else ("🔴" if style < -0.05 else "🟡")
            print(f"     {prefix} [REGIME] {market_regime.upper()} (str={regime_info.get('strength', 0.5):.0%}) -> buy_conf={MIN_CONFIDENCE:.2f}, sell_conf={SELL_CONFIDENCE:.2f}")
            
            if trend == "UPTREND" and confidence >= MIN_CONFIDENCE:
                signal = "BUY"
                conviction = confidence
                
                # Kelly Criterion position sizing (volatility-adjusted)
                position_size = self.calculate_kelly_position(data, confidence)
                
                # News boost: Sigmoid scaling (SOTA approach)
                # Uses logistic function to prevent over-amplification
                if self.use_news and abs(news_sentiment) > 0.1:
                    # Shifted sigmoid keeps positive news above 1.0 and negative below 1.0.
                    sigmoid_boost = 0.5 + (1 / (1 + np.exp(-5 * news_sentiment)))  # 0.5 to 1.5
                    position_size *= sigmoid_boost  # Multiplicative scaling
                    
                    # Update conviction based on news agreement with trend
                    if (news_sentiment > 0 and trend == "UPTREND") or (news_sentiment < 0 and trend == "DOWNTREND"):
                        conviction = min(1.0, conviction * 1.1)  # 10% boost when aligned
                
                # Cap at 80% for safety
                position_size = min(0.80, position_size)
                    
            elif trend == "DOWNTREND" and confidence >= SELL_CONFIDENCE:
                signal = "SELL"
                conviction = -confidence
                position_size = 1.0  # Exit completely
            
            # Integrate fundamental analysis
            fundamental_data = None
            if self.use_fundamentals:
                integrated_score = self.fundamental_integrator.integrate(
                    ticker=ticker,
                    technical_score=technical_score,
                    trend=trend,
                    confidence=confidence
                )
                
                # Align fundamentals with technical trend (regime-aware).
                signal = self._align_integrated_signal_with_trend(
                    trend,
                    integrated_score.recommendation,
                    market_regime=market_regime,
                )
                if signal in ["BUY", "STRONG_BUY"]:
                    conviction = integrated_score.composite_score / 100
                elif signal in ["SELL", "STRONG_SELL"]:
                    conviction = -(integrated_score.composite_score / 100)
                else:
                    conviction = 0.0
                
                # Adjust position size based on fundamentals
                if signal in ["BUY", "STRONG_BUY"]:
                    if position_size <= 0:
                        position_size = self.calculate_kelly_position(data, confidence)

                    # High quality stocks with real ROE/ROIC support can carry more size.
                    if (
                        integrated_score.roe is not None and integrated_score.roe > 20 and
                        integrated_score.roic is not None and integrated_score.roic > 15
                    ):
                        position_size = min(0.80, position_size * 1.2)
                    
                    # Deep value names keep moderate size because they can be slower catalysts.
                    if (
                        integrated_score.pe_ratio is not None and integrated_score.pe_ratio < 10 and
                        integrated_score.pb_ratio is not None and integrated_score.pb_ratio < 1
                    ):
                        position_size = max(0.15, min(0.50, position_size))
                    
                    # Poor fundamentals reduce position size
                    if integrated_score.fundamental_score < 40:
                        position_size *= 0.5
                    
                    # Avoid flag blocks trading entirely
                    if integrated_score.is_avoid:
                        signal = "HOLD"
                        position_size = 0.0
                elif signal in ["SELL", "STRONG_SELL"]:
                    position_size = 1.0
                else:
                    position_size = 0.0
                
                fundamental_data = {
                    'composite_score': integrated_score.composite_score,
                    'fundamental_grade': integrated_score.fundamental_grade,
                    'value_score': integrated_score.value_score,
                    'quality_score': integrated_score.quality_score,
                    'growth_score': integrated_score.growth_score,
                    'is_value_pick': integrated_score.is_value_pick,
                    'is_quality_pick': integrated_score.is_quality_pick,
                    'is_momentum_pick': integrated_score.is_momentum_pick,
                    'is_avoid': integrated_score.is_avoid,
                    'pe_ratio': integrated_score.pe_ratio,
                    'pb_ratio': integrated_score.pb_ratio,
                    'roe': integrated_score.roe,
                    'roic': integrated_score.roic,
                    'div_yield': integrated_score.div_yield,
                    'debt_equity': integrated_score.debt_equity,
                    'strengths': integrated_score.strengths,
                    'weaknesses': integrated_score.weaknesses,
                    'action_notes': integrated_score.action_notes,
                }
            
            # Calculate technical indicators for detailed reporting
            technical_indicators = {}
            
            # RSI (14-period)
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            if len(rsi) > 0:
                technical_indicators['rsi'] = rsi.iloc[-1]
            
            # MACD
            exp12 = data['Close'].ewm(span=12, adjust=False).mean()
            exp26 = data['Close'].ewm(span=26, adjust=False).mean()
            macd = exp12 - exp26
            macd_signal = macd.ewm(span=9, adjust=False).mean()
            if len(macd) > 0 and len(macd_signal) > 0:
                technical_indicators['macd'] = macd.iloc[-1]
                technical_indicators['macd_signal'] = macd_signal.iloc[-1]
            
            # Volume vs average (20-day)
            avg_volume = data['Volume'].rolling(window=20).mean()
            if len(avg_volume) > 0:
                current_volume = data['Volume'].iloc[-1]
                avg_vol = avg_volume.iloc[-1]
                if avg_vol > 0:
                    technical_indicators['volume_vs_avg'] = (current_volume / avg_vol) * 100
                technical_indicators['avg_volume_20d'] = float(avg_vol)
                technical_indicators['avg_turnover_20d'] = float((data['Close'].tail(20) * data['Volume'].tail(20)).mean())
            
            # Price vs moving averages
            ma50 = data['Close'].rolling(window=50).mean()
            ma20 = data['Close'].rolling(window=20).mean()
            if len(ma50) > 0 and len(ma20) > 0:
                current_close = data['Close'].iloc[-1]
                technical_indicators['ma50'] = float(ma50.iloc[-1])
                technical_indicators['ma20'] = float(ma20.iloc[-1])
                technical_indicators['price_vs_50d'] = ((current_close - ma50.iloc[-1]) / ma50.iloc[-1]) * 100
                technical_indicators['price_vs_20d'] = ((current_close - ma20.iloc[-1]) / ma20.iloc[-1]) * 100
            
            # Extract news headlines
            news_headlines = []
            if news_articles:
                for article in news_articles[:3]:
                    title = article.get('title', '')
                    source = article.get('source', article.get('source_publication', ''))
                    if title:
                        if source:
                            news_headlines.append(f"{title} ({source})")
                        else:
                            news_headlines.append(title)

            # KIPP IMPROVEMENT: trailing stop for active positions
            trailing_stop = None
            risk_levels = None
            if signal in ["BUY", "STRONG_BUY"]:
                risk_levels = self._build_trade_levels(current_price, technical_indicators, market_regime)
                # Trailing stop: 8% below current price for BUY signals
                trailing_stop = round(current_price * 0.92, 2)

            signal, conviction, position_size = self._apply_liquidity_adjustment(
                ticker, signal, conviction, position_size, liquidity_profile
            )
            if signal is None:
                return None
            
            return {
                "ticker": ticker,
                "price": current_price,
                "price_source": price_source,
                "price_timestamp": price_timestamp,
                "is_price_stale": is_price_stale,  # True when using previous close during market hours
                "trend": trend,
                "confidence": confidence,  # Add confidence from trend detector
                "fusion_confidence": fusion_confidence,
                "fused_score": fused_score,
                "key_indicators": key_indicators,
                "market_regime": market_regime,
                "news_sentiment": news_sentiment,
                "news_articles": news_articles,
                "news_headlines": news_headlines,  # Add headlines for reporting
                "signal": signal,
                "conviction": conviction,
                "position_size": position_size,
                "features": feature_summary,  # Add feature engineering data
                "fundamentals": fundamental_data,  # Add fundamental data
                "technical_indicators": technical_indicators,  # Add technical indicators
                "risk_levels": risk_levels,
                "trailing_stop": trailing_stop,
                "liquidity": liquidity_profile,
            }
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return None
    
    def get_top_movers(self, results: List[Dict], top_n: int = 10) -> List[str]:
        """
        Identify top movers from analysis results.
        
        Used for smart news fetching: fetch fresh news only for high-conviction signals.
        
        Args:
            results: List of analysis results
            top_n: Number of top movers to return
        
        Returns:
            List of top mover tickers
        """
        if not results:
            return []
        
        # Sort by absolute conviction (both BUY and SELL signals matter)
        sorted_results = sorted(results, key=lambda x: abs(x.get('conviction', 0)), reverse=True)
        
        # Get top N tickers
        top_movers = [r['ticker'] for r in sorted_results[:top_n]]
        
        print(f"\n📊 Top {top_n} Movers (for smart news refresh):")
        for i, ticker in enumerate(top_movers, 1):
            result = next(r for r in sorted_results if r['ticker'] == ticker)
            print(f"   [{i}] {ticker} - {result['signal']} ({result['trend']}, conviction: {result['conviction']:.2f})")
        
        return top_movers
    
    def _analyze_ticker_wrapper(self, ticker: str) -> Dict:
        """Wrapper for parallel processing (needed for Pool.map)."""
        print(f"📊 {ticker}...", end=" ", flush=True)
        result = self.analyze_ticker(ticker)
        if result:
            print(f"{result['signal']} ({result['trend']})")
        else:
            print("SKIP")
        return result
    
    def run(self, tickers: List[str] = None, parallel: bool = True):
        """
        Run analysis on all tickers.
        
        Args:
            tickers: List of tickers to analyze (default: all from validated_tickers.json)
            parallel: Use parallel processing (default: True, ~8x faster)
        """
        if tickers is None:
            tickers = load_tickers()
        
        print(f"\n{'='*70}")
        print(f"🚀 PRODUÇÃO - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"📊 Analyzing {len(tickers)} tickers (IBOV + SMLL)")
        print(f"{'='*70}\n")

        # Detect overall market regime from IBOV (runs once per scan)
        print(f"📈 Detecting market regime from IBOV...")
        regime_info = self.detect_market_regime()
        print(f"   🎯 Market regime: {regime_info.get('regime', 'sideways').upper()} "
              f"(strength: {regime_info.get('strength', 0.5):.0%})")
        if regime_info.get('params'):
            p = regime_info['params']
            print(f"   📐 Thresholds: buy>={p.get('confidence_threshold', 0.5):.0%}, "
                  f"sell>={p.get('sell_threshold', 0.5):.0%}")
            print(f"   💰 Cash buffer: {p.get('max_cash_pct', 0.3)*100:.0f}%")
        print()

        self._prefetch_spot_prices(tickers)
        
        results = []
        
        if parallel and len(tickers) > 1:
            # Parallel processing (8x faster for 150 tickers)
            print(f"⚡ Parallel mode: {self.n_workers} workers\n")
            
            with Pool(self.n_workers) as pool:
                raw_results = pool.map(self._analyze_ticker_wrapper, tickers)
            
            results = [r for r in raw_results if r is not None]
        else:
            # Sequential processing (for debugging or single ticker)
            for ticker in tickers:
                print(f"📊 {ticker}...", end=" ")
                result = self.analyze_ticker(ticker)
                if result:
                    print(f"{result['signal']} ({result['trend']})")
                    results.append(result)
                else:
                    print("SKIP")
        
        # Summary
        self._print_summary(results)
        
        # Zero-signal monitoring: alert if no BUY/SELL signals generated
        buy_signals = [r for r in results if r['signal'] in ('BUY', 'STRONG_BUY')]
        sell_signals = [r for r in results if r['signal'] in ('SELL', 'STRONG_SELL')]
        
        if results and not buy_signals and not sell_signals:
            print("\n" + "!" * 70)
            print("🚨 ZERO-SIGNAL ALERT: No BUY or SELL signals generated!")
            print(f"   Analyzed {len(results)} tickers, all returned HOLD.")
            print("   Possible causes:")
            print("   - Confidence thresholds too high for current regime")
            print("   - Trend/case mismatch between modules")
            print("   - Market in low-conviction sideways regime")
            print("!" * 70 + "\n")
        elif results and not buy_signals:
            print(f"\n⚠️  MONITORING: 0 BUY signals out of {len(results)} tickers. "
                  f"({len(sell_signals)} SELL)")
        
        # Generate formatted alerts for Telegram delivery
        if buy_signals or sell_signals:
            print("\n" + "=" * 70)
            print("📱 TELEGRAM ALERTS")
            print("=" * 70 + "\n")
            alert_message = generate_trading_alerts(results, top_n=5)
            print(alert_message)
        
        return results
    
    def _print_summary(self, results: List[Dict]):
        """Print final table with fundamental data"""
        if not results:
            print("\n❌ No results")
            return
        
        # Sort by conviction
        results_sorted = sorted(results, key=lambda x: abs(x['conviction']), reverse=True)
        
        print(f"\n{'='*70}")
        print(f"📊 RESUMO - {len(results)} tickers")
        print(f"{'='*70}\n")
        
        # Header
        if self.use_fundamentals:
            print(f"{'Ticker':<10} {'Preço':>8} {'Sinal':<8} {'Grade':>4} {'Val':>4} {'Qual':>4} {'Pos%':>5}")
        else:
            print(f"{'Ticker':<10} {'Preço':>8} {'Sinal':<6} {'Trend':<10} {'News':>6} {'Pos%':>5}")
        print(f"{'-'*70}")
        
        # Rows
        for r in results_sorted:
            signal = r['signal']
            signal_emoji = {
                "STRONG_BUY": "🚀",
                "BUY": "🟢",
                "HOLD": "🟡",
                "SELL": "🔴",
                "STRONG_SELL": "💀"
            }.get(signal, "⚪")
            
            pos_pct = f"{r['position_size']*100:.0f}%" if r['position_size'] > 0 else "-"
            
            # Fundamental indicators
            fund = r.get('fundamentals', {})
            if self.use_fundamentals and fund:
                grade = fund.get('fundamental_grade', 'N/A')
                value = f"{fund.get('value_score', 0):.0f}" if fund.get('value_score') else "-"
                quality = f"{fund.get('quality_score', 0):.0f}" if fund.get('quality_score') else "-"
                
                # Add fundamental indicators
                fund_indicators = []
                if fund.get('is_value_pick'):
                    fund_indicators.append('💰')
                if fund.get('is_quality_pick'):
                    fund_indicators.append('⭐')
                if fund.get('is_momentum_pick'):
                    fund_indicators.append('📈')
                if fund.get('is_avoid'):
                    fund_indicators.append('⚠️')
                fund_str = ''.join(fund_indicators)
                
                print(
                    f"{r['ticker']:<10} "
                    f"R${r['price']:>7.2f} "
                    f"{signal_emoji} {signal:<6} "
                    f"{grade:>4} "
                    f"{value:>4} "
                    f"{quality:>4} "
                    f"{pos_pct:>5} "
                    f"{fund_str}"
                )
            else:
                # Original format without fundamentals
                news_str = f"{r['news_sentiment']:+.2f}" if self.use_news else "N/A"
                
                features = r.get('features', {})
                feature_indicators = []
                if features.get('unusual_volume'):
                    feature_indicators.append('📈')
                if features.get('volatility_regime') == 'high':
                    feature_indicators.append('⚡')
                feature_str = ''.join(feature_indicators) if feature_indicators else ''
                
                print(
                    f"{r['ticker']:<10} "
                    f"R${r['price']:>7.2f} "
                    f"{signal_emoji} {signal:<4} "
                    f"{r['trend']:<10} "
                    f"{news_str:>6} "
                    f"{pos_pct:>5} "
                    f"{feature_str}"
                )
        
        # Stats
        buy = [r for r in results if r['signal'] in ("BUY", "STRONG_BUY")]
        sell = [r for r in results if r['signal'] in ("SELL", "STRONG_SELL")]
        
        print(f"\n{'-'*70}")
        print(f"🟢 BUY: {len(buy)} | 🔴 SELL: {len(sell)} | ⚪ HOLD: {len(results) - len(buy) - len(sell)}")
        print(f"{'='*70}\n")
        
        # News details (if enabled)
        if self.use_news:
            print(f"\n{'='*70}")
            print(f"📰 NEWS ANALYSIS DETAILS")
            print(f"{'='*70}\n")
            
            # Show fundamental details for top signals
            if self.use_fundamentals:
                print(f"\n{'='*70}")
                print(f"📊 FUNDAMENTAL ANALYSIS - TOP SIGNALS")
                print(f"{'='*70}\n")
                
                # Show top 5 BUY signals
                buy_signals = [r for r in results_sorted if r['signal'] in ['BUY', 'STRONG_BUY']][:5]
                if buy_signals:
                    print("🟢 TOP BUY SIGNALS:\n")
                    for r in buy_signals:
                        fund = r.get('fundamentals', {})
                        if fund:
                            print(f"  {r['ticker']} - {r['signal']} @ R${r['price']:.2f}")
                            print(f"    Grade: {fund.get('fundamental_grade')} | "
                                  f"Value: {fund.get('value_score', 0):.0f} | "
                                  f"Quality: {fund.get('quality_score', 0):.0f}")
                            if fund.get('pe_ratio'):
                                print(f"    P/E: {fund['pe_ratio']:.1f} | "
                                      f"P/B: {fund.get('pb_ratio', 0):.2f} | "
                                      f"ROE: {fund.get('roe', 0):.1f}% | "
                                      f"Div: {fund.get('div_yield', 0):.1f}%")
                            if fund.get('strengths'):
                                print(f"    ✅ {', '.join(fund['strengths'][:2])}")
                            if fund.get('action_notes'):
                                for note in fund['action_notes'][:2]:
                                    print(f"    {note}")
                            print()
                
                # Show AVOID stocks
                avoid_stocks = [r for r in results_sorted if r.get('fundamentals', {}).get('is_avoid')]
                if avoid_stocks:
                    print("⚠️ AVOID (Poor fundamentals + Poor technicals):\n")
                    for r in avoid_stocks[:5]:
                        fund = r.get('fundamentals', {})
                        print(f"  {r['ticker']} - {fund.get('fundamental_grade')} grade, "
                              f"Value: {fund.get('value_score', 0):.0f}, Quality: {fund.get('quality_score', 0):.0f}")
                        if fund.get('weaknesses'):
                            print(f"    ❌ {', '.join(fund['weaknesses'][:2])}")
                    print()
            
            # News details (if enabled)
            if self.use_news:
                print(f"\n{'='*70}")
                print(f"📰 NEWS ANALYSIS DETAILS")
                print(f"{'='*70}\n")
                
                for r in results_sorted[:10]:  # Top 10 only
                    articles = r.get('news_articles', [])
                    sentiment = r.get('news_sentiment', 0.0)
                    
                    if articles:
                        print(f"📊 {r['ticker']} - Sentiment: {sentiment:+.2f} ({len(articles)} articles)")
                        for i, article in enumerate(articles[:3], 1):  # Show top 3 articles
                            title = article.get('title', 'No title')[:70]
                            art_sentiment = article.get('sentiment', 0.0)
                            date = article.get('date', 'N/A')
                            print(f"   [{i}] ({art_sentiment:+.2f}) {title}...")
                            print(f"       Date: {date} | Source: {article.get('source', 'unknown')}")
                        if len(articles) > 3:
                            print(f"   ... and {len(articles) - 3} more articles")
                    else:
                        print(f"📊 {r['ticker']} - No news found (Sentiment: {sentiment:+.2f})")
                    print()


def main():
    parser = argparse.ArgumentParser(description="Production Runner with Fundamental Analysis")
    parser.add_argument("--ticker", type=str, help="Single ticker (ex: PETR4.SA)")
    parser.add_argument("--no-news", action="store_true", help="Disable news (faster)")
    parser.add_argument("--no-fundamentals", action="store_true", help="Disable fundamentals (technical only)")
    
    args = parser.parse_args()
    
    runner = SimpleProductionRunner(
        use_news=not args.no_news,
        use_fundamentals=not args.no_fundamentals
    )
    
    if args.ticker:
        runner.run(tickers=[args.ticker])
    else:
        runner.run()


if __name__ == "__main__":
    main()
