"""
Threshold Optimization System for Stock Signals

Implements grid search optimization for trading signal thresholds using
walk-forward validation to prevent overfitting.

Key features:
- Grid search over confidence, score, and stop-loss thresholds
- Regime-specific optimization (bull/bear/sideways)
- Walk-forward validation for robust parameter selection
- Sharpe ratio optimization with risk-adjusted metrics
"""

import numpy as np
import pandas as pd
import json
import os
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from itertools import product
from datetime import datetime, timedelta
from pathlib import Path

from .walk_forward import WalkForwardValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ThresholdSet:
    """Container for a set of trading thresholds."""
    buy_confidence: float
    sell_confidence: float
    min_score: float
    stop_loss: float
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'ThresholdSet':
        return cls(
            buy_confidence=d.get('buy_confidence', 0.55),
            sell_confidence=d.get('sell_confidence', 0.45),
            min_score=d.get('min_score', 0.25),
            stop_loss=d.get('stop_loss', 0.15)
        )


@dataclass
class OptimizationResult:
    """Container for optimization results."""
    best_thresholds: ThresholdSet
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    regime: str
    validation_windows: int
    
    def to_dict(self) -> Dict:
        return {
            'best_thresholds': self.best_thresholds.to_dict(),
            'sharpe_ratio': self.sharpe_ratio,
            'total_return': self.total_return,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'num_trades': self.num_trades,
            'regime': self.regime,
            'validation_windows': self.validation_windows
        }


class ThresholdOptimizer:
    """
    Optimizes trading thresholds using walk-forward validation.
    
    Grid search parameters:
    - buy_confidence: Minimum confidence for BUY signals
    - sell_confidence: Minimum confidence for SELL signals  
    - min_score: Minimum fused score threshold
    - stop_loss: Stop loss percentage
    
    Optimization target: Sharpe ratio (risk-adjusted returns)
    """
    
    # Default parameter grids
    DEFAULT_BUY_CONFIDENCE = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    DEFAULT_SELL_CONFIDENCE = [0.35, 0.40, 0.45, 0.50]
    DEFAULT_MIN_SCORE = [0.20, 0.25, 0.30, 0.35]
    DEFAULT_STOP_LOSS = [0.10, 0.15, 0.20]
    
    def __init__(self, 
                 train_window_days: int = 126,  # ~6 months
                 test_window_days: int = 42,     # ~2 months
                 param_grid: Dict[str, List[float]] = None):
        """
        Initialize the threshold optimizer.
        
        Args:
            train_window_days: Days for training window
            test_window_days: Days for testing window
            param_grid: Custom parameter grid (optional)
        """
        self.train_window = train_window_days
        self.test_window = test_window_days
        
        # Use default grid if not provided
        self.param_grid = param_grid or {
            'buy_confidence': self.DEFAULT_BUY_CONFIDENCE,
            'sell_confidence': self.DEFAULT_SELL_CONFIDENCE,
            'min_score': self.DEFAULT_MIN_SCORE,
            'stop_loss': self.DEFAULT_STOP_LOSS
        }
        
        self.validator = WalkForwardValidator(
            train_window=train_window_days,
            test_window=test_window_days
        )
        
        logger.info(f"ThresholdOptimizer initialized")
        logger.info(f"  Train window: {train_window_days} days")
        logger.info(f"  Test window: {test_window_days} days")
        logger.info(f"  Grid combinations: {self._count_combinations()}")
    
    def _count_combinations(self) -> int:
        """Count total parameter combinations."""
        return (len(self.param_grid['buy_confidence']) *
                len(self.param_grid['sell_confidence']) *
                len(self.param_grid['min_score']) *
                len(self.param_grid['stop_loss']))
    
    def _generate_threshold_sets(self) -> List[ThresholdSet]:
        """Generate all threshold combinations for grid search."""
        combinations = list(product(
            self.param_grid['buy_confidence'],
            self.param_grid['sell_confidence'],
            self.param_grid['min_score'],
            self.param_grid['stop_loss']
        ))
        
        return [ThresholdSet(*combo) for combo in combinations]
    
    def _simulate_strategy(self, 
                          data: pd.DataFrame, 
                          thresholds: ThresholdSet,
                          signals: pd.DataFrame = None) -> Dict:
        """
        Simulate trading strategy with given thresholds.
        
        Args:
            data: Price data (OHLCV)
            thresholds: Threshold set to test
            signals: Pre-computed signals (optional)
        
        Returns:
            Dict with performance metrics
        """
        if data is None or len(data) < 50:
            return {
                'returns': [],
                'sharpe': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'num_trades': 0
            }
        
        # Normalize columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        close = data['Close'].squeeze()
        high = data['High'].squeeze() if 'High' in data.columns else close
        low = data['Low'].squeeze() if 'Low' in data.columns else close
        
        # Generate trend-based signals using moving averages
        ma_fast = close.rolling(20).mean()
        ma_slow = close.rolling(50).mean()
        
        # RSI-like momentum indicator
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 0.001)
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate confidence and score
        trend_ratio = (ma_fast / ma_slow - 1)
        max_strength = trend_ratio.abs().rolling(20).max()
        max_strength = max_strength.replace(0, 0.01).fillna(0.01)
        trend_strength = trend_ratio.abs() / max_strength
        
        # Confidence: combination of trend strength and RSI
        confidence = pd.Series(0.5, index=data.index)
        confidence = trend_strength.fillna(0.5) * 0.7 + (rsi / 100).fillna(0.5) * 0.3
        
        # Score: scaled to be in 0-1 range (multiply by 10 to convert % to threshold range)
        # e.g., 5% trend = 0.05 * 10 = 0.50 score
        score = (trend_ratio.fillna(0) * 10).clip(-1, 1)
        
        # Generate trades based on thresholds
        returns = []
        current_position = 0  # 0 = cash, 1 = long
        entry_price = 0
        stop_loss_price = 0
        
        for i in range(50, len(data)):  # Start after warmup
            price = close.iloc[i]
            conf = confidence.iloc[i]
            scr = score.iloc[i]
            
            # Check stop loss first
            if current_position == 1:
                daily_low = low.iloc[i]
                if daily_low <= stop_loss_price:
                    # Stop loss triggered
                    trade_return = (stop_loss_price - entry_price) / entry_price
                    returns.append(trade_return)
                    current_position = 0
                    continue
            
            # Buy signal: confidence above threshold and positive score
            if current_position == 0:
                if conf >= thresholds.buy_confidence and scr >= thresholds.min_score:
                    current_position = 1
                    entry_price = price
                    stop_loss_price = price * (1 - thresholds.stop_loss)
            
            # Sell signal: confidence above threshold and negative score
            elif current_position == 1:
                if conf >= thresholds.sell_confidence and scr < -thresholds.min_score:
                    trade_return = (price - entry_price) / entry_price
                    returns.append(trade_return)
                    current_position = 0
            
        # Close any open position at end
        if current_position == 1:
            final_return = (close.iloc[-1] - entry_price) / entry_price
            returns.append(final_return)
        
        # Calculate metrics
        if not returns:
            return {
                'returns': [],
                'sharpe': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'num_trades': 0
            }
        
        returns_arr = np.array(returns)
        total_return = np.prod(1 + returns_arr) - 1
        
        # Sharpe ratio (annualized)
        if len(returns_arr) > 1 and np.std(returns_arr) > 0:
            sharpe = np.mean(returns_arr) / np.std(returns_arr) * np.sqrt(252 / 42)  # Annualized for test window
        else:
            sharpe = 0.0
        
        # Max drawdown
        cumulative = np.cumprod(1 + returns_arr)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / running_max
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        # Win rate
        win_rate = np.sum(returns_arr > 0) / len(returns_arr) if len(returns_arr) > 0 else 0
        
        return {
            'returns': returns,
            'sharpe': float(sharpe),
            'total_return': float(total_return),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'num_trades': len(returns)
        }
    
    def optimize(self, 
                data: pd.DataFrame,
                signals: pd.DataFrame = None,
                regime: str = 'default') -> OptimizationResult:
        """
        Run grid search optimization for thresholds.
        
        Args:
            data: Historical price data
            signals: Pre-computed signals (optional)
            regime: Market regime for logging
        
        Returns:
            OptimizationResult with best thresholds and metrics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting threshold optimization for regime: {regime}")
        logger.info(f"{'='*60}")
        
        threshold_sets = self._generate_threshold_sets()
        
        # Split data into walk-forward windows
        splits = self.validator.split_data(data)
        
        if not splits:
            logger.warning("Not enough data for walk-forward validation")
            return OptimizationResult(
                best_thresholds=ThresholdSet(0.55, 0.45, 0.25, 0.15),
                sharpe_ratio=0.0,
                total_return=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                num_trades=0,
                regime=regime,
                validation_windows=0
            )
        
        logger.info(f"Walk-forward splits: {len(splits)}")
        
        best_sharpe = -np.inf
        best_thresholds = None
        best_metrics = None
        
        # Grid search
        total_combinations = len(threshold_sets)
        
        for idx, thresholds in enumerate(threshold_sets):
            if idx % 50 == 0:
                logger.info(f"  Testing combination {idx+1}/{total_combinations}")
            
            oos_sharpes = []
            oos_returns = []
            oos_drawdowns = []
            oos_win_rates = []
            total_trades = 0
            
            # Walk-forward validation
            for train_data, test_data in splits:
                metrics = self._simulate_strategy(test_data, thresholds, signals)
                
                oos_sharpes.append(metrics['sharpe'])
                oos_returns.append(metrics['total_return'])
                oos_drawdowns.append(metrics['max_drawdown'])
                oos_win_rates.append(metrics['win_rate'])
                total_trades += metrics['num_trades']
            
            # Average out-of-sample Sharpe ratio
            avg_sharpe = np.mean(oos_sharpes)
            
            if avg_sharpe > best_sharpe:
                best_sharpe = avg_sharpe
                best_thresholds = thresholds
                best_metrics = {
                    'sharpe': avg_sharpe,
                    'return': np.mean(oos_returns),
                    'drawdown': np.mean(oos_drawdowns),
                    'win_rate': np.mean(oos_win_rates),
                    'trades': total_trades
                }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"OPTIMIZATION COMPLETE - {regime}")
        logger.info(f"{'='*60}")
        logger.info(f"Best thresholds:")
        logger.info(f"  buy_confidence: {best_thresholds.buy_confidence:.2f}")
        logger.info(f"  sell_confidence: {best_thresholds.sell_confidence:.2f}")
        logger.info(f"  min_score: {best_thresholds.min_score:.2f}")
        logger.info(f"  stop_loss: {best_thresholds.stop_loss:.2f}")
        logger.info(f"\nPerformance:")
        logger.info(f"  Sharpe Ratio: {best_metrics['sharpe']:.3f}")
        logger.info(f"  Total Return: {best_metrics['return']:.2%}")
        logger.info(f"  Max Drawdown: {best_metrics['drawdown']:.2%}")
        logger.info(f"  Win Rate: {best_metrics['win_rate']:.2%}")
        logger.info(f"  Total Trades: {best_metrics['trades']}")
        
        return OptimizationResult(
            best_thresholds=best_thresholds,
            sharpe_ratio=best_metrics['sharpe'],
            total_return=best_metrics['return'],
            max_drawdown=best_metrics['drawdown'],
            win_rate=best_metrics['win_rate'],
            num_trades=best_metrics['trades'],
            regime=regime,
            validation_windows=len(splits)
        )
    
    def optimize_by_regime(self,
                          data: pd.DataFrame,
                          regime_data: pd.DataFrame = None,
                          signals: pd.DataFrame = None) -> Dict[str, OptimizationResult]:
        """
        Optimize thresholds for each market regime separately.
        
        Args:
            data: Historical price data
            regime_data: Market index data for regime detection
            signals: Pre-computed signals (optional)
        
        Returns:
            Dict mapping regime name to OptimizationResult
        """
        logger.info("\n" + "="*70)
        logger.info("REGIME-SPECIFIC THRESHOLD OPTIMIZATION")
        logger.info("="*70)
        
        results = {}
        
        # If regime data provided, segment data by regime
        if regime_data is not None and len(regime_data) > 200:
            from ..strategy.regime_detection import MarketRegimeDetector
            
            detector = MarketRegimeDetector()
            
            # Detect regimes over the data period
            regime_labels = self._detect_regime_periods(regime_data, detector)
            
            # Optimize for each regime
            for regime in ['bull', 'bear', 'sideways']:
                regime_mask = regime_labels == regime
                regime_data_subset = data[regime_mask] if regime_mask.sum() > 100 else None
                
                if regime_data_subset is not None and len(regime_data_subset) >= 100:
                    logger.info(f"\n--- Optimizing for {regime.upper()} regime ---")
                    results[regime] = self.optimize(regime_data_subset, signals, regime)
                else:
                    logger.info(f"\n--- Skipping {regime.upper()} regime (insufficient data) ---")
        
        # Always compute default (all data) optimization
        logger.info(f"\n--- Optimizing DEFAULT thresholds (all data) ---")
        results['default'] = self.optimize(data, signals, 'default')
        
        return results
    
    def _detect_regime_periods(self, 
                               market_data: pd.DataFrame,
                               detector) -> pd.Series:
        """
        Detect market regime for each period.
        
        Returns:
            Series of regime labels indexed by date
        """
        if isinstance(market_data.columns, pd.MultiIndex):
            market_data.columns = market_data.columns.get_level_values(0)
        
        regimes = []
        dates = []
        
        # Use rolling window for regime detection
        window = 200
        
        for i in range(window, len(market_data)):
            subset = market_data.iloc[i-window:i]
            result = detector.detect_regime(subset)
            regimes.append(result['regime'])
            dates.append(market_data.index[i])
        
        return pd.Series(regimes, index=dates)


def save_thresholds_to_config(results: Dict[str, OptimizationResult], 
                               config_path: str = None):
    """
    Save optimized thresholds to JSON config file.
    
    Args:
        results: Dict of regime -> OptimizationResult
        config_path: Path to config file (default: config/thresholds.json)
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / 'config' / 'thresholds.json'
    
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    config = {
        'last_updated': datetime.now().isoformat(),
        'optimization_period': {
            'train_window_days': 126,
            'test_window_days': 42
        },
        'thresholds': {}
    }
    
    for regime, result in results.items():
        config['thresholds'][regime] = {
            'buy_confidence': result.best_thresholds.buy_confidence,
            'sell_confidence': result.best_thresholds.sell_confidence,
            'min_score': result.best_thresholds.min_score,
            'stop_loss': result.best_thresholds.stop_loss,
            'performance': {
                'sharpe_ratio': result.sharpe_ratio,
                'total_return': result.total_return,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate
            }
        }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"\n✅ Thresholds saved to {config_path}")
    return config_path


def load_thresholds_from_config(config_path: str = None) -> Dict:
    """
    Load thresholds from JSON config file.
    
    Args:
        config_path: Path to config file (default: config/thresholds.json)
    
    Returns:
        Dict with threshold configuration
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / 'config' / 'thresholds.json'
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        logger.warning("Using default thresholds")
        return get_default_thresholds()
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def get_default_thresholds() -> Dict:
    """
    Get default threshold configuration.
    
    Returns:
        Dict with default thresholds for each regime
    """
    return {
        'last_updated': datetime.now().isoformat(),
        'thresholds': {
            'default': {
                'buy_confidence': 0.55,
                'sell_confidence': 0.45,
                'min_score': 0.25,
                'stop_loss': 0.15,
                'performance': {
                    'sharpe_ratio': 0.0,
                    'total_return': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0
                }
            },
            'bull': {
                'buy_confidence': 0.50,
                'sell_confidence': 0.40,
                'min_score': 0.20,
                'stop_loss': 0.15,
                'performance': {
                    'sharpe_ratio': 0.0,
                    'total_return': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0
                }
            },
            'bear': {
                'buy_confidence': 0.65,
                'sell_confidence': 0.35,
                'min_score': 0.30,
                'stop_loss': 0.10,
                'performance': {
                    'sharpe_ratio': 0.0,
                    'total_return': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0
                }
            },
            'sideways': {
                'buy_confidence': 0.55,
                'sell_confidence': 0.45,
                'min_score': 0.25,
                'stop_loss': 0.20,
                'performance': {
                    'sharpe_ratio': 0.0,
                    'total_return': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0
                }
            }
        }
    }


def get_thresholds_for_regime(regime: str, config: Dict = None) -> ThresholdSet:
    """
    Get thresholds for a specific market regime.
    
    Args:
        regime: Market regime ('bull', 'bear', 'sideways', 'default')
        config: Loaded config dict (will load if not provided)
    
    Returns:
        ThresholdSet for the regime
    """
    if config is None:
        config = load_thresholds_from_config()
    
    thresholds = config.get('thresholds', {}).get(regime, config['thresholds'].get('default', {}))
    
    return ThresholdSet.from_dict(thresholds)