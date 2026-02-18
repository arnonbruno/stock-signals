"""
Centralized configuration for the stock signals system.

This module contains all strategy parameters, thresholds, and settings
that should be consistent across the codebase.

Thresholds are loaded from config/thresholds.json at runtime,
allowing updates without code changes.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict
import json
import os
from pathlib import Path


def _load_thresholds_config() -> Dict:
    """Load thresholds from JSON config file."""
    config_path = Path(__file__).parent.parent / 'config' / 'thresholds.json'
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    
    # Return defaults if file not found or error
    return {
        'thresholds': {
            'default': {
                'buy_confidence': 0.55,
                'sell_confidence': 0.45,
                'min_score': 0.25,
                'stop_loss': 0.15
            },
            'bull': {
                'buy_confidence': 0.50,
                'sell_confidence': 0.40,
                'min_score': 0.20,
                'stop_loss': 0.15
            },
            'bear': {
                'buy_confidence': 0.65,
                'sell_confidence': 0.35,
                'min_score': 0.30,
                'stop_loss': 0.10
            },
            'sideways': {
                'buy_confidence': 0.55,
                'sell_confidence': 0.45,
                'min_score': 0.25,
                'stop_loss': 0.20
            }
        }
    }


# Load thresholds at module import
_THRESHOLD_CONFIG = _load_thresholds_config()


@dataclass
class ThresholdConfig:
    """Configuration for trading thresholds."""
    buy_confidence: float = 0.55
    sell_confidence: float = 0.45
    min_score: float = 0.25
    stop_loss: float = 0.15
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'ThresholdConfig':
        return cls(
            buy_confidence=d.get('buy_confidence', 0.55),
            sell_confidence=d.get('sell_confidence', 0.45),
            min_score=d.get('min_score', 0.25),
            stop_loss=d.get('stop_loss', 0.15)
        )
    
    @classmethod
    def for_regime(cls, regime: str = 'default') -> 'ThresholdConfig':
        """Get thresholds for a specific market regime."""
        thresholds = _THRESHOLD_CONFIG.get('thresholds', {}).get(regime)
        if thresholds is None:
            thresholds = _THRESHOLD_CONFIG.get('thresholds', {}).get('default', {})
        return cls.from_dict(thresholds)


@dataclass
class StrategyConfig:
    """Centralized strategy configuration."""
    
    # Confidence thresholds (loaded from config file)
    MIN_CONFIDENCE: float = 0.50  # Minimum confidence for trade signals
    
    # Position sizing
    MAX_POSITION_SIZE: float = 0.60  # Maximum position size (60%)
    MIN_POSITION_SIZE: float = 0.10  # Minimum position size (10%)
    DEFAULT_POSITION_SIZE: float = 0.20  # Default when calculation fails
    KELLY_FRACTION: float = 0.5  # Half-Kelly for safety
    
    # Trend detection periods
    MACRO_PERIOD: int = 50
    MICRO_PERIOD: int = 20
    
    # News settings
    NEWS_CACHE_TTL_HOURS: int = 6
    MIN_NEWS_SENTIMENT: float = 0.1  # Minimum sentiment to affect position sizing
    
    # Risk management
    MAX_PORTFOLIO_EXPOSURE: float = 0.80  # Maximum total portfolio exposure
    
    # Kelly Criterion specific
    KELLY_MIN_DATA_POINTS: int = 30  # Minimum data points for Kelly calculation
    
    # Regime-specific thresholds (loaded from config)
    _thresholds: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Load thresholds from config file."""
        self._thresholds = _THRESHOLD_CONFIG.get('thresholds', {})
    
    def get_thresholds(self, regime: str = 'default') -> ThresholdConfig:
        """Get thresholds for a specific market regime."""
        return ThresholdConfig.for_regime(regime)
    
    def get_buy_confidence(self, regime: str = 'default') -> float:
        """Get buy confidence threshold for regime."""
        return self.get_thresholds(regime).buy_confidence
    
    def get_sell_confidence(self, regime: str = 'default') -> float:
        """Get sell confidence threshold for regime."""
        return self.get_thresholds(regime).sell_confidence
    
    def get_min_score(self, regime: str = 'default') -> float:
        """Get minimum score threshold for regime."""
        return self.get_thresholds(regime).min_score
    
    def get_stop_loss(self, regime: str = 'default') -> float:
        """Get stop loss threshold for regime."""
        return self.get_thresholds(regime).stop_loss
    
    def reload_thresholds(self):
        """Reload thresholds from config file."""
        global _THRESHOLD_CONFIG
        _THRESHOLD_CONFIG = _load_thresholds_config()
        self._thresholds = _THRESHOLD_CONFIG.get('thresholds', {})


# Global config instance
CONFIG = StrategyConfig()


def get_config() -> StrategyConfig:
    """Get the global configuration instance."""
    return CONFIG


def get_thresholds(regime: str = 'default') -> ThresholdConfig:
    """Convenience function to get thresholds for a regime."""
    return CONFIG.get_thresholds(regime)


def reload_config():
    """Reload configuration from files."""
    global _THRESHOLD_CONFIG, CONFIG
    _THRESHOLD_CONFIG = _load_thresholds_config()
    CONFIG = StrategyConfig()
    return CONFIG