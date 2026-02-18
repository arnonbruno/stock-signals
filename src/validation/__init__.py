"""
Validation module for stock signals.

Includes walk-forward validation and threshold optimization.
"""

from .walk_forward import WalkForwardValidator, run_walk_forward_validation
from .threshold_optimizer import (
    ThresholdOptimizer,
    ThresholdSet,
    OptimizationResult,
    save_thresholds_to_config,
    load_thresholds_from_config,
    get_thresholds_for_regime,
    get_default_thresholds
)

__all__ = [
    'WalkForwardValidator',
    'run_walk_forward_validation',
    'ThresholdOptimizer',
    'ThresholdSet',
    'OptimizationResult',
    'save_thresholds_to_config',
    'load_thresholds_from_config',
    'get_thresholds_for_regime',
    'get_default_thresholds'
]