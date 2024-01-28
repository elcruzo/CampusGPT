"""
CampusGPT evaluation package
Comprehensive model evaluation tools and metrics
"""

from .metrics import compute_metrics, EvaluationSuite
from .evaluators import BLEUEvaluator, ROUGEEvaluator, AccuracyEvaluator

__all__ = [
    "compute_metrics", 
    "EvaluationSuite",
    "BLEUEvaluator", 
    "ROUGEEvaluator", 
    "AccuracyEvaluator"
]