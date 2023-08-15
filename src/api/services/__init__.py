"""
CampusGPT API services
Production-ready inference and evaluation services
"""

from .inference_service import InferenceService
from .evaluation_service import EvaluationService

__all__ = ["InferenceService", "EvaluationService"]