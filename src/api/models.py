"""
Pydantic models for CampusGPT API
Request/response schemas for the FastAPI server
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class ChatContext(str, Enum):
    """Available conversation contexts"""
    ACADEMIC = "academic"
    STUDENT_LIFE = "student_life"
    ADMINISTRATIVE = "administrative"
    CAMPUS_SERVICES = "campus_services"
    GENERAL = "general"


class DifficultyLevel(str, Enum):
    """Query difficulty levels"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ChatRequest(BaseModel):
    """Single chat request model"""
    message: str = Field(..., description="User query message", min_length=1, max_length=2000)
    context: Optional[ChatContext] = Field(ChatContext.GENERAL, description="Query context category")
    max_tokens: Optional[int] = Field(256, description="Maximum tokens to generate", ge=1, le=1024)
    temperature: Optional[float] = Field(0.7, description="Sampling temperature", ge=0.0, le=1.0)
    top_p: Optional[float] = Field(0.9, description="Nucleus sampling parameter", ge=0.0, le=1.0)
    include_alternatives: Optional[bool] = Field(False, description="Include alternative responses")
    
    class Config:
        schema_extra = {
            "example": {
                "message": "What are the dining hall hours during finals week?",
                "context": "student_life",
                "max_tokens": 200,
                "temperature": 0.7,
                "include_alternatives": True
            }
        }


class ResponseMetadata(BaseModel):
    """Response metadata information"""
    model: str = Field(..., description="Model name and version")
    category: ChatContext = Field(..., description="Detected query category")
    confidence: float = Field(..., description="Response confidence score", ge=0.0, le=1.0)
    tokens_used: int = Field(..., description="Number of tokens generated")
    latency_ms: int = Field(..., description="Generation latency in milliseconds")
    sources: Optional[List[str]] = Field(None, description="Information sources used")
    difficulty: Optional[DifficultyLevel] = Field(None, description="Detected query difficulty")


class AlternativeResponse(BaseModel):
    """Alternative response option"""
    response: str = Field(..., description="Alternative response text")
    confidence: float = Field(..., description="Confidence score for this alternative")


class ChatResponse(BaseModel):
    """Single chat response model"""
    response: str = Field(..., description="Generated response text")
    metadata: ResponseMetadata = Field(..., description="Response metadata")
    alternatives: Optional[List[AlternativeResponse]] = Field(None, description="Alternative responses")
    
    class Config:
        schema_extra = {
            "example": {
                "response": "During finals week, the dining halls extend their hours. Main Dining Hall is open 7 AM - 11 PM, and Late Night Grill operates until 2 AM for students studying late.",
                "metadata": {
                    "model": "campusgpt-v1.2",
                    "category": "student_life",
                    "confidence": 0.94,
                    "tokens_used": 45,
                    "latency_ms": 234,
                    "sources": ["dining_services", "finals_schedule"],
                    "difficulty": "easy"
                },
                "alternatives": [
                    {
                        "response": "Check the dining services website for specific finals week hours...",
                        "confidence": 0.67
                    }
                ]
            }
        }


class BatchQuery(BaseModel):
    """Single query in a batch request"""
    message: str = Field(..., description="User query message")
    context: Optional[ChatContext] = Field(ChatContext.GENERAL, description="Query context")
    max_tokens: Optional[int] = Field(256, description="Max tokens for this query")
    temperature: Optional[float] = Field(0.7, description="Temperature for this query")


class BatchRequest(BaseModel):
    """Batch processing request"""
    queries: List[BatchQuery] = Field(..., description="List of queries to process", min_items=1, max_items=50)
    
    class Config:
        schema_extra = {
            "example": {
                "queries": [
                    {
                        "message": "What are the library hours?",
                        "context": "campus_services"
                    },
                    {
                        "message": "How do I register for classes?",
                        "context": "academic"
                    }
                ]
            }
        }


class BatchResponse(BaseModel):
    """Batch processing response"""
    responses: List[ChatResponse] = Field(..., description="Generated responses")
    total_queries: int = Field(..., description="Total number of queries")
    processed_queries: int = Field(..., description="Successfully processed queries")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Overall service status")
    timestamp: float = Field(..., description="Health check timestamp")
    model_status: str = Field(..., description="Model loading status")
    version: str = Field(..., description="API version")
    error: Optional[str] = Field(None, description="Error message if unhealthy")


class ModelCapabilities(BaseModel):
    """Model capabilities information"""
    domains: List[str] = Field(..., description="Supported knowledge domains")
    languages: List[str] = Field(..., description="Supported languages")
    max_context_length: int = Field(..., description="Maximum context length in tokens")
    fine_tuned_on: List[str] = Field(..., description="Fine-tuning datasets")


class ModelPerformance(BaseModel):
    """Model performance metrics"""
    bleu_score: float = Field(..., description="BLEU score on test set")
    perplexity: float = Field(..., description="Model perplexity")
    f1_score: float = Field(..., description="F1 score for accuracy")
    response_accuracy: float = Field(..., description="Overall response accuracy percentage")
    category_breakdown: Dict[str, float] = Field(..., description="Accuracy by category")


class ModelInfo(BaseModel):
    """Complete model information"""
    model_name: str = Field(..., description="Model name and version")
    base_model: str = Field(..., description="Base model used for fine-tuning")
    training_data_size: int = Field(..., description="Number of training examples")
    capabilities: ModelCapabilities = Field(..., description="Model capabilities")
    performance: ModelPerformance = Field(..., description="Performance metrics")
    last_updated: datetime = Field(..., description="Last model update timestamp")


class EvaluationMetric(str, Enum):
    """Available evaluation metrics"""
    BLEU = "bleu"
    ROUGE = "rouge"
    PERPLEXITY = "perplexity"
    SIMILARITY = "similarity"
    ACCURACY = "accuracy"
    F1 = "f1"


class EvaluationRequest(BaseModel):
    """Model evaluation request"""
    test_data_path: str = Field(..., description="Path to test dataset")
    metrics: List[EvaluationMetric] = Field(..., description="Metrics to compute")
    output_path: Optional[str] = Field(None, description="Output path for results")
    
    class Config:
        schema_extra = {
            "example": {
                "test_data_path": "data/evaluation/campus_test.jsonl",
                "metrics": ["bleu", "rouge", "accuracy"],
                "output_path": "results/evaluation_report.json"
            }
        }


class FeedbackRequest(BaseModel):
    """User feedback on responses"""
    query_id: str = Field(..., description="ID of the original query")
    rating: int = Field(..., description="Rating from 1-5", ge=1, le=5)
    feedback_text: Optional[str] = Field(None, description="Optional feedback text")
    category_correct: Optional[bool] = Field(None, description="Was the category detection correct?")
    
    class Config:
        schema_extra = {
            "example": {
                "query_id": "12345",
                "rating": 4,
                "feedback_text": "Response was helpful but could be more specific",
                "category_correct": True
            }
        }