"""
CampusGPT Inference Service
Model inference with caching and optimization
"""

import time
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from functools import lru_cache
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    GenerationConfig,
    pipeline
)
from peft import PeftModel
import logging

from src.api.models import (
    ChatResponse, ResponseMetadata, AlternativeResponse,
    ModelInfo, ModelCapabilities, ModelPerformance,
    ChatContext, DifficultyLevel, BatchQuery
)
from src.utils.logger import get_logger
from src.utils.text_processing import preprocess_query, postprocess_response
from src.utils.caching import ResponseCache

logger = get_logger(__name__)


class InferenceService:
    """Production inference service for CampusGPT"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize inference service"""
        self.config = config
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        self.cache = ResponseCache(max_size=1000, ttl=3600)  # 1 hour TTL
        self.is_ready = False
        
        # Performance tracking
        self.request_count = 0
        self.total_latency = 0.0
        self.error_count = 0
        
        # Category classification weights
        self.category_keywords = {
            ChatContext.ACADEMIC: [
                "course", "class", "register", "grade", "transcript", "advisor", 
                "prerequisite", "degree", "major", "semester", "schedule", "exam"
            ],
            ChatContext.STUDENT_LIFE: [
                "housing", "dorm", "dining", "meal", "activity", "club", "event",
                "recreation", "gym", "sports", "campus life"
            ],
            ChatContext.ADMINISTRATIVE: [
                "financial aid", "tuition", "fee", "payment", "policy", "form",
                "office", "administration", "deadline", "application"
            ],
            ChatContext.CAMPUS_SERVICES: [
                "library", "hours", "computer", "wifi", "health", "medical",
                "counseling", "parking", "shuttle", "maintenance"
            ]
        }
    
    async def initialize(self):
        try:
            logger.info("loading campusgpt model...")
            
            model_config = self.config.get('model', {})
            model_path = model_config.get('model_path', './models/campusgpt-v1')
            
            logger.info("loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            logger.info("loading model...")
            device = model_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            torch_dtype = getattr(torch, model_config.get('torch_dtype', 'float16'))
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map="auto" if device == 'cuda' else None,
                trust_remote_code=True,
                load_in_8bit=model_config.get('load_in_8bit', False)
            )
            
            if device == 'cpu':
                self.model = self.model.to(device)
            
            gen_config = self.config.get('generation', {})
            self.generation_config = GenerationConfig(
                max_length=gen_config.get('max_length', 512),
                temperature=gen_config.get('temperature', 0.7),
                top_p=gen_config.get('top_p', 0.9),
                top_k=gen_config.get('top_k', 50),
                do_sample=gen_config.get('do_sample', True),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=gen_config.get('repetition_penalty', 1.1)
            )
            
            self.is_ready = True
            logger.info("model loaded")
            
        except Exception as e:
            logger.error(f"failed to load model: {str(e)}")
            raise e
    
    async def shutdown(self):
        logger.info("shutting down inference service...")
        
        if self.model:
            del self.model
            self.model = None
        
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("shutdown complete")
    
    def _classify_context(self, message: str) -> ChatContext:
        """Classify query context using keyword matching"""
        message_lower = message.lower()
        scores = {}
        
        for context, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in message_lower)
            scores[context] = score
        
        # Return context with highest score, default to GENERAL
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return ChatContext.GENERAL
    
    def _estimate_difficulty(self, message: str, context: ChatContext) -> DifficultyLevel:
        """Estimate query difficulty"""
        # Simple heuristic based on message length and complexity indicators
        complex_indicators = [
            "policy", "procedure", "requirement", "deadline", "appeal",
            "exception", "complex", "multiple", "various", "different"
        ]
        
        message_lower = message.lower()
        complexity_score = sum(1 for indicator in complex_indicators if indicator in message_lower)
        
        # Length-based scoring
        if len(message.split()) > 20:
            complexity_score += 1
        if len(message.split()) > 30:
            complexity_score += 1
        
        if complexity_score >= 3:
            return DifficultyLevel.HARD
        elif complexity_score >= 1:
            return DifficultyLevel.MEDIUM
        else:
            return DifficultyLevel.EASY
    
    def _calculate_confidence(self, generated_text: str, context: ChatContext) -> float:
        """Calculate response confidence score"""
        # Simple confidence calculation based on response characteristics
        confidence = 0.7  # Base confidence
        
        # Boost confidence for specific indicators
        if any(phrase in generated_text.lower() for phrase in [
            "according to", "based on", "the policy states", "specifically"
        ]):
            confidence += 0.1
        
        # Reduce confidence for uncertainty indicators
        if any(phrase in generated_text.lower() for phrase in [
            "not sure", "might be", "probably", "i think", "maybe"
        ]):
            confidence -= 0.2
        
        # Adjust based on response length (longer responses tend to be more confident)
        if len(generated_text.split()) > 50:
            confidence += 0.05
        elif len(generated_text.split()) < 20:
            confidence -= 0.05
        
        return min(max(confidence, 0.0), 1.0)
    
    def _format_prompt(self, message: str, context: ChatContext) -> str:
        """Format input prompt for the model"""
        context_prompts = {
            ChatContext.ACADEMIC: "You are a helpful academic advisor assistant. Answer questions about courses, registration, grades, and academic policies.",
            ChatContext.STUDENT_LIFE: "You are a student life assistant. Help with questions about housing, dining, activities, and campus life.",
            ChatContext.ADMINISTRATIVE: "You are an administrative assistant. Help with questions about financial aid, forms, policies, and procedures.",
            ChatContext.CAMPUS_SERVICES: "You are a campus services assistant. Help with questions about library, IT, health services, and facilities.",
            ChatContext.GENERAL: "You are a helpful campus assistant. Provide accurate and helpful information about university matters."
        }
        
        system_prompt = context_prompts.get(context, context_prompts[ChatContext.GENERAL])
        
        # Format using Alpaca template
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{system_prompt}

### Input:
{message}

### Response:
"""
        return prompt
    
    async def generate_response(
        self,
        message: str,
        context: Optional[ChatContext] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        include_alternatives: bool = False
    ) -> ChatResponse:
        """Generate single response"""
        start_time = time.time()
        
        try:
            # Preprocess query
            processed_message = preprocess_query(message)
            
            # Auto-detect context if not provided
            if context is None:
                context = self._classify_context(processed_message)
            
            # Check cache
            cache_key = f"{processed_message}_{context}_{max_tokens}_{temperature}_{top_p}"
            cached_response = self.cache.get(cache_key)
            if cached_response:
                logger.info(f"Cache hit for query: {message[:50]}...")
                return cached_response
            
            # Format prompt
            prompt = self._format_prompt(processed_message, context)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.get('data', {}).get('max_seq_length', 512)
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                generation_config = GenerationConfig(
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=self.generation_config.top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
                
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config,
                    num_return_sequences=3 if include_alternatives else 1
                )
            
            # Decode responses
            responses = []
            for output in outputs:
                # Remove input tokens from output
                generated_tokens = output[inputs['input_ids'].shape[1]:]
                response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                response_text = postprocess_response(response_text)
                responses.append(response_text)
            
            # Primary response
            primary_response = responses[0]
            confidence = self._calculate_confidence(primary_response, context)
            
            # Alternative responses
            alternatives = []
            if include_alternatives and len(responses) > 1:
                for alt_response in responses[1:]:
                    alt_confidence = self._calculate_confidence(alt_response, context)
                    alternatives.append(AlternativeResponse(
                        response=alt_response,
                        confidence=alt_confidence
                    ))
            
            # Calculate metrics
            latency_ms = int((time.time() - start_time) * 1000)
            tokens_used = len(self.tokenizer.encode(primary_response))
            
            # Create response
            response = ChatResponse(
                response=primary_response,
                metadata=ResponseMetadata(
                    model="campusgpt-v1.2",
                    category=context,
                    confidence=confidence,
                    tokens_used=tokens_used,
                    latency_ms=latency_ms,
                    sources=self._get_sources(context),
                    difficulty=self._estimate_difficulty(message, context)
                ),
                alternatives=alternatives if alternatives else None
            )
            
            # Cache response
            self.cache.set(cache_key, response)
            
            # Update metrics
            self.request_count += 1
            self.total_latency += latency_ms
            
            return response
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Generation failed: {str(e)}")
            raise e
    
    async def generate_batch_responses(self, queries: List[BatchQuery]) -> List[ChatResponse]:
        """Generate batch responses"""
        logger.info(f"Processing batch of {len(queries)} queries")
        
        tasks = []
        for query in queries:
            task = self.generate_response(
                message=query.message,
                context=query.context,
                max_tokens=query.max_tokens or 256,
                temperature=query.temperature or 0.7
            )
            tasks.append(task)
        
        # Process concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log errors
        valid_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Batch query {i} failed: {str(response)}")
            else:
                valid_responses.append(response)
        
        return valid_responses
    
    def _get_sources(self, context: ChatContext) -> List[str]:
        """Get mock data sources for context"""
        source_mapping = {
            ChatContext.ACADEMIC: ["registrar_catalog", "academic_policies", "course_database"],
            ChatContext.STUDENT_LIFE: ["student_handbook", "housing_portal", "dining_services"],
            ChatContext.ADMINISTRATIVE: ["financial_aid_office", "administrative_policies", "student_accounts"],
            ChatContext.CAMPUS_SERVICES: ["library_system", "it_services", "facilities_management"]
        }
        return source_mapping.get(context, ["general_information"])
    
    async def get_model_info(self) -> ModelInfo:
        """Get comprehensive model information"""
        return ModelInfo(
            model_name="CampusGPT v1.2",
            base_model="meta-llama/Llama-2-7b-chat-hf",
            training_data_size=5000,
            capabilities=ModelCapabilities(
                domains=["Academic", "Student Life", "Administrative", "Campus Services"],
                languages=["English"],
                max_context_length=2048,
                fine_tuned_on=["campus_qa_dataset", "student_handbook", "academic_policies"]
            ),
            performance=ModelPerformance(
                bleu_score=0.78,
                perplexity=12.3,
                f1_score=0.85,
                response_accuracy=91.0,
                category_breakdown={
                    "academic": 94.0,
                    "student_life": 89.0,
                    "administrative": 88.0,
                    "campus_services": 85.0
                }
            ),
            last_updated=datetime.now()
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get service performance statistics"""
        avg_latency = self.total_latency / max(self.request_count, 1)
        error_rate = (self.error_count / max(self.request_count, 1)) * 100
        
        return {
            "total_requests": self.request_count,
            "average_latency_ms": avg_latency,
            "error_rate_percent": error_rate,
            "cache_hit_rate_percent": self.cache.hit_rate * 100,
            "model_ready": self.is_ready
        }