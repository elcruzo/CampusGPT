import os
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.models import (
    ChatRequest, ChatResponse, BatchRequest, BatchResponse,
    HealthResponse, ModelInfo, EvaluationRequest
)
from src.api.services.inference_service import InferenceService
from src.api.services.evaluation_service import EvaluationService
from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger(__name__)

inference_service: Optional[InferenceService] = None
evaluation_service: Optional[EvaluationService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference_service, evaluation_service
    
    logger.info("starting campusgpt api...")
    
    try:
        config = load_config("config/inference_config.yaml")
        
        inference_service = InferenceService(config)
        await inference_service.initialize()
        
        evaluation_service = EvaluationService(config)
        
        logger.info("services initialized")
        
    except Exception as e:
        logger.error(f"failed to initialize: {str(e)}")
        raise e
    
    yield
    
    logger.info("shutting down...")
    
    if inference_service:
        await inference_service.shutdown()
    
    logger.info("shutdown complete")


app = FastAPI(
    title="CampusGPT API",
    description="llama 2 fine-tuned on campus Q&A",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "campusgpt api", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        model_status = "healthy" if inference_service and inference_service.is_ready else "unavailable"
        
        return HealthResponse(
            status="healthy" if model_status == "healthy" else "degraded",
            timestamp=asyncio.get_event_loop().time(),
            model_status=model_status,
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            timestamp=asyncio.get_event_loop().time(),
            model_status="error",
            version="1.0.0",
            error=str(e)
        )


@app.get("/api/v1/model/info", response_model=ModelInfo)
async def model_info():
    if not inference_service:
        raise HTTPException(status_code=503, detail="service unavailable")
    
    return await inference_service.get_model_info()


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not inference_service:
        raise HTTPException(status_code=503, detail="service unavailable")
    
    try:
        response = await inference_service.generate_response(
            message=request.message,
            context=request.context,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            include_alternatives=request.include_alternatives
        )
        
        return response
        
    except Exception as e:
        logger.error(f"generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"generation failed: {str(e)}")


@app.post("/api/v1/batch", response_model=BatchResponse)
async def batch_chat(request: BatchRequest):
    if not inference_service:
        raise HTTPException(status_code=503, detail="service unavailable")
    
    try:
        responses = await inference_service.generate_batch_responses(request.queries)
        
        return BatchResponse(
            responses=responses,
            total_queries=len(request.queries),
            processed_queries=len(responses)
        )
        
    except Exception as e:
        logger.error(f"batch failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"batch failed: {str(e)}")


@app.post("/api/v1/evaluate")
async def evaluate_model(request: EvaluationRequest, background_tasks: BackgroundTasks):
    if not evaluation_service:
        raise HTTPException(status_code=503, detail="service unavailable")
    
    background_tasks.add_task(
        evaluation_service.run_evaluation,
        request.test_data_path,
        request.metrics,
        request.output_path
    )
    
    return {"message": "evaluation started", "status": "processing"}


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "internal server error", "error": str(exc)}
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CampusGPT API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
        reload=False
    )