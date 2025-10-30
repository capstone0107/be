"""
Analysis router for conversation analysis.
"""
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List

from models import (
    AnalyzeConversationRequest,
    AnalyzeConversationResponse,
    MetacognitiveInsight,
    ExternalVerification
)
from services.analysis_service import analysis_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["analysis"])


@router.post("/analyze", response_model=AnalyzeConversationResponse)
async def analyze_conversation(request: AnalyzeConversationRequest):
    """
    Analyze a conversation and return insights.
    
    Args:
        request: Conversation messages to analyze
        
    Returns:
        Analysis results with metacognitive insights and external verifications
    """
    try:
        if not analysis_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="분석 서비스를 사용할 수 없습니다. OpenAI API 키를 확인하세요."
            )
        
        # Convert Pydantic models to dict for processing
        messages = [msg.dict() for msg in request.messages]
        
        # Analyze conversation
        result = analysis_service.analyze_conversation(messages)
        
        # Check for errors
        if "error" in result:
            raise HTTPException(
                status_code=500,
                detail=result["error"]
            )
        
        # Add timestamp
        result["analyzed_at"] = datetime.now().isoformat()
        
        # Create response
        response = AnalyzeConversationResponse(
            overall_summary=result.get("overall_summary", ""),
            metacognitive_insights=[
                MetacognitiveInsight(**insight)
                for insight in result.get("metacognitive_insights", [])
            ],
            external_verifications=[
                ExternalVerification(**verification)
                for verification in result.get("external_verifications", [])
            ],
            analyzed_at=result.get("analyzed_at"),
            message_count=result.get("message_count")
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis/status")
async def get_analysis_status():
    """
    Check if the analysis service is available.
    
    Returns:
        Status of the analysis service
    """
    return {
        "available": analysis_service.is_available(),
        "service": "conversation_analysis",
        "model": "gpt-4-turbo-preview"
    }