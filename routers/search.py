"""
Search router for web-based search using GPT-4o Search Preview.
"""
import logging
from fastapi import APIRouter, HTTPException

from services import llm_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["search"])


@router.post("/search")
async def search_query(request: dict):
    """
    Query using GPT-4o Search Preview with web search capabilities.
    Returns answer with sources instead of cards.
    """
    try:
        # request에서 question을 추출 (리스트 형태로 들어올 수 있음)
        question = request.get("question", "")
        
        # question이 리스트인 경우 마지막 질문만 사용
        if isinstance(question, list):
            question = question[-1] if question else ""
        
        if not question.strip():
            raise HTTPException(status_code=400, detail="질문이 비어있습니다.")
        
        # LLM 서비스로 검색
        result = llm_service.generate_text(question)
        
        return {
            "answer": result.answer,
            "sources": [
                {
                    "title": source.title,
                    "url": source.url,
                    "snippet": source.snippet
                }
                for source in result.sources
            ]
        }
    except Exception as e:
        logger.error(f"Error in search query: {e}")
        raise HTTPException(status_code=500, detail=str(e))