"""
Query router for RAG-based document querying.
"""
import logging
from fastapi import APIRouter, HTTPException

from models import QueryRequest, QueryResponse
from services import langchain_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query documents using RAG (Retrieval Augmented Generation).
    """
    try:
        if not langchain_service.is_initialized():
            raise HTTPException(
                status_code=503,
                detail="RAG system not initialized. Please ensure documents are loaded."
            )
        
        result = langchain_service.query(request.question)
        
        return QueryResponse(
            answer=result["answer"],
            cards=result.get("cards", [])
        )
    except ValueError as e:
        logger.error(f"RAG system error: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Error querying documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))
