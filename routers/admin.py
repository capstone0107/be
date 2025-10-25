"""
Admin router for system management operations.
"""
import logging
from fastapi import APIRouter, HTTPException

from models import ReloadResponse
from services import langchain_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["admin"])


@router.post("/reload", response_model=ReloadResponse)
async def reload_documents():
    """
    Reload documents and reinitialize the RAG system.
    """
    try:
        langchain_service.initialize_rag_system()
        return ReloadResponse(
            status="success",
            message="Documents reloaded successfully"
        )
    except Exception as e:
        logger.error(f"Error reloading documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))
