"""
Chat router for direct ChatGPT conversation.
"""
import os
import logging
from fastapi import APIRouter, HTTPException

from models import ChatRequest, ChatResponse, EmbeddingRequest, EmbeddingResponse
from services import langchain_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint using OpenAI's ChatGPT.
    """
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        response = langchain_service.chat(request.message)
        return ChatResponse(response=response)
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embed", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest):
    """
    Create embeddings for a given text using OpenAI embeddings.
    """
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        embedding_vector = langchain_service.create_embedding(request.text)
        
        return EmbeddingResponse(
            embedding=embedding_vector,
            dimension=len(embedding_vector)
        )
    except Exception as e:
        logger.error(f"Error creating embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))
