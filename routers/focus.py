"""
Focus router for conversation classification.
"""
import logging
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any

from services.focus_service import focus_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/focus", tags=["focus"])


@router.post("/classify")
async def classify_conversation(request: Dict[str, Any]):
    """
    Classify a conversation into focus topics.
    
    Request body:
    {
        "conversation_id": "conv_123",
        "messages": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }
    """
    print("Classify conversation request received:", request)
    try:
        conversation_id = request.get("conversation_id")
        messages = request.get("messages", [])
        print("Received messages for classification:", messages)
        if not conversation_id or not messages:
            raise HTTPException(
                status_code=400,
                detail="conversation_id and messages are required"
            )
        
        result = focus_service.classify_conversation(conversation_id, messages)
        
        return result
        
    except Exception as e:
        logger.error(f"Error classifying conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/all")
async def get_all_focuses():
    """
    Get all focuses.
    
    Returns:
    {
        "focuses": {...},
        "metadata": {...}
    }
    """
    try:
        return focus_service.get_all_focuses()
    except Exception as e:
        logger.error(f"Error getting focuses: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{focus_id}")
async def get_focus(focus_id: str):
    """
    Get a specific focus by ID.
    
    Args:
        focus_id: Focus ID (e.g., "F001" or "F001-1")
    
    Returns:
    {
        "focus": {...},
        "type": "focus" or "sub-focus"
    }
    """
    try:
        result = focus_service.search_focus(focus_id)
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Focus {focus_id} not found"
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting focus: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search/keyword/{keyword}")
async def search_by_keyword(keyword: str):
    """
    Search focuses by keyword.
    
    Args:
        keyword: Keyword to search for
    
    Returns:
    {
        "matches": [
            {
                "focus_id": "F001",
                "summary": "...",
                "keywords": [...],
                "type": "focus" or "sub-focus"
            }
        ]
    }
    """
    try:
        focuses = focus_service.get_all_focuses()
        matches = []
        
        for focus_id, focus in focuses["focuses"].items():
            # Search in focus keywords
            if any(keyword.lower() in kw.lower() for kw in focus["keywords"]):
                matches.append({
                    "focus_id": focus_id,
                    "summary": focus["summary"],
                    "keywords": focus["keywords"],
                    "conversation_count": focus["conversation_count"],
                    "type": "focus"
                })
            
            # Search in sub-focus keywords
            for sub_id, sub_focus in focus["sub_focuses"].items():
                if any(keyword.lower() in kw.lower() for kw in sub_focus["keywords"]):
                    matches.append({
                        "focus_id": sub_id,
                        "summary": f"{focus['summary']} > {sub_focus['summary']}",
                        "keywords": sub_focus["keywords"],
                        "conversation_count": sub_focus["conversation_count"],
                        "type": "sub-focus"
                    })
        
        return {"matches": matches, "keyword": keyword, "count": len(matches)}
        
    except Exception as e:
        logger.error(f"Error searching by keyword: {e}")
        raise HTTPException(status_code=500, detail=str(e))