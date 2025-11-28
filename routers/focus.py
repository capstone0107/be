"""
Focus router for conversation classification and retrieval.
Aligned with frontend API requirements.
"""
import logging
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import get_db
from services.focus_service import focus_service
from models.conversation_orm import Conversation, Message, Focus

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/focus", tags=["focus"])


# Request/Response Models
class MessageRequest(BaseModel):
    """Message in a conversation"""
    role: str  # 'user' or 'assistant'
    content: str


class ClassifyConversationRequest(BaseModel):
    """Request to classify a conversation"""
    conversation_id: str
    messages: List[MessageRequest]


class FocusData(BaseModel):
    """Focus data structure"""
    id: str
    name: str
    messageIds: List[str]
    questionTags: List[str]


class FocusAssignment(BaseModel):
    """Focus assignment with confidence"""
    focus_id: str
    confidence: float
    reason: str


class ClassifyConversationResponse(BaseModel):
    """Response from conversation classification"""
    conversation_id: str
    conversation_summary: str
    classified_at: str
    focuses: List[FocusData]
    focus_assignments: List[FocusAssignment]


class SaveConversationRequest(BaseModel):
    """Request to save a conversation with focuses"""
    conversation_id: str
    title: str
    messages: List[Dict[str, Any]]  # Full message objects with sources
    classification_result: Dict[str, Any]  # Result from classify endpoint


# ============================================
# POST /api/focus/classify
# ============================================
@router.post("/classify", response_model=ClassifyConversationResponse)
async def classify_conversation(
    request: ClassifyConversationRequest
):
    """
    Classify a conversation into focus topics.
    
    This endpoint analyzes conversation messages and groups them into
    semantically related focus topics.
    
    Args:
        request: Conversation ID and messages
        
    Returns:
        Classification result with focuses and assignments
        
    Raises:
        503: Classification service not available
        400: Invalid input (insufficient messages, etc.)
        500: Classification error
    """
    try:
        # Check service availability
        if not focus_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="분류 서비스를 사용할 수 없습니다. OpenAI API 키를 확인하세요."
            )
        
        # Convert Pydantic models to dict
        messages = [msg.dict() for msg in request.messages]
        
        # Perform classification
        result = focus_service.classify_conversation(
            conversation_id=request.conversation_id,
            messages=messages
        )
        
        # Check for errors
        if "error" in result:
            error_code = result.get("error")
            message = result.get("message")
            details = result.get("details")
            
            if error_code == "INSUFFICIENT_MESSAGES":
                raise HTTPException(status_code=400, detail=f"{message}. {details}")
            elif error_code == "INVALID_CONVERSATION_ID":
                raise HTTPException(status_code=400, detail=message)
            else:
                raise HTTPException(status_code=500, detail=f"{message}. {details}")
        
        # Return successful result
        return ClassifyConversationResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in classify_conversation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"분류 중 예상치 못한 오류 발생: {str(e)}"
        )


# ============================================
# POST /api/focus/save
# ============================================
@router.post("/save")
async def save_conversation_with_focuses(
    request: SaveConversationRequest,
    db: Session = Depends(get_db)
):
    """
    Save a conversation with its classified focuses to database.
    
    This endpoint should be called after /classify to persist the results.
    
    Args:
        request: Conversation data and classification result
        db: Database session
        
    Returns:
        Success message with conversation ID
        
    Raises:
        500: Database save error
    """
    try:
        conversation = focus_service.save_conversation_with_focuses(
            conversation_id=request.conversation_id,
            title=request.title,
            messages=request.messages,
            classification_result=request.classification_result,
            db=db
        )
        
        return {
            "status": "success",
            "message": "대화가 성공적으로 저장되었습니다",
            "conversation_id": conversation.id,
            "focus_count": len(request.classification_result.get("focuses", []))
        }
        
    except Exception as e:
        logger.error(f"Error saving conversation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"대화 저장 중 오류 발생: {str(e)}"
        )


# ============================================
# GET /api/focus/all
# ============================================
@router.get("/all")
async def get_all_focuses(db: Session = Depends(get_db)):
    """
    Get all focuses from database.
    
    NOTE: This is a placeholder implementation for future use.
    Currently returns empty structure.
    
    Returns:
        Empty focus structure (to be implemented)
    """
    # TODO: Implement focus aggregation logic
    # For now, return empty structure as per frontend requirements
    return {
        "focuses": {},
        "metadata": {
            "total_focuses": 0,
            "total_sub_focuses": 0,
            "last_id": ""
        }
    }


# ============================================
# GET /api/focus/{focus_id}
# ============================================
@router.get("/{focus_id}")
async def get_focus(focus_id: str, db: Session = Depends(get_db)):
    """
    Get a specific focus by ID.
    
    NOTE: This is a placeholder implementation for future use.
    Currently returns 404.
    
    Args:
        focus_id: Focus ID
        db: Database session
        
    Returns:
        Focus data (to be implemented)
        
    Raises:
        404: Focus not found
    """
    # TODO: Implement focus retrieval logic
    # For now, return 404 as per frontend requirements
    raise HTTPException(
        status_code=404,
        detail=f"Focus {focus_id}를 찾을 수 없습니다 (구현 예정)"
    )


# ============================================
# GET /api/focus/search/keyword/{keyword}
# ============================================
@router.get("/search/keyword/{keyword}")
async def search_by_keyword(keyword: str, db: Session = Depends(get_db)):
    """
    Search focuses by keyword.
    
    NOTE: This is a placeholder implementation for future use.
    Currently returns empty results.
    
    Args:
        keyword: Search keyword
        db: Database session
        
    Returns:
        Empty search results (to be implemented)
    """
    # TODO: Implement keyword search logic
    # For now, return empty results as per frontend requirements
    return {
        "matches": [],
        "keyword": keyword,
        "count": 0
    }


# ============================================
# GET /api/focus/conversation/{conversation_id}
# ============================================
@router.get("/conversation/{conversation_id}")
async def get_conversation_focuses(
    conversation_id: str,
    db: Session = Depends(get_db)
):
    """
    Get focuses for a specific conversation.
    
    This endpoint retrieves the classification results for a saved conversation.
    
    Args:
        conversation_id: Conversation ID
        db: Database session
        
    Returns:
        Conversation with focuses
        
    Raises:
        404: Conversation not found
    """
    try:
        # Query conversation with focuses
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        
        if not conversation:
            raise HTTPException(
                status_code=404,
                detail=f"대화 {conversation_id}를 찾을 수 없습니다"
            )
        
        # Build focus data
        focuses = []
        for focus in conversation.focuses:
            focuses.append({
                "id": focus.id,
                "name": focus.name,
                "messageIds": focus.message_ids,
                "questionTags": focus.question_tags
            })
        
        return {
            "conversation_id": conversation.id,
            "title": conversation.title,
            "summary": conversation.summary,
            "timestamp": conversation.timestamp.isoformat(),
            "focuses": focuses
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving conversation focuses: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"대화 조회 중 오류 발생: {str(e)}"
        )