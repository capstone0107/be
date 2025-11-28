"""
Focus router for conversation classification and retrieval.
Aligned with frontend API requirements.
"""
import logging
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import desc

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
async def get_all_focuses_grouped(
    limit: int = 20, 
    offset: int = 0, 
    db: Session = Depends(get_db)
):
    """
    저장된(is_saved=1) 모든 대화를 가져와서 Conversation ID 별로 Focus를 묶어서 반환합니다.
    
    Returns:
        {
            "conversations": {
                "auto-12345": {
                    "title": "CPU 스케줄링",
                    "timestamp": "...",
                    "focuses": [
                        {"id": "focus-1", "name": "FCFS 특징", ...},
                        {"id": "focus-2", "name": "Round Robin", ...}
                    ]
                },
                ...
            },
            "metadata": { ... }
        }
    """
    from models.conversation_orm import Conversation

    try:
        # 1. 쿼리 작성: 저장된 대화만 조회 + Focus 정보 함께 로드 (Eager Loading)
        # joinedload를 써야 DB 쿼리가 한 번만 나갑니다.
        query = db.query(Conversation)\
            .options(joinedload(Conversation.focuses))\
            .filter(Conversation.is_saved == 1)\
            .order_by(desc(Conversation.timestamp)) # 최신 대화 순

        # 2. 전체 개수 및 페이징
        total_conversations = query.count()
        conversations = query.offset(offset).limit(limit).all()

        # 3. 데이터 구조 변환 (Conversation ID를 Key로 그룹화)
        grouped_result = {}
        
        for conv in conversations:
            # 해당 대화에 속한 Focus들 정리
            focus_list = []
            for focus in conv.focuses:
                focus_list.append({
                    "id": focus.id,
                    "name": focus.name,
                    "questionTags": focus.question_tags if focus.question_tags else [],
                    # 필요시 message_ids 개수 등 추가 정보 포함
                    "messageCount": len(focus.message_ids) if focus.message_ids else 0
                })

            grouped_result[conv.id] = {
                "title": conv.title,
                "summary": conv.summary,
                "timestamp": conv.timestamp.isoformat() if conv.timestamp else None,
                "focus_count": len(focus_list),
                "focuses": focus_list  # 여기에 Focus 목록이 들어갑니다
            }

        return {
            "conversations": grouped_result,
            "metadata": {
                "total_conversations": total_conversations,
                "current_limit": limit,
                "current_offset": offset
            }
        }

    except Exception as e:
        logger.error(f"Error getting grouped focuses: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# GET /api/focus/{focus_id}
# ============================================
@router.get("/{focus_id}")
async def get_focus(focus_id: str, db: Session = Depends(get_db)):
    """
    특정 Focus의 상세 정보와 포함된 메시지 목록을 조회합니다.
    
    Args:
        focus_id: 조회할 Focus ID
        db: 데이터베이스 세션
        
    Returns:
        Focus 정보와 실제 메시지 객체 리스트
        
    Raises:
        404: 해당 ID의 Focus가 없을 경우
    """
    from models.conversation_orm import Focus, Message

    try:
        # 1. Focus 기본 정보 조회
        focus = db.query(Focus).filter(Focus.id == focus_id).first()
        
        if not focus:
            raise HTTPException(
                status_code=404,
                detail=f"Focus {focus_id}를 찾을 수 없습니다"
            )

        # 2. 포함된 메시지들의 실제 내용 조회
        # Focus.message_ids는 JSON 필드이므로 파이썬 리스트로 자동 변환됩니다. (예: ['msg-1', 'msg-2'])
        target_message_ids = focus.message_ids if focus.message_ids else []
        
        messages = []
        if target_message_ids:
            # ID 리스트에 포함된 메시지들을 한 번에 조회 (WHERE id IN (...))
            # 원래 대화 순서대로 정렬 (order_by message_order)
            messages = db.query(Message)\
                .filter(Message.id.in_(target_message_ids))\
                .order_by(Message.message_order)\
                .all()

        # 3. 응답 구성
        return {
            "focus": {
                "id": focus.id,
                "name": focus.name,
                "questionTags": focus.question_tags,
                # 프론트엔드에서 보여줄 실제 메시지 데이터
                "messages": [
                    {
                        "id": msg.id,
                        "role": msg.role,
                        "content": msg.content,
                        "message_order": msg.message_order,
                        "timestamp": msg.created_at.isoformat() if hasattr(msg, 'created_at') and msg.created_at else None
                    }
                    for msg in messages
                ]
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting focus detail: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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