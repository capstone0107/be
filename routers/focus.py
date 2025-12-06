"""
Focus router for conversation classification and retrieval.
⭐ user_id 기반 필터링 추가
"""
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import desc

from database import get_db
from services.focus_service import focus_service
from models.conversation_orm import Conversation, Message, Focus

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/focus", tags=["focus"])


# ==========================================
# Helper: 인증된 사용자 정보 가져오기
# ==========================================

def get_current_user_id(request: Request) -> Optional[int]:
    """AuthMiddleware에서 설정한 사용자 정보를 가져옵니다."""
    if hasattr(request.state, 'is_authenticated') and request.state.is_authenticated:
        return getattr(request.state, 'user_id', None)
    return None


# Request/Response Models (동일)
class MessageRequest(BaseModel):
    role: str
    content: str

class ClassifyConversationRequest(BaseModel):
    conversation_id: str
    messages: List[MessageRequest]

class FocusData(BaseModel):
    id: str
    name: str
    messageIds: List[str]
    questionTags: List[str]

class FocusAssignment(BaseModel):
    focus_id: str
    confidence: float
    reason: str

class ClassifyConversationResponse(BaseModel):
    conversation_id: str
    conversation_summary: str
    classified_at: str
    focuses: List[FocusData]
    focus_assignments: List[FocusAssignment]

class SaveConversationRequest(BaseModel):
    conversation_id: str
    title: str
    messages: List[Dict[str, Any]]
    classification_result: Dict[str, Any]


# ============================================
# POST /api/focus/classify (변경 없음)
# ============================================
@router.post("/classify", response_model=ClassifyConversationResponse)
async def classify_conversation(request: ClassifyConversationRequest):
    """대화를 Focus로 분류"""
    try:
        if not focus_service.is_available():
            raise HTTPException(status_code=503, detail="분류 서비스를 사용할 수 없습니다.")
        
        messages = [msg.dict() for msg in request.messages]
        result = focus_service.classify_conversation(
            conversation_id=request.conversation_id,
            messages=messages
        )
        
        if "error" in result:
            error_code = result.get("error")
            message = result.get("message")
            
            if error_code == "INSUFFICIENT_MESSAGES":
                raise HTTPException(status_code=400, detail=message)
            else:
                raise HTTPException(status_code=500, detail=message)
        
        return ClassifyConversationResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in classify_conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# POST /api/focus/save (변경 없음)
# ============================================
@router.post("/save")
async def save_conversation_with_focuses(
    request: SaveConversationRequest,
    db: Session = Depends(get_db)
):
    """분류된 Focus를 DB에 저장"""
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
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# GET /api/focus/all (⭐ user_id 필터링 추가)
# ============================================
@router.get("/all")
async def get_all_focuses_grouped(
    request: Request,  # ⭐ Request 추가
    limit: int = 20, 
    offset: int = 0, 
    db: Session = Depends(get_db)
):
    """
    저장된(is_saved=1) 대화를 Conversation ID별로 Focus를 묶어서 반환
    
    ⭐ 변경사항: 인증된 사용자의 대화만 조회
    """
    try:
        # ⭐ 인증된 사용자 ID 가져오기
        user_id = get_current_user_id(request)
        
        # 1. 쿼리 작성
        query = db.query(Conversation)\
            .options(joinedload(Conversation.focuses))\
            .filter(Conversation.is_saved == 1)\
            .order_by(desc(Conversation.timestamp))
        
        # ⭐ user_id 필터링 추가
        if user_id:
            query = query.filter(Conversation.user_id == user_id)
            logger.info(f"Filtering conversations for user_id: {user_id}")
        else:
            logger.warning("No user_id found - returning all conversations")

        # 2. 전체 개수 및 페이징
        total_conversations = query.count()
        conversations = query.offset(offset).limit(limit).all()

        # 3. 데이터 구조 변환
        grouped_result = {}
        
        for conv in conversations:
            focus_list = []
            for focus in conv.focuses:
                focus_list.append({
                    "id": focus.id,
                    "name": focus.name,
                    "questionTags": focus.question_tags if focus.question_tags else [],
                    "messageCount": len(focus.message_ids) if focus.message_ids else 0
                })

            grouped_result[conv.id] = {
                "title": conv.title,
                "summary": conv.summary,
                "timestamp": conv.timestamp.isoformat() if conv.timestamp else None,
                "user_id": conv.user_id,  # ⭐ user_id 포함
                "focus_count": len(focus_list),
                "focuses": focus_list
            }

        return {
            "conversations": grouped_result,
            "metadata": {
                "total_conversations": total_conversations,
                "current_limit": limit,
                "current_offset": offset,
                "filtered_by_user": user_id is not None,  # ⭐ 필터링 여부
                "user_id": user_id  # ⭐ 현재 사용자 ID
            }
        }

    except Exception as e:
        logger.error(f"Error getting grouped focuses: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# GET /api/focus/{focus_id} (⭐ 권한 체크 추가)
# ============================================
@router.get("/{focus_id}")
async def get_focus(
    focus_id: str, 
    request: Request,  # ⭐ Request 추가
    db: Session = Depends(get_db)
):
    """
    특정 Focus의 상세 정보와 포함된 메시지 목록을 조회
    
    ⭐ 변경사항: 자신의 Focus만 조회 가능
    """
    try:
        # 1. Focus 기본 정보 조회
        focus = db.query(Focus).filter(Focus.id == focus_id).first()
        
        if not focus:
            raise HTTPException(
                status_code=404,
                detail=f"Focus {focus_id}를 찾을 수 없습니다"
            )
        
        # ⭐ 권한 체크
        user_id = get_current_user_id(request)
        if user_id and focus.user_id and focus.user_id != user_id:
            raise HTTPException(
                status_code=403,
                detail="다른 사용자의 Focus는 조회할 수 없습니다"
            )

        # 2. 포함된 메시지들의 실제 내용 조회
        target_message_ids = focus.message_ids if focus.message_ids else []
        
        messages = []
        if target_message_ids:
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
                "user_id": focus.user_id,  # ⭐ user_id 포함
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
# GET /api/focus/search/keyword/{keyword} (⭐ 필터링 추가)
# ============================================
@router.get("/search/keyword/{keyword}")
async def search_by_keyword(
    keyword: str, 
    request: Request,  # ⭐ Request 추가
    db: Session = Depends(get_db)
):
    """
    키워드로 Focus 검색
    
    ⭐ TODO: 구현 필요 (현재는 빈 결과 반환)
    ⭐ 구현 시 user_id 필터링 적용
    """
    user_id = get_current_user_id(request)
    
    # TODO: 실제 검색 로직 구현
    # 예시:
    # focuses = db.query(Focus)\
    #     .filter(Focus.name.contains(keyword))\
    #     .filter(Focus.user_id == user_id if user_id else True)\
    #     .all()
    
    return {
        "matches": [],
        "keyword": keyword,
        "count": 0,
        "user_id": user_id  # ⭐ 검색 범위 표시
    }


# ============================================
# GET /api/focus/conversation/{conversation_id} (⭐ 권한 체크 추가)
# ============================================
@router.get("/conversation/{conversation_id}")
async def get_conversation_focuses(
    conversation_id: str,
    request: Request,  # ⭐ Request 추가
    db: Session = Depends(get_db)
):
    """
    특정 대화의 Focus 조회
    
    ⭐ 변경사항: 자신의 대화만 조회 가능
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
        
        # ⭐ 권한 체크
        user_id = get_current_user_id(request)
        if user_id and conversation.user_id and conversation.user_id != user_id:
            raise HTTPException(
                status_code=403,
                detail="다른 사용자의 대화는 조회할 수 없습니다"
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
            "user_id": conversation.user_id,  # ⭐ user_id 포함
            "timestamp": conversation.timestamp.isoformat(),
            "focuses": focuses
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving conversation focuses: {e}")
        raise HTTPException(status_code=500, detail=str(e))