import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import get_db
# 서비스를 전역으로 import 합니다.
from services.conversation_service import conversation_service
from services.llm_service import llm_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/search", tags=["search-improved"])


# ==========================================
# Request/Response Models
# ==========================================

class StartConversationRequest(BaseModel):
    conversation_id: str
    user_id: Optional[int] = None

class StartConversationResponse(BaseModel):
    status: str
    conversation_id: str
    timestamp: Optional[str] = None

class QueryRequest(BaseModel):
    conversation_id: str
    question: str

class QueryResponse(BaseModel):
    conversation_id: str
    message_id: str
    answer: str
    sources: List[Dict[str, str]]
    message_order: int

class FinalizeRequest(BaseModel):
    conversation_id: str
    user_title: Optional[str] = None

class FocusInfo(BaseModel):
    id: str
    name: str
    messageIds: List[str]
    questionTags: List[str]

class FinalizeResponse(BaseModel):
    status: str
    conversation_id: str
    title: str
    summary: Optional[str] = None
    message_count: int
    focus_count: int
    focuses: List[FocusInfo]


# ==========================================
# PHASE 1: 대화 시작
# ==========================================

@router.post("/start", response_model=StartConversationResponse)
async def start_conversation(
    request: StartConversationRequest,
    db: Session = Depends(get_db)
):
    """대화 시작 - Conversation 레코드 생성"""
    try:
        logger.info(f"Starting conversation {request.conversation_id} for user {request.user_id}")
        # 전역 인스턴스 사용
        result = conversation_service.create_conversation(
            conversation_id=request.conversation_id,
            user_id=request.user_id,
            db=db
        )
        
        if result["status"] == "error":
            raise HTTPException(
                status_code=400,
                detail=result.get("message")
            )
        
        return StartConversationResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# PHASE 2: 질문 및 메시지 저장
# ==========================================

@router.post("/query", response_model=QueryResponse)
async def query_and_save(
    request: QueryRequest,
    db: Session = Depends(get_db)
):
    """질문하고 응답을 즉시 DB에 저장"""
    try:
        # 1. LLM 호출
        logger.info(f"Calling LLM for question: {request.question[:50]}...")
        llm_result = llm_service.generate_text(request.question)
        
        # 2. 메시지 쌍 저장 (전역 서비스 사용)
        logger.info(f"Saving message pair to DB...")
        save_result = conversation_service.save_message_pair(
            conversation_id=request.conversation_id,
            user_message=request.question,
            assistant_message=llm_result.answer,
            sources=[
                {
                    "title": s.title,
                    "url": s.url,
                    "snippet": s.snippet
                }
                for s in llm_result.sources
            ],
            db=db
        )
        
        if save_result["status"] != "saved":
            raise HTTPException(
                status_code=500,
                detail=f"메시지 저장 실패: {save_result.get('message')}"
            )
        
        # 3. 응답 구성
        response = QueryResponse(
            conversation_id=request.conversation_id,
            message_id=save_result["assistant_message_id"],
            answer=llm_result.answer,
            sources=[
                {
                    "title": s.title,
                    "url": s.url,
                    "snippet": s.snippet or ""
                }
                for s in llm_result.sources
            ],
            message_order=save_result["assistant_message_order"]
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in query_and_save: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# PHASE 3: 사용자 저장 (Focus 분류)
# ==========================================

@router.post("/finalize", response_model=FinalizeResponse)
async def finalize_conversation(
    request: FinalizeRequest,
    db: Session = Depends(get_db)
):
    """사용자가 '저장' 버튼을 누를 때 호출"""
    try:
        logger.info(
            f"Finalizing conversation {request.conversation_id} "
            f"(title: {request.user_title})"
        )
        
        # 전역 서비스 사용
        result = conversation_service.finalize_conversation(
            conversation_id=request.conversation_id,
            db=db,
            user_title=request.user_title
        )
        
        if result["status"] == "error":
            error_code = result.get("error")
            
            if error_code == "NOT_FOUND":
                raise HTTPException(status_code=404, detail=result["message"])
            elif error_code == "INSUFFICIENT_MESSAGES":
                raise HTTPException(status_code=400, detail=result["message"])
            elif error_code == "CLASSIFICATION_FAILED":
                raise HTTPException(
                    status_code=500,
                    detail=f"{result['message']}: {result.get('details')}"
                )
            else:
                raise HTTPException(status_code=500, detail=result["message"])
        
        if result["status"] == "already_saved":
            raise HTTPException(status_code=400, detail=result["message"])
        
        # Convert focuses to Pydantic models
        focuses = [FocusInfo(**f) for f in result["focuses"]]
        
        return FinalizeResponse(
            status=result["status"],
            conversation_id=result["conversation_id"],
            title=result["title"],
            summary=result.get("summary"),
            message_count=result["message_count"],
            focus_count=result["focus_count"],
            focuses=focuses
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finalizing conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 유틸리티 엔드포인트
# ==========================================

@router.get("/status/{conversation_id}")
async def get_conversation_status(
    conversation_id: str,
    db: Session = Depends(get_db)
):
    """대화 상태 조회"""
    try:
        status = conversation_service.get_conversation_status(conversation_id, db)
        return status
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversation/{conversation_id}")
async def get_full_conversation(
    conversation_id: str,
    db: Session = Depends(get_db)
):
    """전체 대화 조회 (메시지 + Focus 포함)"""
    from models.conversation_orm import Conversation
    
    try:
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        
        if not conversation:
            raise HTTPException(
                status_code=404,
                detail=f"대화 {conversation_id}를 찾을 수 없습니다"
            )
        
        # Build response
        messages = []
        for msg in sorted(conversation.messages, key=lambda m: m.message_order):
            messages.append({
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "sources": msg.sources,
                "message_order": msg.message_order
            })
        
        focuses = []
        for focus in conversation.focuses:
            focuses.append({
                "id": focus.id,
                "name": focus.name,
                "messageIds": focus.message_ids,
                "questionTags": focus.question_tags
            })
        
        return {
            "conversation": {
                "id": conversation.id,
                "title": conversation.title,
                "summary": conversation.summary,
                "is_saved": conversation.is_saved == 1,
                "timestamp": conversation.created_at.isoformat() if hasattr(conversation, 'created_at') else None,
                "messages": messages,
                "focuses": focuses
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))