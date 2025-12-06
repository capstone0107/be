import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import get_db
from services.conversation_service import conversation_service
from services.llm_service import llm_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["search-improved"])


# ==========================================
# Helper: 인증된 사용자 정보 가져오기
# ==========================================

def get_current_user_id(request: Request) -> Optional[int]:
    """
    AuthMiddleware에서 설정한 사용자 정보를 가져옵니다.
    
    Returns:
        user_id: 인증된 사용자 ID, 비인증 시 None
    """
    if hasattr(request.state, 'is_authenticated') and request.state.is_authenticated:
        return getattr(request.state, 'user_id', None)
    return None


# ==========================================
# Request/Response Models (동일)
# ==========================================

class StartConversationRequest(BaseModel):
    conversation_id: str
    user_id: Optional[int] = None  # Deprecated: 이제 토큰에서 가져옴

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
# PHASE 1: 대화 시작 (⭐ 인증된 사용자 ID 사용)
# ==========================================

@router.post("/start", response_model=StartConversationResponse)
async def start_conversation(
    request_data: StartConversationRequest,
    request: Request,  # ⭐ Request 객체 추가
    db: Session = Depends(get_db)
):
    """
    대화 시작 - Conversation 레코드 생성
    
    ⭐ 변경사항: AuthMiddleware에서 설정한 user_id를 자동으로 사용
    """
    try:
        # ⭐ 인증된 사용자 ID 가져오기 (토큰 기반)
        user_id = get_current_user_id(request)
        
        logger.info(f"Starting conversation {request_data.conversation_id} for user {user_id}")
        
        result = conversation_service.create_conversation(
            conversation_id=request_data.conversation_id,
            user_id=user_id,  # ⭐ 토큰에서 가져온 user_id 사용
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
# PHASE 2: 질문 및 메시지 저장 (변경 없음)
# ==========================================

@router.post("/query", response_model=QueryResponse)
async def query_and_save(
    request_data: QueryRequest,
    db: Session = Depends(get_db)
):
    """질문하고 응답을 즉시 DB에 저장"""
    try:
        logger.info(f"=== Query Request Started ===")
        logger.info(f"Conversation ID: {request_data.conversation_id}")
        logger.info(f"Question: {request_data.question[:100]}...")
        
        # LLM 호출
        logger.info(f"[Step 1] Calling LLM service...")
        try:
            llm_result = llm_service.generate_text(request_data.question)
            logger.info(f"[Step 1] LLM response received successfully")
        except Exception as e:
            logger.error(f"[Step 1] LLM service error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"LLM 호출 실패: {str(e)}")
        
        # Sources 준비
        logger.info(f"[Step 2] Preparing sources...")
        sources = [
            {
                "title": s.title,
                "url": s.url,
                "snippet": s.snippet
            }
            for s in llm_result.sources
        ]
        
        # 메시지 쌍 저장
        logger.info(f"[Step 3] Saving message pair to DB...")
        save_result = conversation_service.save_message_pair(
            conversation_id=request_data.conversation_id,
            user_message=request_data.question,
            assistant_message=llm_result.answer,
            sources=sources,
            db=db
        )
        
        if save_result["status"] != "saved":
            raise HTTPException(status_code=500, detail=save_result.get('message', 'Unknown error'))
        
        response = QueryResponse(
            conversation_id=request_data.conversation_id,
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
            message_order=save_result.get("assistant_message_order", 0)
        )
        
        logger.info(f"=== Query Request Completed Successfully ===")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"=== Unexpected error in query_and_save ===", exc_info=True)
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")


# ==========================================
# PHASE 3: 사용자 저장 (⭐ user_id 전달)
# ==========================================

@router.post("/finalize", response_model=FinalizeResponse)
async def finalize_conversation(
    request_data: FinalizeRequest,
    request: Request,  # ⭐ Request 객체 추가
    db: Session = Depends(get_db)
):
    """
    사용자가 '저장' 버튼을 누를 때 호출
    
    ⭐ 변경사항: 인증된 사용자 ID를 Focus 생성 시 전달
    """
    try:
        # ⭐ 인증된 사용자 ID 가져오기
        user_id = get_current_user_id(request)
        
        logger.info(
            f"Finalizing conversation {request_data.conversation_id} "
            f"(user_id: {user_id}, title: {request_data.user_title})"
        )
        
        # ⭐ user_id를 finalize_conversation에 전달
        result = conversation_service.finalize_conversation(
            conversation_id=request_data.conversation_id,
            db=db,
            user_id=user_id,  # ⭐ 추가
            user_title=request_data.user_title
        )
        
        if result["status"] == "error":
            error_code = result.get("error")
            
            if error_code == "NOT_FOUND":
                raise HTTPException(status_code=404, detail=result["message"])
            elif error_code == "FORBIDDEN":  # ⭐ 새로운 에러 타입
                raise HTTPException(status_code=403, detail=result["message"])
            elif error_code == "INSUFFICIENT_MESSAGES":
                raise HTTPException(status_code=400, detail=result["message"])
            elif error_code == "CLASSIFICATION_FAILED":
                raise HTTPException(status_code=500, detail=result["message"])
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
# 유틸리티 엔드포인트 (동일)
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
    request: Request,  # ⭐ 권한 체크를 위해 추가
    db: Session = Depends(get_db)
):
    """
    전체 대화 조회 (메시지 + Focus 포함)
    
    ⭐ 권한 체크: 자신의 대화만 조회 가능
    """
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
        
        # ⭐ 권한 체크
        user_id = get_current_user_id(request)
        if user_id and conversation.user_id and conversation.user_id != user_id:
            raise HTTPException(
                status_code=403,
                detail="다른 사용자의 대화는 조회할 수 없습니다"
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
                "user_id": conversation.user_id,  # ⭐ user_id 포함
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