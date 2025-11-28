import logging
import time
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from services import llm_service
from services.background_save_service import background_save_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["search"])


# Request/Response Models
class ChatMessage(BaseModel):
    """Single chat message"""
    role: str
    content: str
    sources: Optional[List[Dict[str, str]]] = None


class SearchRequest(BaseModel):
    """Search request"""
    question: List[str]
    conversation_id: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None
    auto_save: bool = True
    save_threshold: int = 2


class SearchResponse(BaseModel):
    """Search response"""
    answer: str
    sources: List[Dict[str, str]]
    conversation_id: str
    metadata: Optional[Dict[str, Any]] = None


def _generate_conversation_id() -> str:
    """Generate unique conversation ID"""
    return f"auto-{int(time.time() * 1000)}"


def _build_message_list(
    request: SearchRequest,
    response_answer: str,
    response_sources: List[Dict[str, str]]
) -> List[Dict[str, Any]]:
    """Build complete message list including current response"""
    messages = []
    
    if request.messages:
        # Use provided full message context
        for msg in request.messages:
            messages.append({
                "role": msg.role,
                "content": msg.content,
                "sources": msg.sources
            })
    else:
        # Reconstruct from question array
        for i, text in enumerate(request.question):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({
                "role": role,
                "content": text,
                "sources": [] if role == "user" else None
            })
    
    # Add current response
    messages.append({
        "role": "assistant",
        "content": response_answer,
        "sources": response_sources
    })
    
    return messages


@router.post("/search", response_model=SearchResponse)
async def search_query(request: SearchRequest, background_tasks: BackgroundTasks):
    """
    Query with automatic conversation ID management.
    
    Simple usage (recommended):
    {
        "question": ["current question"]
    }
    
    With conversation_id (for continuing):
    {
        "question": ["current question"],
        "conversation_id": "auto-1732950000000"
    }
    
    Backend will:
    1. Auto-generate conversation_id (first message)
    2. Return conversation_id in response
    3. Frontend uses returned ID for subsequent messages
    4. Auto-save when threshold reached
    """
    try:
        # Extract current question
        question_text = request.question[-1] if request.question else ""
        
        if not question_text.strip():
            raise HTTPException(status_code=400, detail="질문이 비어있습니다.")
        
        logger.info(f"Processing query: {question_text[:50]}...")
        # message 전체 확인 logger
        logger.info(f"Full message context: {request.messages}")
        # ===== CONVERSATION ID MANAGEMENT =====
        conversation_id = request.conversation_id
        is_new_conversation = False
        
        if not conversation_id:
            conversation_id = _generate_conversation_id()
            is_new_conversation = True
            logger.info(f"✨ New conversation created: {conversation_id}")
        else:
            logger.info(f"📝 Continuing conversation: {conversation_id}")
        
        # ===== GENERATE AI RESPONSE =====
        result = llm_service.generate_text(question_text)
        
        response_data = {
            "answer": result.answer,
            "sources": [
                {
                    "title": source.title,
                    "url": source.url,
                    "snippet": source.snippet
                }
                for source in result.sources
            ],
            "conversation_id": conversation_id
        }
        
        # ===== AUTO-SAVE LOGIC =====
        if request.auto_save:
            messages = _build_message_list(request, result.answer, response_data["sources"])
            
            # Q&A pairs = total messages / 2
            qa_pairs = len(messages) // 2
            should_save = qa_pairs >= request.save_threshold
            
            if should_save:
                # Queue background save
                background_tasks.add_task(
                    background_save_service.save_conversation_async,
                    conversation_id=conversation_id,
                    messages=messages,
                    auto_title=None
                )
                
                logger.info(
                    f"✅ Queued {conversation_id} for background save "
                    f"({len(messages)} messages, {qa_pairs} Q&A pairs)"
                )
                
                response_data["metadata"] = {
                    "auto_saved": True,
                    "message_count": len(messages),
                    "qa_pairs": qa_pairs,
                    "save_status": "queued",
                    "is_new_conversation": is_new_conversation
                }
            else:
                response_data["metadata"] = {
                    "auto_saved": False,
                    "message_count": len(messages),
                    "qa_pairs": qa_pairs,
                    "reason": f"Waiting for {request.save_threshold} Q&A pairs (current: {qa_pairs})",
                    "is_new_conversation": is_new_conversation
                }
        else:
            # Auto-save disabled
            response_data["metadata"] = {
                "auto_saved": False,
                "reason": "Auto-save disabled by request"
            }
        
        return SearchResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in search query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversation/{conversation_id}/status")
async def get_conversation_save_status(conversation_id: str):
    """
    Check if a conversation has been saved to database.
    
    Response:
    {
        "saved": true/false,
        "conversation_id": "...",
        "title": "...",
        "message_count": 10,
        "focus_count": 3,
        "saved_at": "2024-11-29T12:00:00"
    }
    """
    try:
        from database import SessionLocal
        from models.conversation_orm import Conversation
        
        db = SessionLocal()
        try:
            conversation = db.query(Conversation).filter(
                Conversation.id == conversation_id
            ).first()
            
            if conversation:
                return {
                    "saved": True,
                    "conversation_id": conversation.id,
                    "title": conversation.title,
                    "message_count": len(conversation.messages),
                    "focus_count": len(conversation.focuses),
                    "saved_at": conversation.created_at.isoformat()
                }
            else:
                return {
                    "saved": False,
                    "conversation_id": conversation_id,
                    "status": "pending",
                    "message": "백그라운드 저장 처리 중이거나 아직 저장되지 않았습니다."
                }
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error checking save status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """
    Get full conversation with focuses from database.
    
    Response:
    {
        "conversation": {
            "id": "...",
            "title": "...",
            "messages": [...],
            "focuses": [...]
        }
    }
    """
    try:
        from database import SessionLocal
        from models.conversation_orm import Conversation
        
        db = SessionLocal()
        try:
            conversation = db.query(Conversation).filter(
                Conversation.id == conversation_id
            ).first()
            
            if not conversation:
                raise HTTPException(
                    status_code=404,
                    detail=f"대화 {conversation_id}를 찾을 수 없습니다."
                )
            
            # Build messages
            messages = []
            for msg in sorted(conversation.messages, key=lambda m: m.message_order):
                messages.append({
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "sources": msg.sources
                })
            
            # Build focuses
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
                    "timestamp": conversation.timestamp.isoformat(),
                    "messages": messages,
                    "focuses": focuses
                }
            }
            
        finally:
            db.close()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))