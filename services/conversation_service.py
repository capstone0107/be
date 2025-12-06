import logging
import uuid 
from typing import Dict, Any, List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from models.conversation_orm import Conversation, Message, Focus, conversation_focus

logger = logging.getLogger(__name__)

class ConversationService:
    
    def __init__(self):
        """
        초기화 시점에 FocusService를 내부에서 가져옵니다.
        이렇게 하면 순환 참조(Circular Import) 문제를 피할 수 있습니다.
        """
        # 여기서 import를 수행하여 인스턴스를 연결합니다.
        from services.focus_service import focus_service
        self.focus_service = focus_service
    
    # ==========================================
    # PHASE 1: 대화 시작 (Conversation 생성)
    # ==========================================
    
    def create_conversation(
        self,
        conversation_id: str,
        db: Session,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        try:
            # 기존 대화 확인
            existing = db.query(Conversation).filter(
                Conversation.id == conversation_id
            ).first()
            
            if existing:
                logger.info(f"Conversation {conversation_id} already exists")
                return {
                    "status": "exists",
                    "conversation_id": conversation_id,
                    "is_saved": existing.is_saved
                }
            
            # 새 대화 생성
            conversation = Conversation(
                id=conversation_id,
                user_id=user_id,
                is_saved=0  # 임시 상태
            )
            
            db.add(conversation)
            db.commit()
            
            logger.info(f"✨ Created new conversation: {conversation_id}")
            
            return {
                "status": "created",
                "conversation_id": conversation_id,
                "timestamp": conversation.created_at.isoformat() if hasattr(conversation, 'created_at') else None
            }
            
        except IntegrityError as e:
            db.rollback()
            logger.error(f"Failed to create conversation: {e}")
            return {
                "status": "error",
                "error": "DUPLICATE_ID",
                "message": "대화 ID가 이미 존재합니다"
            }
        except Exception as e:
            db.rollback()
            logger.error(f"Unexpected error: {e}")
            return {
                "status": "error",
                "error": "CREATE_FAILED",
                "message": str(e)
            }
    
    # ==========================================
    # PHASE 2: 메시지 저장 (실시간)
    # ==========================================
        
    def save_message(
            self,
            conversation_id: str,
            role: str,
            content: str,
            db: Session,
            sources: Optional[List[Dict[str, str]]] = None
        ) -> Dict[str, Any]:
        try:
            # Conversation 존재 확인
            conversation = db.query(Conversation).filter(
                Conversation.id == conversation_id
            ).first()
            
            if not conversation:
                return {
                    "status": "error",
                    "error": "CONVERSATION_NOT_FOUND",
                    "message": f"대화 {conversation_id}를 찾을 수 없습니다"
                }
            
            # 현재 메시지 개수 확인 (순서 결정을 위해 필요)
            message_count = db.query(Message).filter(
                Message.conversation_id == conversation_id
            ).count()
            
            # [수정된 부분] -------------------------------------------------------
            # 기존: message_id = f"msg-{message_count + 1}"  <-- 중복 원인
            # 변경: UUID를 사용하여 전역적으로 유일한 ID 생성
            message_id = str(uuid.uuid4()) 
            # -------------------------------------------------------------------
            
            # 메시지 생성
            message = Message(
                id=message_id,
                conversation_id=conversation_id,
                role=role,
                content=content,
                sources=sources if role == "assistant" else None,
                message_order=message_count # 순서는 여전히 0, 1, 2... 로 유지됨
            )
            
            db.add(message)
            db.commit()
            db.refresh(message)
            
            logger.info(
                f"💬 Saved {role} message: {message_id} "
                f"(conversation: {conversation_id})"
            )
            
            return {
                "status": "saved",
                "message_id": message_id,
                "message_order": message_count,
                "role": role,
                "conversation_id": conversation_id
            }
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to save message: {e}")
            return {
                "status": "error",
                "error": "SAVE_FAILED",
                "message": str(e)
            }
    def save_message_pair(
            self,
            conversation_id: str,
            user_message: str,
            assistant_message: str,
            db: Session,
            sources: Optional[List[Dict[str, str]]] = None
        ) -> Dict[str, Any]:
        # User 메시지 저장
        user_result = self.save_message(
            conversation_id, "user", user_message, db=db
        )
        
        if user_result["status"] != "saved":
            return user_result
        
        # Assistant 메시지 저장
        assistant_result = self.save_message(
            conversation_id, "assistant", assistant_message, db=db, sources=sources
        )
        
        result = {
            "status": "saved",
            "user_message_id": user_result["message_id"],
            "assistant_message_id": assistant_result["message_id"],
            "conversation_id": conversation_id
        }
        
        # message_order가 있는 경우에만 추가
        if "message_order" in assistant_result:
            result["assistant_message_order"] = assistant_result["message_order"]
        
        return result
    # ==========================================
    # PHASE 3: 사용자 저장 (Focus 분류)
    # ==========================================


    def finalize_conversation(
        self,
        conversation_id: str,
        db: Session,
        user_id: Optional[int] = None,  # ⭐ user_id 파라미터 추가
        user_title: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        사용자가 저장 버튼을 눌렀을 때 실행
        
        Args:
            conversation_id: 대화 ID
            db: DB 세션
            user_id: 현재 로그인한 사용자 ID ⭐
            user_title: 사용자 지정 제목 (Optional)
        """
        try:
            from models.conversation_orm import Conversation, Message

            # 1. Conversation 존재 및 상태 확인
            conversation = db.query(Conversation).filter(
                Conversation.id == conversation_id
            ).first()
            
            if not conversation:
                return {
                    "status": "error",
                    "error": "NOT_FOUND",
                    "message": f"대화 {conversation_id}를 찾을 수 없습니다"
                }
            
            # ⭐ 권한 체크: 해당 대화가 현재 사용자의 것인지 확인
            if user_id and conversation.user_id and conversation.user_id != user_id:
                return {
                    "status": "error",
                    "error": "FORBIDDEN",
                    "message": "다른 사용자의 대화는 저장할 수 없습니다"
                }
            
            if conversation.is_saved == 1:
                return {
                    "status": "already_saved",
                    "conversation_id": conversation_id,
                    "message": "이미 저장된 대화입니다"
                }
            
            # 메시지 개수 확인
            msg_count = db.query(Message).filter(Message.conversation_id == conversation_id).count()
            if msg_count < 2:
                 return {
                    "status": "error",
                    "error": "INSUFFICIENT_MESSAGES",
                    "message": "메시지가 충분하지 않습니다 (최소 2개 필요)"
                }

            # 2. FocusService에게 분석 및 저장 위임 (⭐ user_id 전달)
            logger.info(f"🔍 Delegating analysis for {conversation_id} to FocusService (user_id: {user_id})...")
            
            result = self.focus_service.analyze_and_save_focus(
                conversation_id=conversation_id,
                db=db,
                user_id=user_id  # ⭐ user_id 전달
            )
            
            if result["status"] == "error":
                return {
                    "status": "error", 
                    "error": "CLASSIFICATION_FAILED", 
                    "message": result.get("message")
                }
            
            # 3. 사용자 지정 제목이 있다면 덮어쓰기
            if user_title:
                conversation.title = user_title
                db.commit()
                result["title"] = user_title
            else:
                result["title"] = conversation.title

            # 반환 포맷
            return {
                "status": "finalized",
                "conversation_id": conversation_id,
                "title": result.get("title"), 
                "summary": result.get("summary"),
                "message_count": msg_count,
                "focus_count": len(result.get("focuses", [])),
                "focuses": result.get("focuses", [])
            }
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to finalize conversation: {e}")
            return {
                "status": "error",
                "error": "FINALIZE_FAILED",
                "message": str(e)
            }  

    # ==========================================
    # 유틸리티 메서드
    # ==========================================
    
    def get_conversation_status(
        self,
        conversation_id: str,
        db: Session
    ) -> Dict[str, Any]:
        """대화 상태 조회"""
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        
        if not conversation:
            return {
                "exists": False,
                "conversation_id": conversation_id
            }
        
        message_count = db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).count()
        
        return {
            "exists": True,
            "conversation_id": conversation_id,
            "is_saved": conversation.is_saved == 1,
            "title": conversation.title,
            "message_count": message_count,
            "focus_count": len(conversation.focuses),
            "created_at": conversation.created_at.isoformat()
        }

# 인스턴스 생성 (이제 인자 없이 생성 가능)
conversation_service = ConversationService()