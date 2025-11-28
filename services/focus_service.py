"""
Focus classification service for conversation analysis.
Simplified version aligned with frontend requirements.

FIXED: Foreign key constraint error by adding db.flush() before inserting relationships
"""
import json
import os
import logging
from typing import Dict, Any, List
from datetime import datetime
from openai import OpenAI
from sqlalchemy.orm import Session

from models.conversation_orm import Conversation, Message, Focus, conversation_focus
from database import SessionLocal

logger = logging.getLogger(__name__)


CLASSIFICATION_PROMPT = """
당신은 대화 로그를 분석하여 주제별 Focus로 분류하는 AI 어시스턴트입니다.

[입력 정보]
대화 ID: {conversation_id}
현재 시각: {current_timestamp}

대화 메시지:
{conversation_text}

[작업 지시]
1. 대화 내용을 분석하여 의미적으로 연관된 메시지들을 그룹화하세요.
2. 각 그룹에 대해 Focus를 생성하세요.
3. 각 Focus는 2-5개의 메시지를 포함해야 합니다.
4. 전체 대화를 2-5개의 Focus로 분류하는 것을 목표로 하세요.

[Focus 생성 규칙]
- Focus 이름은 10-30자의 간결하고 명확한 제목으로 작성
- questionTags는 해당 Focus의 핵심 키워드 2-3개
- messageIds는 입력된 메시지의 순서 인덱스 (0부터 시작)

[메시지 ID 매핑]
입력된 메시지의 순서를 기반으로 ID를 생성하세요:
- 첫 번째 메시지 (index 0) → "msg-1"
- 두 번째 메시지 (index 1) → "msg-2"
- 세 번째 메시지 (index 2) → "msg-3"
... 이런 식으로

[출력 형식]
반드시 순수한 JSON만 출력하세요. 마크다운 코드 블록을 사용하지 마세요.

{{
  "conversation_summary": "대화 전체를 한 문장으로 요약",
  "focuses": [
    {{
      "id": "focus-unique-id-1",
      "name": "첫 번째 Focus 이름",
      "messageIds": ["msg-1", "msg-2", "msg-3"],
      "questionTags": ["키워드1", "키워드2"]
    }},
    {{
      "id": "focus-unique-id-2",
      "name": "두 번째 Focus 이름",
      "messageIds": ["msg-4", "msg-5"],
      "questionTags": ["키워드3", "키워드4"]
    }}
  ],
  "focus_assignments": [
    {{
      "focus_id": "focus-unique-id-1",
      "confidence": 0.95,
      "reason": "첫 번째 Focus에 할당한 이유"
    }},
    {{
      "focus_id": "focus-unique-id-2",
      "confidence": 0.90,
      "reason": "두 번째 Focus에 할당한 이유"
    }}
  ]
}}

[주의사항]
- Focus ID는 영문 소문자와 하이픈만 사용 (예: "focus-cpu-scheduling")
- messageIds는 반드시 "msg-1", "msg-2" 형식
- 모든 메시지가 최소 하나의 Focus에 포함되어야 함
- 대화가 2-3개 메시지만 있으면 1개의 Focus로 충분

[전체 대화록]
{conversation_text}
"""


class FocusClassificationService:
    """Service for classifying conversations into focus topics."""
    
    def __init__(self):
        """Initialize focus classification service."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set. Classification service will not be available.")
            self.client = None
        else:
            try:
                self.client = OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized for focus classification")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None
    
    def _format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """Format conversation messages to numbered text."""
        formatted = []
        for i, msg in enumerate(messages):
            role = "사용자" if msg.get("role") == "user" else "AI 어시스턴트"
            content = msg.get("content", "")
            formatted.append(f"[{i}] {role}: {content}")
        return "\n\n".join(formatted)
    
    def _extract_json_from_response(self, content: str) -> Dict[str, Any]:
        """Extract and parse JSON from AI response."""
        # Remove markdown code blocks
        content_cleaned = content.strip()
        if content_cleaned.startswith("```json"):
            content_cleaned = content_cleaned[7:]
        elif content_cleaned.startswith("```"):
            content_cleaned = content_cleaned[3:]
        if content_cleaned.endswith("```"):
            content_cleaned = content_cleaned[:-3]
        content_cleaned = content_cleaned.strip()
        
        return json.loads(content_cleaned)
    
    def classify_conversation(
        self,
        conversation_id: str,
        messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Classify a conversation into focus topics.
        
        Args:
            conversation_id: Unique conversation ID
            messages: List of messages [{"role": "user/assistant", "content": "..."}]
            
        Returns:
            Classification result with focuses and assignments
        """
        if not self.client:
            return {
                "error": "CLASSIFICATION_FAILED",
                "message": "분류 서비스를 사용할 수 없습니다. OpenAI API 키를 확인하세요."
            }
        
        # Validate input
        if not conversation_id:
            return {
                "error": "INVALID_CONVERSATION_ID",
                "message": "대화 ID가 제공되지 않았습니다."
            }
        
        if not messages or len(messages) < 2:
            return {
                "error": "INSUFFICIENT_MESSAGES",
                "message": "메시지 수가 부족합니다 (최소 2개 필요)",
                "details": f"현재 메시지 수: {len(messages) if messages else 0}"
            }
        
        try:
            # Format conversation
            conversation_text = self._format_conversation(messages)
            
            # Create prompt
            prompt = CLASSIFICATION_PROMPT.format(
                conversation_id=conversation_id,
                current_timestamp=datetime.now().isoformat(),
                conversation_text=conversation_text
            )
            
            logger.info(f"Classifying conversation {conversation_id} with {len(messages)} messages")
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.3,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            logger.info(f"Received classification response (length: {len(content)})")
            
            # Parse JSON
            classification = self._extract_json_from_response(content)
            
            # Validate result structure
            if "focuses" not in classification:
                classification["focuses"] = []
            if "focus_assignments" not in classification:
                classification["focus_assignments"] = []
            if "conversation_summary" not in classification:
                classification["conversation_summary"] = "대화 요약"
            
            # Build response
            result = {
                "conversation_id": conversation_id,
                "conversation_summary": classification["conversation_summary"],
                "classified_at": datetime.now().isoformat(),
                "focuses": classification["focuses"],
                "focus_assignments": classification["focus_assignments"]
            }
            
            logger.info(f"Successfully classified conversation with {len(result['focuses'])} focuses")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.error(f"Response content: {content if 'content' in locals() else 'N/A'}")
            return {
                "error": "CLASSIFICATION_FAILED",
                "message": "응답 파싱에 실패했습니다",
                "details": str(e)
            }
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return {
                "error": "LLM_ERROR",
                "message": "분류 중 오류가 발생했습니다",
                "details": str(e)
            }
    
    def save_conversation_with_focuses(
        self,
        conversation_id: str,
        title: str,
        messages: List[Dict[str, Any]],
        classification_result: Dict[str, Any],
        db: Session
    ) -> Conversation:
        """
        Save conversation, messages, and focuses to database.
        
        FIXED: Added db.flush() before inserting conversation_focus relationships
        to ensure parent records exist first.
        
        Args:
            conversation_id: Conversation ID
            title: Conversation title
            messages: List of messages
            classification_result: Result from classify_conversation
            db: Database session
            
        Returns:
            Created Conversation object
        """
        try:
            # 1. Create Conversation
            conversation = Conversation(
                id=conversation_id,
                title=title,
                summary=classification_result.get("conversation_summary"),
                timestamp=datetime.now()
            )
            db.add(conversation)
            
            # 2. Create Messages
            for i, msg in enumerate(messages):
                message = Message(
                    id=f"msg-{i+1}",  # msg-1, msg-2, ...
                    conversation_id=conversation_id,
                    role=msg.get("role"),
                    content=msg.get("content"),
                    sources=msg.get("sources"),  # JSON field
                    message_order=i
                )
                db.add(message)
            
            # 3. Create Focuses
            focuses = classification_result.get("focuses", [])
            focus_assignments = classification_result.get("focus_assignments", [])
            
            # Create assignment lookup
            assignment_map = {
                assignment["focus_id"]: assignment
                for assignment in focus_assignments
            }
            
            for focus_data in focuses:
                focus = Focus(
                    id=focus_data["id"],
                    name=focus_data["name"],
                    message_ids=focus_data["messageIds"],  # JSON field
                    question_tags=focus_data["questionTags"]  # JSON field
                )
                db.add(focus)
            
            # ⭐ CRITICAL FIX: Flush to insert Conversation and Focus into DB first
            # This ensures parent records exist before creating foreign key relationships
            logger.info(f"Flushing session to ensure parent records exist...")
            db.flush()
            logger.info(f"Flush completed, now creating relationships...")
            
            # 4. Now create conversation_focus relationships (after flush)
            for focus_data in focuses:
                assignment = assignment_map.get(focus_data["id"], {})
                confidence = assignment.get("confidence", 1.0)
                reason = assignment.get("reason")
                
                # Insert into association table
                stmt = conversation_focus.insert().values(
                    conversation_id=conversation_id,
                    focus_id=focus_data["id"],
                    confidence=confidence,
                    reason=reason
                )
                db.execute(stmt)
            
            # 5. Commit transaction
            db.commit()
            db.refresh(conversation)
            
            logger.info(f"✅ Successfully saved conversation {conversation_id} with {len(messages)} messages and {len(focuses)} focuses")
            return conversation
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error saving conversation: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if the classification service is available."""
        return self.client is not None


# Global service instance
focus_service = FocusClassificationService()