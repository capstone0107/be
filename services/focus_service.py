import json
import os
import logging
from typing import Dict, Any, List
from datetime import datetime
from openai import OpenAI
from sqlalchemy.orm import Session

# 모델 임포트 (경로는 프로젝트 구조에 맞게 조정하세요)
from models.conversation_orm import Conversation, Message, Focus, conversation_focus

logger = logging.getLogger(__name__)

# 프롬프트 수정: 복잡한 ID 대신 '메시지 번호(Index)'를 사용하도록 변경
CLASSIFICATION_PROMPT = """
당신은 대화 로그를 분석하여 주제별 Focus로 분류하는 AI 어시스턴트입니다.

[입력 정보]
대화 ID: {conversation_id}
현재 시각: {current_timestamp}

[대화 내용]
{conversation_text}

[작업 지시]
1. 대화 내용을 분석하여 의미적으로 연관된 메시지들을 그룹화하세요.
2. 각 그룹에 대해 Focus(주제)를 생성하세요.
3. 각 Focus는 2-5개의 메시지를 포함해야 합니다.
4. "message_indexes"에는 해당 메시지의 [번호]를 정수 리스트로 넣으세요. (ID가 아닙니다!)

[출력 형식]
반드시 순수한 JSON만 출력하세요.

{{
  "conversation_summary": "대화 전체 요약 (한 문장)",
  "focuses": [
    {{
      "name": "Focus 주제 제목 (10-30자)",
      "message_indexes": [0, 1, 2],
      "questionTags": ["태그1", "태그2"]
    }},
    {{
      "name": "두 번째 주제",
      "message_indexes": [3, 4],
      "questionTags": ["태그3"]
    }}
  ],
  "focus_assignments": [
    {{
      "focus_index": 0,
      "confidence": 0.95,
      "reason": "첫 번째 주제 선정 이유"
    }},
    {{
      "focus_index": 1,
      "confidence": 0.90,
      "reason": "두 번째 주제 선정 이유"
    }}
  ]
}}

[주의사항]
- "message_indexes"는 반드시 입력된 대화의 [번호] 숫자여야 합니다. (예: 0, 1, 2)
- 모든 메시지가 최소 하나의 Focus에 포함되도록 하세요.
"""

class FocusClassificationService:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set.")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)

    def _format_messages_for_prompt(self, messages: List[Message]) -> str:
        """메시지 객체 리스트를 LLM 프롬프트용 텍스트로 변환 (인덱스 부여)"""
        formatted = []
        for i, msg in enumerate(messages):
            # role이 user면 '사용자', assistant면 'AI' 등
            role_str = "사용자" if msg.role == "user" else "AI"
            # [0] 사용자: 안녕하세요 형태
            formatted.append(f"[{i}] {role_str}: {msg.content}")
        return "\n".join(formatted)

    def _extract_json(self, content: str) -> Dict[str, Any]:
        """마크다운 코드 블록 제거 및 JSON 파싱"""
        cleaned = content.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return json.loads(cleaned.strip())

    def analyze_and_save_focus(self, conversation_id: str, db: Session) -> Dict[str, Any]:
        """
        [핵심 메서드]
        DB에서 메시지를 가져와 분석 후, Focus 결과를 다시 DB에 저장합니다.
        
        Args:
            conversation_id: 대상 대화 ID
            db: DB 세션
        """
        if not self.client:
            return {"status": "error", "message": "OpenAI Client not available"}

        try:
            # 1. DB에서 메시지 가져오기 (순서대로 정렬 필수)
            messages = db.query(Message).filter(
                Message.conversation_id == conversation_id
            ).order_by(Message.message_order).all()

            if not messages:
                return {"status": "error", "message": "No messages found for this conversation"}

            # 2. 인덱스 매핑 생성 (Index -> Real DB Message ID)
            # LLM은 0, 1, 2로 답하고, 우리는 이걸 실제 ID로 바꿉니다.
            index_to_id_map = {i: msg.id for i, msg in enumerate(messages)}

            # 3. 프롬프트용 텍스트 생성
            conversation_text = self._format_messages_for_prompt(messages)
            
            # 4. LLM 요청
            prompt = CLASSIFICATION_PROMPT.format(
                conversation_id=conversation_id,
                current_timestamp=datetime.now().isoformat(),
                conversation_text=conversation_text
            )

            logger.info(f"Analyzing conversation {conversation_id} with {len(messages)} messages...")
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.3
            )
            
            result_json = self._extract_json(response.choices[0].message.content)
            
            # 5. DB 저장 로직 (업데이트)
            # 5-1. Conversation 테이블 업데이트 (요약, 제목 등)
            conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
            if conversation:
                conversation.summary = result_json.get("conversation_summary")
                # 제목이 없으면 요약 내용을 제목으로 사용하거나 첫 Focus 이름 사용
                if not conversation.title:
                    conversation.title = result_json.get("conversation_summary")[:50]
                conversation.is_saved = 1  # 저장 완료 상태로 변경

            # 5-2. Focus 및 관계 저장
            focuses_data = result_json.get("focuses", [])
            assignments_data = result_json.get("focus_assignments", [])
            
            # Focus 결과를 반환하기 위해 저장해둘 리스트
            saved_focus_list = []

            for idx, focus_item in enumerate(focuses_data):
                # A. Focus ID 생성 (Unique하게)
                focus_db_id = f"focus-{conversation_id}-{idx+1}"
                
                # B. 인덱스를 실제 Message ID로 변환 ⭐ (가장 중요한 부분)
                real_message_ids = []
                for msg_idx in focus_item.get("message_indexes", []):
                    if msg_idx in index_to_id_map:
                        real_message_ids.append(index_to_id_map[msg_idx])
                
                # C. Focus 객체 생성
                new_focus = Focus(
                    id=focus_db_id,
                    name=focus_item["name"],
                    message_ids=real_message_ids,   # JSON 타입 컬럼에 실제 ID 리스트 저장
                    question_tags=focus_item.get("questionTags", [])
                )
                db.add(new_focus)
                
                # D. 관계 테이블 (Conversation <-> Focus) 준비
                # assignments_data에서 현재 순서(idx)에 맞는 정보 찾기 (혹은 단순 매칭)
                assignment_info = next((a for a in assignments_data if a.get("focus_index") == idx), {})
                
                # Flush하여 Focus ID가 DB에 인식되게 함
                db.flush() 

                # E. 관계 연결
                stmt = conversation_focus.insert().values(
                    conversation_id=conversation_id,
                    focus_id=focus_db_id,
                    confidence=assignment_info.get("confidence", 1.0),
                    reason=assignment_info.get("reason", "")
                )
                db.execute(stmt)

                # 반환용 데이터 구성
                saved_focus_list.append({
                    "id": focus_db_id,
                    "name": focus_item["name"],
                    "messageIds": real_message_ids,
                    "questionTags": focus_item.get("questionTags", [])
                })

            db.commit()
            logger.info(f"✅ Successfully analyzed and saved focuses for {conversation_id}")

            return {
                "status": "success",
                "conversation_id": conversation_id,
                "summary": conversation.summary,
                "focuses": saved_focus_list
            }

        except Exception as e:
            db.rollback()
            logger.error(f"Error in analyze_and_save_focus: {e}")
            return {"status": "error", "message": str(e)}

# 전역 인스턴스
focus_service = FocusClassificationService()