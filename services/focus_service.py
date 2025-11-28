"""
Focus classification service.
"""
import json
import os
import logging
from typing import Dict, Any, List
from datetime import datetime
from openai import OpenAI

logger = logging.getLogger(__name__)


CLASSIFICATION_PROMPT = """
당신은 대화 로그를 분석하여 구조화된 데이터로 변환하는 'Conversation Structuring Engine'입니다.

[입력 정보]
아래 데이터는 ID가 포함된 JSON 형식의 대화 로그입니다:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{conversation_json}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[지시 사항]
제공된 대화를 분석하여 아래 규칙에 따라 JSON 객체를 생성하십시오.

Rule 0: **ID 무결성 유지 (가장 중요)**
   - 입력된 각 메시지의 `id` 값을 절대 변경하거나 새로 생성하지 마십시오.
   - `focuses` 배열의 `messageIds`에는 반드시 **[입력 정보]에 존재하는 `id` 값**만 사용해야 합니다.

Rule 1: **메타데이터 생성**
   - 대화 전체를 관통하는 적절한 `title`을 생성하십시오.
   - 출력의 `id`는 입력받은 conversation_id를 사용하십시오.

Rule 2: **메시지 구조화**
   - 입력된 메시지 목록을 그대로 `messages` 필드에 포함하되, 답변에 URL/참조가 있다면 `sources` 필드를 추출하여 추가하십시오.

Rule 3: **Focus (주제) 클러스터링**
   - 대화의 흐름을 분석하여 밀접하게 관련된 Q&A 세트들을 하나의 `focus`로 묶으십시오.
   - `messageIds`: 해당 주제에 속하는 메시지들의 `id` 리스트.
   - `questionTags`: 검색을 위한 핵심 키워드 2~3개.

[출력 포맷]
반드시 아래 JSON 스키마를 엄격히 따르십시오. (주석 제외)

{{
  "id": "{conversation_id}", 
  "title": "대화 요약 제목",
  "timestamp": "{current_timestamp}",
  "messages": [
    {{
      "id": "입력받은_msg_id_그대로_사용",
      "role": "user",
      "content": "..."
    }},
    {{
      "id": "입력받은_msg_id_그대로_사용",
      "role": "assistant",
      "content": "...",
      "sources": [ 
        {{ "title": "출처 제목", "url": "URL", "snippet": "발췌문" }}
      ]
    }}
  ],
  "focuses": [
    {{
      "id": "focus-generated-uuid-1",
      "name": "주제 그룹 명칭",
      "messageIds": ["입력받은_msg_id_1", "입력받은_msg_id_2"],
      "questionTags": ["태그1", "태그2"]
    }}
  ]
}}
"""

class FocusService:
    """Service for classifying conversations into focus topics."""
    
    def __init__(self, storage_path: str = "data/focus_db.json"):
        """Initialize focus service."""
        self.storage_path = storage_path
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.focuses = self._load_focuses()
    
    def _load_focuses(self) -> Dict[str, Any]:
        """Load focuses from storage."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load focuses: {e}")
        
        return {
            "focuses": {},
            "metadata": {
                "total_focuses": 0,
                "total_sub_focuses": 0,
                "last_id": "F000"
            }
        }
    
    def _save_focuses(self):
        """Save focuses to storage."""
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.focuses, f, ensure_ascii=False, indent=2)
                print("Focuses saved successfully.")
                print(self.focuses)
        except Exception as e:
            logger.error(f"Failed to save focuses: {e}")
    
    def _format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """Format conversation messages to text."""
        text = []
        for msg in messages:
            role = "사용자" if msg.get("role") == "user" else "AI"
            text.append(f"{role}: {msg.get('content', '')}")
        return "\n".join(text)
    
    def _format_focuses(self) -> str:
        """Format existing focuses for prompt."""
        if not self.focuses["focuses"]:
            return "현재 Focus가 없습니다."
        
        text = []
        for focus_id, focus in self.focuses["focuses"].items():
            keywords = ", ".join(focus.get("keywords", [])[:5])
            text.append(f"{focus_id}: {focus['summary']} [{keywords}]")
            
            for sub_focus in focus.get("sub_focuses", {}).values():
                sub_kw = ", ".join(sub_focus.get("keywords", [])[:3])
                text.append(f"  {sub_focus['id']}: {sub_focus['summary']} [{sub_kw}]")
        
        return "\n".join(text)
    

    def classify_conversation(
            self,
            conversation_id: str,
            messages: List[Dict[str, str]]  # 반드시 [{'id': 'uuid', 'role': '...', 'content': '...'}] 형태여야 함
        ) -> Dict[str, Any]:
            """
            대화를 분석하여 프론트엔드용 구조화된 JSON(Focus 그룹 포함)으로 변환합니다.
            """
            
            # 1. 메시지를 ID와 함께 포맷팅 (중요!)
            # LLM이 ID를 인식할 수 있도록 포맷팅해야 함
            formatted_messages = json.dumps([
                {"id": msg["id"], "role": msg["role"], "content": msg["content"]} 
                for msg in messages
            ], ensure_ascii=False, indent=2)

            # 2. 프롬프트 포맷팅 (새로운 구조화 프롬프트 사용)
            # existing_focuses는 내부 구조화에는 불필요할 수 있어 제거하거나 참고용으로만 둡니다.
            prompt = CLASSIFICATION_PROMPT.format(
                conversation_text=formatted_messages,
                current_timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"Structuring conversation {conversation_id}")
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"} # JSON 모드 강제 (모델 지원 시)
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON
            try:
                classification = json.loads(result_text)
            except json.JSONDecodeError:
                # 마크다운 백틱 제거 로직 (필요시 유지)
                if result_text.startswith("```json"):
                    result_text = result_text[7:-3]
                classification = json.loads(result_text)
            
            # 3. 결과 반환 (프론트엔드 구조에 맞춤)
            # self._apply_classification(...) 로직은 이 구조에 맞게 변경되어야 함
            
            return {
                "id": conversation_id,
                "title": classification.get("title", "무제 대화"),
                "conversation_summary": classification.get("title"), # 요약으로 제목 사용
                "focuses": classification.get("focuses", []), # 여기가 핵심: messageIds가 포함된 그룹
                "messages": classification.get("messages", []), # Source 등이 추가된 메시지 목록
                "classified_at": datetime.now().isoformat()
            }

    def _apply_classification(
        self,
        conversation_id: str,
        classification: Dict[str, Any]
    ):
        """Apply classification results to focus DB."""
        next_id_num = int(self.focuses["metadata"]["last_id"][1:]) + 1
        
        # Create new focuses
        for new_focus in classification["new_focuses"]:
            if new_focus["type"] == "focus":
                focus_id = f"F{next_id_num:03d}"
                next_id_num += 1
                
                self.focuses["focuses"][focus_id] = {
                    "id": focus_id,
                    "summary": new_focus["summary"],
                    "keywords": new_focus["keywords"],
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "conversation_count": 1,
                    "conversation_ids": [conversation_id],
                    "sub_focuses": {}
                }
                
                classification["focus_assignments"].append({
                    "focus_id": focus_id,
                    "confidence": 0.95,
                    "reason": new_focus["reason"]
                })
            
            elif new_focus["type"] == "sub-focus":
                parent_id = new_focus["parent_id"]
                if parent_id in self.focuses["focuses"]:
                    sub_count = len(self.focuses["focuses"][parent_id]["sub_focuses"])
                    sub_focus_id = f"{parent_id}-{sub_count + 1}"
                    
                    self.focuses["focuses"][parent_id]["sub_focuses"][sub_focus_id] = {
                        "id": sub_focus_id,
                        "summary": new_focus["summary"],
                        "keywords": new_focus["keywords"],
                        "context": new_focus["initial_context"],
                        "conversation_count": 1,
                        "conversation_ids": [conversation_id]
                    }
                    
                    classification["focus_assignments"].append({
                        "focus_id": sub_focus_id,
                        "confidence": 0.9,
                        "reason": new_focus["reason"]
                    })
        
        # Update existing focuses
        for update in classification["updates"]:
            focus_id = update["focus_id"]
            
            if "-" in focus_id:  # Sub-focus
                parent_id = focus_id.split("-")[0]
                if parent_id in self.focuses["focuses"]:
                    parent = self.focuses["focuses"][parent_id]
                    if focus_id in parent["sub_focuses"]:
                        sub = parent["sub_focuses"][focus_id]
                        
                        if "new_keywords" in update:
                            existing = set(sub.get("keywords", []))
                            existing.update(update["new_keywords"])
                            sub["keywords"] = list(existing)
                        
                        if "updated_context" in update and update["updated_context"]:
                            sub["context"] = update["updated_context"]
                        
                        if conversation_id not in sub["conversation_ids"]:
                            sub["conversation_ids"].append(conversation_id)
                            sub["conversation_count"] += 1
            else:  # Focus
                if focus_id in self.focuses["focuses"]:
                    focus = self.focuses["focuses"][focus_id]
                    
                    if "new_keywords" in update:
                        existing = set(focus.get("keywords", []))
                        existing.update(update["new_keywords"])
                        focus["keywords"] = list(existing)
                    
                    if conversation_id not in focus["conversation_ids"]:
                        focus["conversation_ids"].append(conversation_id)
                        focus["conversation_count"] += 1
        
        # Add conversation references
        for assignment in classification["focus_assignments"]:
            focus_id = assignment["focus_id"]
            
            if "-" in focus_id:  # Sub-focus
                parent_id = focus_id.split("-")[0]
                if parent_id in self.focuses["focuses"]:
                    parent = self.focuses["focuses"][parent_id]
                    if conversation_id not in parent["conversation_ids"]:
                        parent["conversation_ids"].append(conversation_id)
                        parent["conversation_count"] += 1
            else:  # Focus
                if focus_id in self.focuses["focuses"]:
                    focus = self.focuses["focuses"][focus_id]
                    if conversation_id not in focus["conversation_ids"]:
                        focus["conversation_ids"].append(conversation_id)
                        focus["conversation_count"] += 1
        
        # Update metadata
        self.focuses["metadata"]["last_id"] = f"F{next_id_num - 1:03d}"
        self.focuses["metadata"]["total_focuses"] = len(self.focuses["focuses"])
        self.focuses["metadata"]["total_sub_focuses"] = sum(
            len(f["sub_focuses"]) for f in self.focuses["focuses"].values()
        )
        
        # Save
        self._save_focuses()
    
    def get_all_focuses(self) -> Dict[str, Any]:
        """Get all focuses."""
        return self.focuses
    
    def search_focus(self, focus_id: str) -> Dict[str, Any]:
        """Search for a specific focus."""
        if "-" in focus_id:  # Sub-focus
            parent_id = focus_id.split("-")[0]
            if parent_id in self.focuses["focuses"]:
                parent = self.focuses["focuses"][parent_id]
                if focus_id in parent["sub_focuses"]:
                    return {
                        "focus": parent["sub_focuses"][focus_id],
                        "parent": parent,
                        "type": "sub-focus"
                    }
        else:  # Focus
            if focus_id in self.focuses["focuses"]:
                return {
                    "focus": self.focuses["focuses"][focus_id],
                    "type": "focus"
                }
        
        return None


# Global service instance
focus_service = FocusService()