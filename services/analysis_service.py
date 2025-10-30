"""
Conversation analysis service using OpenAI.
"""
import json
import re
import os
import logging
from typing import Dict, Any, List
from openai import OpenAI

logger = logging.getLogger(__name__)

# 핵심 프롬프트 템플릿
PROMPT_TEMPLATE = """
당신은 사용자의 학습 대화 내용을 분석하고, 관련 정보를 검증하는 'AI 학습 코치'이자 '정보 연구원'입니다.
당신은 신뢰할 수 있는 정보를 찾기 위해 'search(query: str)' 툴을 사용할 수 있습니다.

대화가 모두 종료되었습니다. 이제 주어진 '전체 대화록'({document})을 분석하여 다음 작업을 수행하세요.

[수행 작업]
1.  [대화 분석]: '전체 대화록'({document})을 분석하여 '주요 주제(Topic)'들을 식별합니다.
2.  [메타인지 분석]: '각 주제'별로 대화록 '내부' 문맥을 분석하여 '메타인지 카드 ID'와 '내부 검색 키워드'를 결정합니다.
3.  [외부 정보 기반 심화 인사이트 생성]:
	a.  URL 내의 내용이 신뢰도가 매우 높고 실제로 확인 가능해야 함을 유의해야 합니다.(위키피디아 뿐 아니라 다른 공신력 있는 기술 사이트도 포함)
	b.  'search' 툴을 사용하여 '각 주제'에 대한 신뢰할 수 있는 외부 자료 1개를 찾습니다.
	c.  해당 자료(및 당신의 내장 지식)를 바탕으로, '단순 요약'이 아닌 **'심화 학습 정보(Insight)'**를 생성합니다.
	d.  이 '심화 학습 정보'는 아래 [심화 학습 정보 가이드라인]의 예시처럼, 사용자가 미처 몰랐을 법한 **맥락, 트레이드오프, 예외 상황**을 짚어줘야 합니다.
	e.  해당 자료의 '출처 URL'을 함께 제공합니다. 
	f.  (⭐ 추가됨) 생성된 '심화 학습 정보(Insight)'를 바탕으로, 사용자의 추가적인 탐색을 유도할 수 있는 **'연관 질문(follow_up_questions)'** 2개(최대 3개)를 생성합니다.

[메타인지 카드 선택 로직] (4가지 카드 대상)
* `CARD_TRADE_OFF` ⚖️: (주제: 비교) 대화가 두 대상의 장단점, '희생' 관계를 주로 다루었을 때.
* `CARD_CONTEXT` 💡: (주제: 비교 또는 개념) 대화가 해당 개념의 '등장 배경', '역사', '해결 문제'를 다루었을 때.
* `CARD_PRECONDITION` 🎯: (주제: 개념) 대화가 개념 성립의 '필수 조건'(e.g., 데드락 4대 조건)을 핵심으로 다루었을 때.
* `CARD_EDGE_CASE` 🐛: (주제: 개념) 대화가 개념의 일반적인 정의/원리를 설명하고, 위 3가지에 해당하지 않을 때.

[외부 정보 탐색 규칙]
* 'search' 툴을 각 주제당 1회 이상 사용하여 신뢰할 수 있는 출처를 찾으세요.
* 단순 블로그 포스트보다는 공식 문서, 위키피디아, 공신력 있는 기술 사이트를 우선하세요.
* 'summary'는 검색된 외부 자료의 내용을 요약해야 하며, 대화록의 내용을 반복해선 안 됩니다.

[심화 학습 정보(Summary) 가이드라인]
'summary'는 다음과 같은 형태의 'nuanced insight'여야 합니다:
* (예시 1 - 우선순위 vs 라운드 로빈): "실시간 시스템에서는 라운드 로빈보다 우선순위 기반 스케줄링이 더 적합한 경우가 많습니다. 특히 데드라인이 엄격한 태스크가 있을 때 그렇습니다."
* (예시 2 - 라운드 로빈 오버헤드): "라운드 로빈은 대형 시스템에서는 효율적이지만, 마이크로 디바이스에서는 오히려 오버헤드가 크다는 연구가 있습니다. 특히 컨텍스트 스위칭 비용이 작업 실행 시간보다 클 수 있습니다."
* (예시 3 - FIFO와 임베디드): "일부 임베디드 시스템에서는 FIFO가 더 효율적일 수 있으며, 컨텍스트 스위칭 비용이 중요한 요소입니다. 실시간 제약이 있는 경우 우선순위 기반 스케줄링이 선호됩니다."
또한 'summary'는 사용자가 손쉽게 읽을 수 있는 200자 이내의 텍스트 여야 하며, 원문 보기를 클릭하고 싶어지는 형태여야 합니다.

[연관 질문(follow_up_questions) 가이드라인] (⭐ 추가됨)
* 'follow_up_questions'는 'summary'에서 제시된 심화 인사이트(맥락, 트레이드오프, 예외)를 사용자가 스스로 적용해 보거나 더 깊이 고민해 볼 수 있도록 유도해야 합니다.
* (예시): "그렇다면, I/O 작업이 많은 시스템에서는 어떤 스케줄링이 더 유리할까요?", "이 트레이드오프가 실시간 시스템이 아닌 웹 서버에서는 어떻게 다르게 적용될까요?"
---
[출력 형식]
반드시 순수한 JSON만 출력하세요. 마크다운 코드 블록(``` 또는 ```json)을 사용하지 마세요.

{{{{
	"overall_summary": "...",
	"metacognitive_insights": [
		{{{{
			"topic": "첫 번째로 식별된 학습 주제 (예: 데드락)",
			"card_id": "해당 주제에 매칭된 카드 ID (예: CARD_PRECONDITION)",
			"search_keywords": [
				"첫 번째 주제 관련 내부 키워드 1 (메타인지용)",
				"첫 번째 주제 관련 내부 키워드 2"
			]
		}}}},
		{{{{
			"topic": "두 번째로 식별된 학습 주제 (예: TCP vs UDP)",
			"card_id": "해당 주제에 매칭된 카드 ID (예: CARD_TRADE_OFF)",
			"search_keywords": [
		  _     "두 번째 주제 관련 내부 키워드 1 (메타인지용)",
				"두 번째 주제 관련 내부 키워드 2"
			]
		}}}}
	],
"external_verifications": [
		{{{{
_               "topic": "첫 번째 학습 주제 (예: 스케줄링)",
			"summary": "여기에 [심화 학습 정보 가이드라인]의 예시 1, 2, 3과 같은 스타일로 생성된 '심화 인사이트'를 작성.",
			"source": "https://example.com/scheduling_paper",
			"follow_up_questions": [
				"summary의 심화 인사이트와 연관된 첫 번째 추가 질문",
				"summary의 심화 인사이트와 연관된 두 번째 추가 질문"
			]
		}}}}
	]
}}}}
---

[전체 대화록]
{document}
"""
class ConversationAnalysisService:
    """Service for analyzing conversations using OpenAI."""
    
    def __init__(self):
        """Initialize the OpenAI client."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set. Analysis service will not be available.")
            self.client = None
        else:
            try:
                self.client = OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized for conversation analysis")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None
    
    def format_conversation(self, messages: List[Dict[str, Any]]) -> str:
        """
        Format conversation messages into a readable string.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Formatted conversation string
        """
        conversation = []
        for i, msg in enumerate(messages, 1):
            role = "사용자" if msg.get("role") == "user" else "AI 어시스턴트"
            content = msg.get("content", "")
            conversation.append(f"[{i}] {role}: {content}")
        
        return "\n\n".join(conversation)
    
    def extract_json_from_response(self, content: str) -> Dict[str, Any]:
        """
        Extract and parse JSON from AI response.
        
        Args:
            content: Raw response content from OpenAI
            
        Returns:
            Parsed JSON dictionary
        """
        # Remove markdown code blocks
        content_cleaned = re.sub(r'^```json?\s*\n?', '', content, flags=re.MULTILINE)
        content_cleaned = re.sub(r'\n?```\s*$', '', content_cleaned, flags=re.MULTILINE)
        
        # Find JSON object
        json_match = re.search(r'\{[\s\S]*\}', content_cleaned)
        
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
        else:
            raise ValueError("No JSON object found in response")

    def analyze_conversation(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze a conversation and return insights.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Analysis results dictionary
        """
        if not self.client:
            return {
                "error": "분석 서비스를 사용할 수 없습니다. OpenAI API 키를 확인하세요."
            }
        
        if not messages:
            return {
                "error": "분석할 대화가 없습니다."
            }
        
        try:
            # Format conversation
            document = self.format_conversation(messages)
            
            # Create prompt
            prompt = PROMPT_TEMPLATE.format(document=document)
            
            # Call OpenAI API
            logger.info("Calling OpenAI API for conversation analysis...")
            response = self.client.chat.completions.create(
                model="gpt-4o-mini-search-preview-2025-03-11",
                messages=[
                    {"role": "system", "content": prompt}
                ],
                max_tokens=2000
            )
            
            # 1. 전체 response 객체 확인
            print("=" * 50)
            print("전체 Response 객체:")
            print(response)
            print("=" * 50)
            
            # Extract response
            content = response.choices[0].message.content
            
            # 2. 추출된 content (raw text) 확인
            print("추출된 Content (Raw):")
            print(content)
            print("=" * 50)
            
            logger.info("Received response from OpenAI")
            
            # Parse JSON
            result = self.extract_json_from_response(content)
            
            # 3. JSON 파싱 후 결과 확인
            print("파싱된 JSON 결과:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print("=" * 50)
            
            # Add metadata
            result["analyzed_at"] = None  # Will be set by the caller
            result["message_count"] = len(messages)
            
            return result
            
        except json.JSONDecodeError as e:
            print("=" * 50)
            print("JSON 파싱 에러 발생!")
            print(f"에러: {e}")
            if 'content' in locals():
                print("문제가 된 Content:")
                print(content)
            print("=" * 50)
            
            logger.error(f"JSON parsing failed: {e}")
            return {
                "error": f"응답 파싱 실패: {str(e)}",
                "original_response": content if 'content' in locals() else None
            }
        except Exception as e:
            print("=" * 50)
            print("예상치 못한 에러 발생!")
            print(f"에러 타입: {type(e).__name__}")
            print(f"에러 메시지: {e}")
            if 'content' in locals():
                print("받은 Content (있는 경우):")
                print(content)
            if 'response' in locals():
                print("Response 객체 (있는 경우):")
                print(response)
            print("=" * 50)
            
            logger.error(f"Conversation analysis error: {e}")
            return {
                "error": f"대화 분석 중 오류 발생: {str(e)}"
            } 

    def is_available(self) -> bool:
        """Check if the analysis service is available."""
        return self.client is not None


# Global service instance
analysis_service = ConversationAnalysisService()

