import os
from typing import Dict, List, Optional
from openai import OpenAI
from pydantic import BaseModel
import logging
import json

logger = logging.getLogger(__name__)

class Source(BaseModel):
    """출처 정보 모델"""
    title: str
    url: str
    snippet: Optional[str] = None


class LLMResponse(BaseModel):
    """LLM 응답 모델"""
    answer: str
    sources: List[Source]


class LLMService:
    def __init__(self):
        """LLM 서비스 초기화"""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-search-preview")
        
    def generate_prompt(self, user_query: str) -> List[Dict]:
        """
        출처를 포함한 JSON 응답을 생성하도록 프롬프트 생성
        
        Args:
            user_query: 사용자 질문
            
        Returns:
            메시지 리스트
        """
        system_prompt = """
            당신은 정확한 정보를 제공하는 AI 어시스턴트입니다.

            응답 규칙:
            1. 사용자의 질문에 대해 정확하고 상세한 답변을 제공하세요.
            2. 답변 생성 시 사용한 모든 출처를 반드시 포함하세요.
            3. 응답은 반드시 순수한 JSON 형식으로만 제공해야 합니다.

            출처 우선순위 (높은 순서):
            1. 학술 논문 (arXiv, Google Scholar, PubMed, IEEE, Nature, Science 등)
            2. 공식 웹사이트 (정부 기관 .gov, 공식 조직, 기업 공식 사이트)
            3. 신뢰할 수 있는 뉴스 매체 (주요 언론사)
            4. 전문 기술 문서 (공식 문서, API 문서)
            5. 기타 웹사이트

            위 우선순위에 따라 가장 신뢰도 높은 출처를 먼저 제공하세요.
            논문이나 공식 사이트가 있다면 반드시 우선적으로 포함하세요.

            CRITICAL: 마크다운 코드 블록(```json 또는 ```)을 사용하지 마세요. 순수 JSON만 출력하세요.

            {
            "answer": "사용자 질문에 대한 상세한 답변",
            "sources": [
                {
                "title": "출처 제목",
                "url": "출처 URL",
                "snippet": "관련 내용 발췌 (선택사항)"
                }
            ]
            }

            중요:
            - 반드시 위 JSON 형식만 출력하세요
            - 마크다운 코드 블록을 절대 사용하지 마세요
            - JSON 외의 어떤 텍스트도 포함하지 마세요
            - sources는 신뢰도가 높은 순서대로 정렬하세요 (논문/공식사이트 → 뉴스 → 기타)
            - 답변 생성에 실제로 사용된 출처만 포함하세요
            - 검색 결과가 있다면 반드시 sources에 포함하세요
            - 검색 결과가 없으면 sources는 빈 배열로 제공하세요
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        return messages

    def generate_text(self, prompt: str) -> LLMResponse:
        """
        LLM을 사용하여 텍스트 생성 (출처 포함)
        
        Args:
            prompt: 사용자 프롬프트
            
        Returns:
            LLMResponse 객체 (답변 + 출처)
        """
        try:
            messages = self.generate_prompt(prompt)
            
            # OpenAI API 호출 - GPT-4o Search Preview용 설정
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            
            # 응답 파싱
            content = response.choices[0].message.content
            
            # 응답 로깅 (디버깅용)
            logger.info(f"GPT 원본 응답 (처음 200자): {content[:200] if content else 'None'}")
            
            # 빈 응답 체크
            if not content or not content.strip():
                logger.error("빈 응답 받음")
                return LLMResponse(
                    answer="응답이 비어있습니다.",
                    sources=[]
                )
            
            # 마크다운 코드 블록 제거
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
                logger.info("```json 제거")
            elif content.startswith("```"):
                content = content[3:]
                logger.info("``` 제거")
            if content.endswith("```"):
                content = content[:-3]
                logger.info("끝의 ``` 제거")
            content = content.strip()
            
            # JSON 파싱
            parsed_response = json.loads(content)
            
            # LLMResponse 객체로 변환
            sources = [
                Source(**source) for source in parsed_response.get("sources", [])
            ]
            
            return LLMResponse(
                answer=parsed_response.get("answer", ""),
                sources=sources
            )
            
        except json.JSONDecodeError as e:
            # JSON 파싱 실패 시 상세 로깅
            logger.error(f"JSON 파싱 오류: {e}")
            logger.error(f"파싱 실패한 내용 (전체): {content if 'content' in locals() else 'content 변수 없음'}")
            return LLMResponse(
                answer=f"응답 파싱 중 오류가 발생했습니다: {str(e)}",
                sources=[]
            )
        except Exception as e:
            # 기타 오류 처리
            logger.error(f"예상치 못한 오류: {e}")
            return LLMResponse(
                answer=f"오류가 발생했습니다: {str(e)}",
                sources=[]
            )


# Global service instance
llm_service = LLMService()