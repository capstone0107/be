import os
from typing import Dict, List, Optional
from openai import OpenAI
from pydantic import BaseModel
import json


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
        system_prompt = """당신은 정확한 정보를 제공하는 AI 어시스턴트입니다.

응답 규칙:
1. 사용자의 질문에 대해 정확하고 상세한 답변을 제공하세요.
2. 답변 생성 시 사용한 모든 출처를 반드시 포함하세요.
3. 응답은 반드시 다음과 같은 JSON 형식으로 제공해야 합니다:

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
- JSON 형식만 출력하고, 다른 텍스트는 포함하지 마세요.
- sources 배열에는 답변 생성에 실제로 사용된 출처만 포함하세요.
- 검색 결과가 있다면 반드시 sources에 포함하세요.
- 검색 결과가 없거나 일반 지식으로 답변하는 경우 sources는 빈 배열로 제공하세요."""

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
                temperature=0.7,
                response_format={"type": "json_object"},  # JSON 응답 강제
                web_search_options={
                    "search_context_size": "medium"  # low, medium, high 중 선택
                }
            )
            
            # 응답 파싱
            content = response.choices[0].message.content
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
            # JSON 파싱 실패 시 기본 응답
            return LLMResponse(
                answer=f"응답 파싱 중 오류가 발생했습니다: {str(e)}",
                sources=[]
            )
        except Exception as e:
            # 기타 오류 처리
            return LLMResponse(
                answer=f"오류가 발생했습니다: {str(e)}",
                sources=[]
            )


# Global service instance
llm_service = LLMService()