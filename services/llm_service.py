import os
from typing import Dict, List, Optional
from openai import OpenAI
from pydantic import BaseModel
import logging
import json
import re

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


class QuizData(BaseModel):
    """퀴즈 데이터 모델"""
    question: str
    options: List[str]
    correct_answer: int
    explanation: str

class DocumentData(BaseModel):
    """도큐먼트 데이터 모델"""
    title: str
    content: str

class LLMService:
    def __init__(self):
        """LLM 서비스 초기화"""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-search-preview")
    
    def generate_quiz(
    self,
    title: str,
    summary: str,
    source_url: str,
    user_question: str
) -> Optional[QuizData]:
        """
        북마크 정보를 기반으로 퀴즈 생성 (검색 모델 사용)
        """
        try:
            system_prompt = """
                당신은 학습 퀴즈를 생성하는 AI입니다.

                주어진 정보와 웹 검색을 통해 정확한 4지선다 퀴즈를 생성하세요.

                응답 규칙:
                1. 반드시 순수한 JSON 형식으로만 응답하세요.
                2. 마크다운 코드 블록(```json 또는 ```)을 사용하지 마세요.
                3. 문제는 주어진 내용을 이해했는지 확인할 수 있는 것이어야 합니다.
                4. 선택지는 4개여야 하며, 그럴듯한 오답을 포함해야 합니다.
                5. correct_answer는 정답의 인덱스입니다 (0, 1, 2, 3 중 하나).
                6. explanation은 왜 그것이 정답인지 설명합니다.

                응답 형식:
                {
                    "question": "퀴즈 문제",
                    "options": ["선택지1", "선택지2", "선택지3", "선택지4"],
                    "correct_answer": 0,
                    "explanation": "정답에 대한 해설"
                }
            """

            user_prompt = f"""
                다음 정보를 바탕으로 퀴즈를 생성해주세요.
                필요하다면 출처 URL을 검색해서 더 정확한 정보를 확인하세요.

                [사용자가 했던 질문]
                {user_question}

                [출처 제목]
                {title}

                [내용 요약]
                {summary}

                [출처 URL]
                {source_url}
            """

            response = self.client.chat.completions.create(
                model=self.model,  # gpt-4o-search-preview 사용
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
            )

            content = response.choices[0].message.content
            
            if not content or not content.strip():
                logger.error("퀴즈 생성: 빈 응답")
                return None

            # 마크다운 코드 블록 제거
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            parsed = json.loads(content)
            
            return QuizData(
                question=parsed["question"],
                options=parsed["options"],
                correct_answer=parsed["correct_answer"],
                explanation=parsed["explanation"]
            )

        except json.JSONDecodeError as e:
            logger.error(f"퀴즈 JSON 파싱 오류: {e}")
            return None
        except Exception as e:
            logger.error(f"퀴즈 생성 오류: {e}")
            return None
    
    def generate_document(
        self,
        title: str,
        summary: str,
        source_url: str,
        user_question: str
    ) -> Optional[DocumentData]:
        """
        북마크 정보를 기반으로 학습 도큐먼트 생성
        """
        try:
            system_prompt = """
    당신은 학습 자료를 생성하는 AI입니다.

    주어진 정보와 웹 검색을 통해 해당 주제에 대한 상세한 학습 문서를 작성하세요.

    응답 규칙:
    1. 반드시 순수한 JSON 형식으로만 응답하세요.
    2. 마크다운 코드 블록(```json 또는 ```)을 사용하지 마세요.
    3. title은 학습 문서의 제목입니다.
    4. content는 마크다운 형식의 상세한 학습 내용입니다.
    5. content에는 개념 설명, 예시, 주의사항 등을 포함하세요.
    6. 출처의 내용을 바탕으로 정확한 정보를 제공하세요.

    응답 형식:
    {
        "title": "학습 문서 제목",
        "content": "## 개요\\n\\n내용...\\n\\n### 세부 내용\\n\\n..."
    }
    """

            user_prompt = f"""
    다음 정보를 바탕으로 학습 문서를 생성해주세요.
    출처 URL을 검색해서 정확한 정보를 확인하고 작성하세요.

    [사용자가 했던 질문]
    {user_question}

    [출처 제목]
    {title}

    [내용 요약]
    {summary}

    [출처 URL]
    {source_url}
    """

            response = self.client.chat.completions.create(
                model=self.model,  # gpt-4o-search-preview
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
            )

            content = response.choices[0].message.content
            
            if not content or not content.strip():
                logger.error("도큐먼트 생성: 빈 응답")
                return None

            # 마크다운 코드 블록 제거
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            parsed = json.loads(content)
            
            return DocumentData(
                title=parsed["title"],
                content=parsed["content"]
            )

        except json.JSONDecodeError as e:
            logger.error(f"도큐먼트 JSON 파싱 오류: {e}")
            return None
        except Exception as e:
            logger.error(f"도큐먼트 생성 오류: {e}")
            return None
        
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
            - you must double-escape all backslashes
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
            
            parsed_response = {}
            # JSON 파싱 (LaTeX 역슬래시 자동 수정 로직 추가)
            try:
                # 1차 시도: 일반적인 파싱
                parsed_response = json.loads(content)
            except json.JSONDecodeError:
                # 2차 시도: LaTeX 역슬래시(\) 문제일 수 있으므로 정규식으로 수리 후 재시도
                logger.warning("JSON 파싱 실패. LaTeX 역슬래시 자동 수정 시도...")
                
                # 유효한 JSON 이스케이프 문자(", \, /, b, f, n, r, t, u)가 아닌 역슬래시만 찾아서 두 번 쓴 것으로 교체
                fixed_content = re.sub(r'\\(?![\\"/bfnrtu])', r'\\\\', content)
                
                # 수정된 내용으로 다시 파싱 (여기서 실패하면 진짜 오류임)
                parsed_response = json.loads(fixed_content)
                logger.info("LaTeX 역슬래시 수정 후 파싱 성공")
                
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