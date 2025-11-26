"""
Graph generation service using OpenAI.
Handles both single conversation graphs and combined graphs.
"""
import json
import re
import os
import logging
from typing import Dict, Any, List
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

logger = logging.getLogger(__name__)

# 1. Single Conversation Graph Prompt
GRAPH_PROMPT_TEMPLATE = """
당신은 사용자의 학습 대화 내용을 분석하고, 지식 그래프를 생성하는 'AI 학습 코치'입니다.
대화가 종료되었습니다. 주어진 '전체 대화록'({document})을 분석하여 다음 작업을 수행하세요.

[수행 작업]
1.  [대화 분석]: 대화의 흐름을 분석하여 주요 '학습 주제(Concepts)'와 그들 간의 '논리적 관계'를 식별합니다.
2.  [지식 그래프 생성]: 식별된 정보를 바탕으로 'graph' 객체를 생성합니다.
    a. **Nodes (주제)**:
       - id: 영문 식별자 (예: "NeuralNetwork")
       - label: 한글 표기 (예: "신경망")
       - description: **해당 개념을 1-2문장으로 명확하게 설명하는 요약문.** (사용자가 노드를 눌렀을 때 학습에 도움이 되어야 함)
       - related_message_indices: 해당 개념이 주로 논의된 메시지의 순번(0부터 시작) 리스트.
    b. **Edges (관계)**:
       - source: 시작 노드 ID
       - target: 끝 노드 ID
       - label: 관계의 성격. 다음 중 하나를 우선 사용:
         ['IS_A' (하위 개념), 'PART_OF' (구성 요소), 'LEADS_TO' (학습 흐름/전제), 'SOLVES_PROBLEM' (해결책), 'CAUSES' (원인), 'CONTRASTS_WITH' (비교/대조), 'RELATED_TO' (기타 관련)]

[출력 형식]
반드시 순수한 JSON만 출력하세요. 마크다운 코드 블록을 사용하지 마세요.

{{{{
    "graph": {{{{
        "nodes": [
            {{{{ "id": "Overfitting", "label": "과적합", "description": "모델이 훈련 데이터에 너무 익숙해져 새로운 데이터에 대한 예측 성능이 떨어지는 현상입니다.", "related_message_indices": [5, 6] }}}}
        ],
        "edges": [
            {{{{ "source": "Overfitting", "target": "Regularization", "label": "SOLVES_PROBLEM" }}}}
        ]
    }}}}
}}}}

[전체 대화록]
{document}
"""

# 2. Combined Graph "Meta-Analysis" Prompt
COMBINED_GRAPH_PROMPT = """
당신은 여러 개의 학습 대화 그래프를 하나로 통합하는 '지식 통합 전문가'입니다.
사용자가 서로 다른 시점에 학습한 내용들(Graph Summaries)을 입력받아, 하나의 연결된 '지식 맵'으로 병합하세요.

[입력 데이터]
당신에게는 여러 개의 JSON 그래프 데이터가 주어집니다. 각 그래프는 특정 'conversation_id'에 속해 있습니다.

[수행 작업]
1. **노드 병합 (De-duplication)**:
   - 서로 다른 그래프에서 동일한 개념(예: "Loss Function"과 "손실 함수")이 등장하면, 이를 하나의 노드로 통합하세요.
   - 통합된 노드는 가장 설명이 잘 된 description을 유지하거나 내용을 보완하세요.
   - **중요**: 통합된 노드는 `conversation_id`를 배열 형태로 모두 포함해야 합니다. (예: ["convo_1", "convo_2"])

2. **새로운 연결 발견 (Inter-conversation Links)**:
   - **가장 중요한 작업입니다.** 서로 다른 대화에서 온 노드들 사이에 숨겨진 연관성을 찾으세요.
   - 예: 대화 A의 '선형 회귀'와 대화 B의 '신경망' 사이에 'FOUNDATION_OF' 관계를 추가.
   - 예: 대화 A의 '문제점'과 대화 B의 '해결책'을 연결.

3. **출력 생성**:
   - 병합된 nodes와 edges를 포함하는 하나의 JSON 객체를 반환하세요.

[출력 형식]
순수 JSON만 출력.

{{{{
    "nodes": [
        {{{{
            "id": "ConceptA",
            "label": "개념 A",
            "description": "통합된 설명...",
            "conversation_ids": ["id_1", "id_3"] 
        }}}}
    ],
    "edges": [
        {{{{ "source": "ConceptA", "target": "ConceptB", "label": "RELATED_TO" }}}}
    ]
}}}}

[그래프 데이터 목록]
{graphs_json}
"""

class GraphService:
    """Service for generating knowledge graphs using OpenAI."""
    
    def __init__(self):
        """Initialize the OpenAI client."""
        
        # --- FAILSAFE LOADING START ---
        if not os.getenv("OPENAI_API_KEY"):
            env_path = Path(__file__).resolve().parent.parent / ".env"
            load_dotenv(dotenv_path=env_path)
            logger.info(f"GraphService attempting to load .env from: {env_path}")
        # --- FAILSAFE LOADING END ---

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set. Graph service will not be available.")
            self.client = None
        else:
            try:
                self.client = OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized for graph service")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None
    
    def format_conversation(self, messages: List[Dict[str, Any]]) -> str:
        conversation = []
        for i, msg in enumerate(messages):
            role = "사용자" if msg.get("role") == "user" else "AI 어시스턴트"
            content = msg.get("content", "")
            conversation.append(f"[{i}] {role}: {content}")
        return "\n\n".join(conversation)
    
    def extract_json_from_response(self, content: str) -> Dict[str, Any]:
        try:
            content_cleaned = re.sub(r'^```json?\s*\n?', '', content, flags=re.MULTILINE)
            content_cleaned = re.sub(r'\n?```\s*$', '', content_cleaned, flags=re.MULTILINE)
            json_match = re.search(r'\{[\s\S]*\}', content_cleaned)
            if json_match:
                return json.loads(json_match.group())
            else:
                return json.loads(content_cleaned)
        except Exception as e:
            logger.error(f"JSON extraction failed: {e}")
            raise ValueError("Failed to parse JSON response")

    def generate_graph(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a graph for a single conversation."""
        if not self.client:
            return {"error": "Service unavailable"}
        
        try:
            document = self.format_conversation(messages)
            prompt = GRAPH_PROMPT_TEMPLATE.format(document=document)
            
            response = self.client.chat.completions.create(
                model="gpt-4o", # Use a smart model for structured output
                messages=[{"role": "system", "content": prompt}],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            result = self.extract_json_from_response(content)
            
            if "graph" not in result:
                result["graph"] = {"nodes": [], "edges": []}
                
            return result
            
        except Exception as e:
            logger.error(f"Graph generation error: {e}")
            return {"error": str(e)}

    def generate_combined_graph(self, graphs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple graph summaries into one."""
        if not self.client:
            return {"error": "Service unavailable"}

        try:
            graphs_json = json.dumps(graphs, ensure_ascii=False, indent=2)
            prompt = COMBINED_GRAPH_PROMPT.format(graphs_json=graphs_json)

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.2,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            result = self.extract_json_from_response(content)
            return result

        except Exception as e:
            logger.error(f"Combined graph generation error: {e}")
            return {"error": str(e)}

    def is_available(self) -> bool:
        return self.client is not None

# Global service instance
graph_service = GraphService()