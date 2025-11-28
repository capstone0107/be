"""
Matching service for LLM-based presentation-note matching with critical thinking analysis.
"""
import os
import json
import logging
from typing import Dict, Any, List
from openai import OpenAI

logger = logging.getLogger(__name__)


class MatchingService:
    """Service for matching presentation sections with user notes using LLM with critical thinking."""
    
    def __init__(self):
        """Initialize the OpenAI client."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set. Matching service will not be available.")
            self.client = None
        else:
            try:
                self.client = OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized for matching service")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None
    
    def match_note_to_section(self, section: Dict[str, Any], note: Dict[str, Any]) -> Dict[str, Any]:
        """
        Match a single note to a presentation section using LLM with critical thinking analysis.
        
        Args:
            section: Presentation section with title and content
            note: User note with author, type, title, content
            
        Returns:
            Matching result with relevance, relationship, critical analysis, and perspectives
        """
        if not self.client:
            return {
                "error": "Matching service not available. Check OpenAI API key."
            }
        
        try:
            prompt = f"""
발표 섹션과 노트를 **비판적 사고** 관점에서 분석하세요.

━━━ 발표 섹션 ━━━
제목: {section.get('title', '')}
내용:
{section.get('content', '')}

━━━ 노트 ━━━
작성자: {note.get('author', '')}
유형: {note.get('type', '')}
제목: {note.get('title', '')}
내용:
{note.get('content', '')}

[분석 목표]
1. 두 텍스트가 다룬 주제의 관련성 파악
2. 각 텍스트가 제시하는 **관점(perspective)** 식별
3. 관점 간 **대립 지점(conflict points)** 발견
4. 비판적 사고를 위한 **추가 고려사항** 제시

[관점 분류]
- 긍정적 관점: 특정 개념/방법론의 장점, 효과, 가치를 강조
- 비판적 관점: 한계점, 단점, 위험성을 지적
- 절충적 관점: 장단점을 균형있게 제시, 조건부 적용 제안
- 대안 제시: 다른 접근법이나 해결책 제안
- 중립적 설명: 객관적 사실이나 정의만 제시

[대립 지점 식별 기준]
- 서로 다른 가정이나 전제 조건
- 상반된 평가나 결론
- 적용 맥락의 차이로 인한 갈등
- 우선순위나 가치 판단의 차이

다음 형식의 JSON으로만 답변하세요. 마크다운 코드 블록을 사용하지 마세요:
{{
  "relevance": "높음|중간|낮음|없음",
  "relationship": "같은내용|보완|상충|무관",
  "section_perspective": "발표 섹션의 관점 (긍정적|비판적|절충적|대안제시|중립적)",
  "note_perspective": "노트의 관점 (긍정적|비판적|절충적|대안제시|중립적)",
  "conflict_points": [
    "대립 지점 1 (구체적으로, 없으면 빈 배열)",
    "대립 지점 2"
  ],
  "critical_insights": [
    "비판적으로 고려할 점 1 (예: 전제 조건, 맥락, 트레이드오프)",
    "비판적으로 고려할 점 2"
  ],
  "perspective_diversity_score": "관점 다양성 점수 (0-10, 높을수록 다양한 관점 제시)",
  "explanation": "두 텍스트의 관계를 비판적 관점에서 2-3문장으로 설명"
}}

[예시 1: 대립하는 관점]
- 발표: "라운드 로빈 스케줄링은 모든 프로세스에 공정하게 CPU 시간을 분배한다"
- 노트: "실시간 시스템에서는 라운드 로빈이 데드라인 보장에 실패할 수 있어 부적합하다"
→ section_perspective: "긍정적"
→ note_perspective: "비판적"
→ conflict_points: ["공정성 vs 실시간 성능", "일반 시스템 vs 특수 환경 적용"]
→ critical_insights: ["'공정성'의 정의가 맥락에 따라 달라짐", "시스템 특성을 고려한 평가 필요"]
→ perspective_diversity_score: 8

[예시 2: 보완하는 관점]
- 발표: "캐시 메모리는 CPU와 RAM 사이의 속도 차이를 줄인다"
- 노트: "다만 캐시 적중률이 낮으면 오히려 성능 저하가 발생할 수 있다"
→ section_perspective: "중립적"
→ note_perspective: "비판적"
→ conflict_points: []
→ critical_insights: ["캐시 효율성은 접근 패턴에 의존적", "모든 상황에서 캐시가 유리한 것은 아님"]
→ perspective_diversity_score: 5
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0
            )
            
            content = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            result = json.loads(content)
            
            # Add note metadata to result
            result["note_id"] = note.get("note_id")
            result["author"] = note.get("author")
            result["note_title"] = note.get("title")
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.error(f"Response content: {content if 'content' in locals() else 'N/A'}")
            return {
                "error": f"Failed to parse LLM response: {str(e)}",
                "note_id": note.get("note_id")
            }
        except Exception as e:
            logger.error(f"Matching error: {e}")
            return {
                "error": str(e),
                "note_id": note.get("note_id")
            }
    
    def match_section_to_all_notes(
        self, 
        section: Dict[str, Any], 
        notes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Match a section to all notes with critical thinking analysis.
        
        Args:
            section: Presentation section
            notes: List of all user notes
            
        Returns:
            List of matching results sorted by relevance and perspective diversity
        """
        results = []
        
        for note in notes:
            result = self.match_note_to_section(section, note)
            if "error" not in result:
                results.append(result)
            else:
                logger.warning(f"Skipping note {note.get('note_id')}: {result.get('error')}")
        
        # Sort by relevance priority, then by perspective diversity
        relevance_order = {"높음": 0, "중간": 1, "낮음": 2, "없음": 3}
        results.sort(
            key=lambda x: (
                relevance_order.get(x.get("relevance", "없음"), 4),
                -x.get("perspective_diversity_score", 0)  # 높은 다양성 점수 우선
            )
        )
        
        return results
    
    def analyze_perspective_distribution(
        self,
        matches: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze the distribution of perspectives in matched notes.
        
        Args:
            matches: List of matching results
            
        Returns:
            Statistical analysis of perspective distribution
        """
        perspective_counts = {
            "긍정적": 0,
            "비판적": 0,
            "절충적": 0,
            "대안제시": 0,
            "중립적": 0
        }
        
        conflict_count = 0
        total_insights = 0
        
        for match in matches:
            # Count perspectives
            note_perspective = match.get("note_perspective", "")
            if note_perspective in perspective_counts:
                perspective_counts[note_perspective] += 1
            
            # Count conflicts
            if match.get("conflict_points") and len(match.get("conflict_points", [])) > 0:
                conflict_count += 1
            
            # Count insights
            total_insights += len(match.get("critical_insights", []))
        
        return {
            "perspective_distribution": perspective_counts,
            "conflict_count": conflict_count,
            "total_matches": len(matches),
            "average_insights_per_match": total_insights / len(matches) if matches else 0,
            "diversity_index": self._calculate_diversity_index(perspective_counts)
        }
    
    def _calculate_diversity_index(self, perspective_counts: Dict[str, int]) -> float:
        """
        Calculate Shannon diversity index for perspective distribution.
        
        Args:
            perspective_counts: Dictionary of perspective counts
            
        Returns:
            Diversity index (0-1, higher is more diverse)
        """
        import math
        
        total = sum(perspective_counts.values())
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for count in perspective_counts.values():
            if count > 0:
                proportion = count / total
                entropy -= proportion * math.log2(proportion)
        
        # Normalize to 0-1 range (max entropy for 5 categories is log2(5))
        max_entropy = math.log2(5)
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def match_presentation_to_notes(
        self,
        presentation: Dict[str, Any],
        notes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Match entire presentation to all notes with critical thinking analysis.
        
        Args:
            presentation: Full presentation with sections
            notes: List of all user notes
            
        Returns:
            Matching results for all sections with perspective analysis
        """
        results = {
            "presentation_id": presentation.get("presentation_id"),
            "presentation_title": presentation.get("title"),
            "author": presentation.get("author"),
            "total_notes": len(notes),
            "sections": []
        }
        
        for section in presentation.get("sections", []):
            logger.info(f"Matching section: {section.get('title')}")
            
            section_matches = self.match_section_to_all_notes(section, notes)
            
            # Categorize by relevance
            high_relevance = [m for m in section_matches if m.get("relevance") == "높음"]
            medium_relevance = [m for m in section_matches if m.get("relevance") == "중간"]
            low_relevance = [m for m in section_matches if m.get("relevance") == "낮음"]
            no_relevance = [m for m in section_matches if m.get("relevance") == "없음"]
            
            # Categorize by conflict
            conflicting = [m for m in section_matches if m.get("conflict_points") and len(m.get("conflict_points", [])) > 0]
            complementary = [m for m in section_matches if m.get("relationship") == "보완"]
            
            # Perspective analysis
            perspective_analysis = self.analyze_perspective_distribution(section_matches)
            
            results["sections"].append({
                "section_id": section.get("section_id"),
                "section_number": section.get("number"),
                "section_title": section.get("title"),
                "total_matches": len(section_matches),
                "matches": {
                    "high": high_relevance,
                    "medium": medium_relevance,
                    "low": low_relevance,
                    "none": no_relevance
                },
                "critical_thinking": {
                    "conflicting_notes": conflicting,
                    "complementary_notes": complementary,
                    "perspective_analysis": perspective_analysis
                },
                "summary": {
                    "high_count": len(high_relevance),
                    "medium_count": len(medium_relevance),
                    "low_count": len(low_relevance),
                    "none_count": len(no_relevance),
                    "conflict_count": len(conflicting),
                    "complementary_count": len(complementary),
                    "diversity_index": perspective_analysis.get("diversity_index", 0)
                }
            })
        
        return results
    
    def is_available(self) -> bool:
        """Check if the matching service is available."""
        return self.client is not None


# Global service instance
matching_service = MatchingService()