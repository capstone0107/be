from pydantic import BaseModel
from typing import List, Optional

class QuizResponse(BaseModel):
    """퀴즈 응답"""
    id: int
    question: str
    options: List[str]
    correct_answer: int
    explanation: Optional[str]
    related_question: str
    source_url: Optional[str]
    source_title: Optional[str]

    model_config = {
        "from_attributes": True
    }


class QuizListResponse(BaseModel):
    """퀴즈 목록 응답"""
    quizzes: List[QuizResponse]
    total: int