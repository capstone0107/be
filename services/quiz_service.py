import logging
from typing import List, Tuple, Optional
from sqlalchemy.orm import Session

from models.orm import Quiz as DBQuiz
from services.llm_service import llm_service

logger = logging.getLogger(__name__)


class QuizService:

    def create_quiz(
        self,
        user_id: int,
        title: str,
        summary: str,
        source_url: str,
        user_question: str,
        db: Session
    ) -> Optional[DBQuiz]:
        """
        LLM을 통해 퀴즈 생성 후 DB에 저장
        """
        try:
            # 1. LLM으로 퀴즈 생성
            quiz_data = llm_service.generate_quiz(
                title=title,
                summary=summary,
                source_url=source_url,
                user_question=user_question
            )
            
            if not quiz_data:
                logger.warning("퀴즈 생성 실패: LLM 응답 없음")
                return None

            # 2. DB에 저장
            quiz = DBQuiz(
                user_id=user_id,
                question=quiz_data.question,
                options=quiz_data.options,
                correct_answer=quiz_data.correct_answer,
                explanation=quiz_data.explanation,
                related_question=user_question,
                source_url=source_url,
                source_title=title
            )
            
            db.add(quiz)
            db.commit()
            db.refresh(quiz)
            
            logger.info(f"Quiz created: id={quiz.id}, user_id={user_id}")
            return quiz

        except Exception as e:
            logger.error(f"퀴즈 생성/저장 오류: {e}", exc_info=True)
            db.rollback()
            return None

    def get_user_quizzes(
        self,
        user_id: int,
        db: Session
    ) -> Tuple[List[DBQuiz], int]:
        """사용자의 전체 퀴즈 조회"""
        query = db.query(DBQuiz).filter(DBQuiz.user_id == user_id)
        total = query.count()
        quizzes = query.order_by(DBQuiz.id.desc()).all()
        
        return quizzes, total

    def delete_quiz(
        self,
        quiz_id: int,
        user_id: int,
        db: Session
    ) -> bool:
        """퀴즈 삭제"""
        quiz = db.query(DBQuiz).filter(
            DBQuiz.id == quiz_id,
            DBQuiz.user_id == user_id
        ).first()
        
        if not quiz:
            return False
        
        db.delete(quiz)
        db.commit()
        
        logger.info(f"Quiz deleted: id={quiz_id}, user_id={user_id}")
        return True


quiz_service = QuizService()