import logging
from typing import List, Tuple
from sqlalchemy.orm import Session

from models.orm import Quiz as DBQuiz

logger = logging.getLogger(__name__)


class QuizService:

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