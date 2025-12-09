import logging
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from models.quiz import QuizResponse, QuizListResponse
from services.quiz_service import quiz_service
from routers.user import get_current_user
from models.orm import User as DBUser

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/quiz", tags=["quiz"])


@router.get("", response_model=QuizListResponse)
async def get_quizzes(
    current_user: Annotated[DBUser, Depends(get_current_user)],
    db: Session = Depends(get_db)
):
    """현재 사용자의 전체 퀴즈 조회"""
    try:
        quizzes, total = quiz_service.get_user_quizzes(
            user_id=current_user.id,
            db=db
        )
        
        return QuizListResponse(
            quizzes=[QuizResponse.model_validate(q) for q in quizzes],
            total=total
        )
        
    except Exception as e:
        logger.error(f"Error getting quizzes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{quiz_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_quiz(
    quiz_id: int,
    current_user: Annotated[DBUser, Depends(get_current_user)],
    db: Session = Depends(get_db)
):
    """퀴즈 삭제"""
    try:
        success = quiz_service.delete_quiz(
            quiz_id=quiz_id,
            user_id=current_user.id,
            db=db
        )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="퀴즈를 찾을 수 없습니다"
            )
        
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting quiz: {e}")
        raise HTTPException(status_code=500, detail=str(e))