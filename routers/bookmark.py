import logging
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from database import get_db
from models.bookmark import (
    BookmarkCreateRequest,
    BookmarkUpdateRequest,
    BookmarkResponse,
    BookmarkListResponse
)
from services.bookmark_service import bookmark_service
from services.quiz_service import quiz_service
from services.document_service import document_service
from routers.user import get_current_user
from models.orm import User as DBUser

from fastapi import BackgroundTasks

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/bookmarks", tags=["bookmarks"])

print("=" * 50)
print("BOOKMARK ROUTER LOADED!")
print("=" * 50)
logger.warning("BOOKMARK ROUTER LOADED - THIS SHOULD APPEAR IN LOGS")



@router.post("", response_model=BookmarkResponse, status_code=status.HTTP_201_CREATED)
async def create_bookmark(
    request: BookmarkCreateRequest,
    current_user: Annotated[DBUser, Depends(get_current_user)],
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    try:
        # 1. 북마크 생성
        bookmark = bookmark_service.create_bookmark(
            user_id=current_user.id,
            knowledge_id=request.knowledge_id,
            source_url=request.source_url,
            title=request.title,
            summary=request.summary,
            question=request.question,
            model_version=request.model_version,
            db=db
        )
        
        # 2. 퀴즈 생성 백그라운드 태스크
        background_tasks.add_task(
            create_quiz_background,
            current_user.id,
            request.title,
            request.summary or "",
            request.source_url,
            request.question or "기타"
        )
        
        # 3. 도큐먼트 생성 백그라운드 태스크
        background_tasks.add_task(
            create_document_background,
            current_user.id,
            request.title,
            request.summary or "",
            request.source_url,
            request.question or "기타"
        )
        
        return BookmarkResponse.model_validate(bookmark)
        
    except Exception as e:
        logger.error(f"Error creating bookmark: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def create_quiz_background(
    user_id: int,
    title: str,
    summary: str,
    source_url: str,
    user_question: str
):
    """백그라운드에서 퀴즈 생성"""
    from database import SessionLocal
    
    db = SessionLocal()
    try:
        quiz = quiz_service.create_quiz(
            user_id=user_id,
            title=title,
            summary=summary,
            source_url=source_url,
            user_question=user_question,
            db=db
        )
        if quiz:
            logger.info(f"Quiz auto-created in background: id={quiz.id}")
        else:
            logger.warning("Background quiz creation failed")
    except Exception as e:
        logger.error(f"Background quiz creation error: {e}")
    finally:
        db.close()

def create_document_background(
    user_id: int,
    title: str,
    summary: str,
    source_url: str,
    user_question: str
):
    """백그라운드에서 도큐먼트 생성"""
    from database import SessionLocal
    
    db = SessionLocal()
    try:
        document = document_service.create_document(
            user_id=user_id,
            title=title,
            summary=summary,
            source_url=source_url,
            user_question=user_question,
            db=db
        )
        if document:
            logger.info(f"Document auto-created in background: id={document.id}")
        else:
            logger.warning("Background document creation failed")
    except Exception as e:
        logger.error(f"Background document creation error: {e}")
    finally:
        db.close()


@router.get("", response_model=BookmarkListResponse)
async def get_bookmarks(
    current_user: Annotated[DBUser, Depends(get_current_user)],
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1, description="페이지 번호"),
    page_size: int = Query(20, ge=1, le=100, description="페이지 크기")
):
    """
    현재 사용자의 북마크 목록 조회
    - 최신순으로 정렬
    - 페이징 지원
    """
    try:
        bookmarks, total = bookmark_service.get_user_bookmarks(
            user_id=current_user.id,
            db=db,
            page=page,
            page_size=page_size
        )
        
        return BookmarkListResponse(
            bookmarks=[BookmarkResponse.model_validate(b) for b in bookmarks],
            total=total,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Error getting bookmarks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search", response_model=BookmarkListResponse)
async def search_bookmarks(
    current_user: Annotated[DBUser, Depends(get_current_user)],
    keyword: str = Query(..., min_length=1, description="검색 키워드"),
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100)
):
    """
    북마크 검색
    
    - 제목 또는 요약에서 키워드 검색
    - 대소문자 구분 없음
    """
    try:
        bookmarks, total = bookmark_service.search_bookmarks(
            user_id=current_user.id,
            keyword=keyword,
            db=db,
            page=page,
            page_size=page_size
        )
        
        return BookmarkListResponse(
            bookmarks=[BookmarkResponse.model_validate(b) for b in bookmarks],
            total=total,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Error searching bookmarks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{bookmark_id}", response_model=BookmarkResponse)
async def get_bookmark(
    bookmark_id: int,
    current_user: Annotated[DBUser, Depends(get_current_user)],
    db: Session = Depends(get_db)
):
    """특정 북마크 조회"""
    try:
        bookmark = bookmark_service.get_bookmark_by_id(
            bookmark_id=bookmark_id,
            user_id=current_user.id,
            db=db
        )
        
        if not bookmark:
            raise HTTPException(
                status_code=404,
                detail="북마크를 찾을 수 없습니다"
            )
        
        return BookmarkResponse.model_validate(bookmark)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting bookmark: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{bookmark_id}", response_model=BookmarkResponse)
async def update_bookmark(
    bookmark_id: int,
    request: BookmarkUpdateRequest,
    current_user: Annotated[DBUser, Depends(get_current_user)],
    db: Session = Depends(get_db)
):
    """
    북마크 수정
    
    - 제목과 요약만 수정 가능
    - 본인의 북마크만 수정 가능
    """
    try:
        bookmark = bookmark_service.update_bookmark(
            bookmark_id=bookmark_id,
            user_id=current_user.id,
            title=request.title,
            summary=request.summary,
            db=db
        )
        
        if not bookmark:
            raise HTTPException(
                status_code=404,
                detail="북마크를 찾을 수 없습니다"
            )
        
        return BookmarkResponse.model_validate(bookmark)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating bookmark: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{bookmark_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_bookmark(
    bookmark_id: int,
    current_user: Annotated[DBUser, Depends(get_current_user)],
    db: Session = Depends(get_db)
):
    """
    북마크 삭제
    
    - 본인의 북마크만 삭제 가능
    """
    try:
        success = bookmark_service.delete_bookmark(
            bookmark_id=bookmark_id,
            user_id=current_user.id,
            db=db
        )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="북마크를 찾을 수 없습니다"
            )
        
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting bookmark: {e}")
        raise HTTPException(status_code=500, detail=str(e))