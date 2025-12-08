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
from routers.user import get_current_user
from models.orm import User as DBUser

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
    db: Session = Depends(get_db)
):
    """
    새로운 북마크 생성
    """
    try:
        # ⭐ 디버깅: 받은 데이터 출력
        logger.info(f"=== Bookmark Request ===")
        logger.info(f"User ID: {current_user.id}")
        logger.info(f"Request data: {request.dict()}")
        logger.info(f"========================")
        
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
        
        return BookmarkResponse.model_validate(bookmark)
        
    except Exception as e:
        logger.error(f"Error creating bookmark: {e}", exc_info=True)  # ⭐ 스택 트레이스 포함
        raise HTTPException(status_code=500, detail=str(e))
    

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