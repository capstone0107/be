import logging
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc

from models.bookmark_orm import KnowledgeCard

logger = logging.getLogger(__name__)


class BookmarkService:
    """북마크 관리 서비스"""
    
    def create_bookmark(
        self,
        user_id: int,
        knowledge_id: str,
        source_url: str,
        title: str,
        summary: str,
        model_version: Optional[str],
        db: Session
    ) -> KnowledgeCard:
        """
        새로운 북마크 생성
        
        Args:
            user_id: 사용자 ID
            knowledge_id: 지식 카드 식별자
            source_url: 출처 URL
            title: 제목
            summary: 요약
            model_version: AI 모델 버전
            db: DB 세션
            
        Returns:
            생성된 북마크
        """
        try:
            # 중복 체크 (같은 사용자가 같은 knowledge_id를 이미 북마크했는지)
            existing = db.query(KnowledgeCard).filter(
                KnowledgeCard.user_id == user_id,
                KnowledgeCard.knowledge_id == knowledge_id
            ).first()
            
            if existing:
                logger.info(f"Bookmark already exists: {knowledge_id} for user {user_id}")
                return existing
            
            # 새 북마크 생성
            bookmark = KnowledgeCard(
                user_id=user_id,
                knowledge_id=knowledge_id,
                source_url=source_url,
                title=title,
                summary=summary,
                model_version=model_version
            )
            
            db.add(bookmark)
            db.commit()
            db.refresh(bookmark)
            
            logger.info(f"✅ Created bookmark {bookmark.id} for user {user_id}")
            return bookmark
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create bookmark: {e}")
            raise
    
    def get_user_bookmarks(
        self,
        user_id: int,
        db: Session,
        page: int = 1,
        page_size: int = 20
    ) -> tuple[List[KnowledgeCard], int]:
        """
        사용자의 북마크 목록 조회
        
        Args:
            user_id: 사용자 ID
            db: DB 세션
            page: 페이지 번호 (1부터 시작)
            page_size: 페이지 크기
            
        Returns:
            (북마크 리스트, 전체 개수)
        """
        try:
            # 전체 개수
            total = db.query(KnowledgeCard).filter(
                KnowledgeCard.user_id == user_id
            ).count()
            
            # 페이징 적용
            offset = (page - 1) * page_size
            bookmarks = db.query(KnowledgeCard).filter(
                KnowledgeCard.user_id == user_id
            ).order_by(
                desc(KnowledgeCard.created_at)
            ).offset(offset).limit(page_size).all()
            
            return bookmarks, total
            
        except Exception as e:
            logger.error(f"Failed to get bookmarks: {e}")
            raise
    
    def get_bookmark_by_id(
        self,
        bookmark_id: int,
        user_id: int,
        db: Session
    ) -> Optional[KnowledgeCard]:
        """
        특정 북마크 조회 (본인 것만 조회 가능)
        
        Args:
            bookmark_id: 북마크 ID
            user_id: 사용자 ID
            db: DB 세션
            
        Returns:
            북마크 객체 또는 None
        """
        try:
            return db.query(KnowledgeCard).filter(
                KnowledgeCard.id == bookmark_id,
                KnowledgeCard.user_id == user_id
            ).first()
            
        except Exception as e:
            logger.error(f"Failed to get bookmark: {e}")
            raise
    
    def update_bookmark(
        self,
        bookmark_id: int,
        user_id: int,
        title: Optional[str],
        summary: Optional[str],
        db: Session
    ) -> Optional[KnowledgeCard]:
        """
        북마크 수정
        
        Args:
            bookmark_id: 북마크 ID
            user_id: 사용자 ID
            title: 새 제목
            summary: 새 요약
            db: DB 세션
            
        Returns:
            수정된 북마크 또는 None
        """
        try:
            bookmark = self.get_bookmark_by_id(bookmark_id, user_id, db)
            
            if not bookmark:
                return None
            
            if title:
                bookmark.title = title
            if summary:
                bookmark.summary = summary
            
            db.commit()
            db.refresh(bookmark)
            
            logger.info(f"✅ Updated bookmark {bookmark_id}")
            return bookmark
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to update bookmark: {e}")
            raise
    
    def delete_bookmark(
        self,
        bookmark_id: int,
        user_id: int,
        db: Session
    ) -> bool:
        """
        북마크 삭제
        
        Args:
            bookmark_id: 북마크 ID
            user_id: 사용자 ID
            db: DB 세션
            
        Returns:
            삭제 성공 여부
        """
        try:
            bookmark = self.get_bookmark_by_id(bookmark_id, user_id, db)
            
            if not bookmark:
                return False
            
            db.delete(bookmark)
            db.commit()
            
            logger.info(f"✅ Deleted bookmark {bookmark_id}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete bookmark: {e}")
            raise
    
    def search_bookmarks(
        self,
        user_id: int,
        keyword: str,
        db: Session,
        page: int = 1,
        page_size: int = 20
    ) -> tuple[List[KnowledgeCard], int]:
        """
        북마크 검색 (제목 또는 요약에서 키워드 검색)
        
        Args:
            user_id: 사용자 ID
            keyword: 검색 키워드
            db: DB 세션
            page: 페이지 번호
            page_size: 페이지 크기
            
        Returns:
            (검색 결과, 전체 개수)
        """
        try:
            # 검색 쿼리
            query = db.query(KnowledgeCard).filter(
                KnowledgeCard.user_id == user_id,
                (KnowledgeCard.title.contains(keyword)) | 
                (KnowledgeCard.summary.contains(keyword))
            )
            
            total = query.count()
            
            offset = (page - 1) * page_size
            bookmarks = query.order_by(
                desc(KnowledgeCard.created_at)
            ).offset(offset).limit(page_size).all()
            
            return bookmarks, total
            
        except Exception as e:
            logger.error(f"Failed to search bookmarks: {e}")
            raise


# 전역 서비스 인스턴스
bookmark_service = BookmarkService()