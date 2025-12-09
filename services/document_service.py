import logging
from typing import List, Tuple, Optional
from sqlalchemy.orm import Session

from models.orm import Document as DBDocument
from services.llm_service import llm_service

logger = logging.getLogger(__name__)


class DocumentService:

    def create_document(
        self,
        user_id: int,
        title: str,
        summary: str,
        source_url: str,
        user_question: str,
        db: Session
    ) -> Optional[DBDocument]:
        """
        LLM을 통해 도큐먼트 생성 후 DB에 저장
        """
        try:
            # 1. LLM으로 도큐먼트 생성
            doc_data = llm_service.generate_document(
                title=title,
                summary=summary,
                source_url=source_url,
                user_question=user_question
            )
            
            if not doc_data:
                logger.warning("도큐먼트 생성 실패: LLM 응답 없음")
                return None

            # 2. DB에 저장
            document = DBDocument(
                user_id=user_id,
                title=doc_data.title,
                content=doc_data.content,
                source_title=title,
                source_url=source_url,
                related_question=user_question
            )
            
            db.add(document)
            db.commit()
            db.refresh(document)
            
            logger.info(f"Document created: id={document.id}, user_id={user_id}")
            return document

        except Exception as e:
            logger.error(f"도큐먼트 생성/저장 오류: {e}", exc_info=True)
            db.rollback()
            return None

    def get_user_documents(
        self,
        user_id: int,
        db: Session
    ) -> Tuple[List[DBDocument], int]:
        """사용자의 전체 도큐먼트 조회"""
        query = db.query(DBDocument).filter(DBDocument.user_id == user_id)
        total = query.count()
        documents = query.order_by(DBDocument.id.desc()).all()
        
        return documents, total

    def delete_document(
        self,
        document_id: int,
        user_id: int,
        db: Session
    ) -> bool:
        """도큐먼트 삭제"""
        document = db.query(DBDocument).filter(
            DBDocument.id == document_id,
            DBDocument.user_id == user_id
        ).first()
        
        if not document:
            return False
        
        db.delete(document)
        db.commit()
        
        logger.info(f"Document deleted: id={document_id}, user_id={user_id}")
        return True


document_service = DocumentService()