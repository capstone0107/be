import logging
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from models.document import DocumentResponse, DocumentListResponse
from services import document_service
from routers.user import get_current_user
from models.orm import User as DBUser

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=DocumentListResponse)
async def get_documents(
    current_user: Annotated[DBUser, Depends(get_current_user)],
    db: Session = Depends(get_db)
):
    """현재 사용자의 전체 도큐먼트 조회"""
    try:
        documents, total = document_service.get_user_documents(
            user_id=current_user.id,
            db=db
        )
        
        return DocumentListResponse(
            documents=[DocumentResponse.model_validate(d) for d in documents],
            total=total
        )
        
    except Exception as e:
        logger.error(f"Error getting documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: int,
    current_user: Annotated[DBUser, Depends(get_current_user)],
    db: Session = Depends(get_db)
):
    """도큐먼트 삭제"""
    try:
        success = document_service.delete_document(
            document_id=document_id,
            user_id=current_user.id,
            db=db
        )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="도큐먼트를 찾을 수 없습니다"
            )
        
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))