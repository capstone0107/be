from pydantic import BaseModel, HttpUrl
from typing import Optional
from datetime import datetime

# Request Models
class BookmarkCreateRequest(BaseModel):
    """북마크 생성 요청"""
    knowledge_id: str  # "conv-123-msg-5" 형식
    source_url: str
    title: str
    summary: str
    model_version: Optional[str] = None

class BookmarkUpdateRequest(BaseModel):
    """북마크 수정 요청 (제목, 요약만 수정 가능)"""
    title: Optional[str] = None
    summary: Optional[str] = None

# Response Models
class BookmarkResponse(BaseModel):
    """북마크 응답"""
    id: int
    user_id: int
    knowledge_id: str
    source_url: str
    title: str
    summary: str
    model_version: Optional[str]
    created_at: datetime
    
    model_config = {
        "from_attributes": True
    }

class BookmarkListResponse(BaseModel):
    """북마크 목록 응답"""
    bookmarks: list[BookmarkResponse]
    total: int
    page: int
    page_size: int