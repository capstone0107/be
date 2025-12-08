from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class KnowledgeCard(Base):
    """
    사용자가 북마크한 지식 카드를 저장하는 테이블
    
    AI 응답에서 제공된 출처 정보를 사용자가 저장할 때 사용
    """
    __tablename__ = "knowledge_cards"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    
    # 지식 카드 식별자 (conversation_id + message_order 조합 등)
    knowledge_id = Column(String(100), nullable=False, index=True)
    
    # 출처 정보
    source_url = Column(String(500), nullable=False)
    title = Column(String(200), nullable=False)
    summary = Column(Text, nullable=False)
    question = Column(Text, nullable=True)
    # 메타데이터
    model_version = Column(String(50), nullable=True)  # 사용된 AI 모델 버전
    
    # 타임스탬프
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", backref="bookmarks")
    
    def __repr__(self):
        return f"<KnowledgeCard(id={self.id}, user_id={self.user_id}, title='{self.title}')>"