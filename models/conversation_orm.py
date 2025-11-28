"""
ORM models for conversation, message, and focus data.
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Table, Float
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.mysql import JSON
from datetime import datetime
from database import Base

# Many-to-Many relationship table for Conversation <-> Focus
conversation_focus = Table(
    'conversation_focus',
    Base.metadata,
    Column('conversation_id', String(50), ForeignKey('conversations.id'), primary_key=True),
    Column('focus_id', String(50), ForeignKey('focuses.id'), primary_key=True),
    Column('confidence', Float, default=1.0),
    Column('reason', Text, nullable=True)
)


class Conversation(Base):
    """
    대화 세션을 저장하는 테이블
    """
    __tablename__ = "conversations"
    
    id = Column(String(50), primary_key=True)  # "1732800000000" 형식
    title = Column(String(200), nullable=False)
    summary = Column(Text, nullable=True)  # 대화 전체 요약
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)  # 향후 사용자 연동
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    focuses = relationship("Focus", secondary=conversation_focus, back_populates="conversations")
    
    def __repr__(self):
        return f"<Conversation(id='{self.id}', title='{self.title}')>"


class Message(Base):
    """
    대화 메시지를 저장하는 테이블
    """
    __tablename__ = "messages"
    
    id = Column(String(50), primary_key=True)  # "msg-1", "msg-2" 형식
    conversation_id = Column(String(50), ForeignKey('conversations.id'), nullable=False)
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    sources = Column(JSON, nullable=True)  # [{"title": "", "url": "", "snippet": ""}]
    message_order = Column(Integer, nullable=False)  # 메시지 순서 (0부터 시작)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    
    def __repr__(self):
        return f"<Message(id='{self.id}', role='{self.role}', order={self.message_order})>"


class Focus(Base):
    """
    Focus 주제를 저장하는 테이블
    """
    __tablename__ = "focuses"
    
    id = Column(String(50), primary_key=True)  # "focus-cpu-scheduling" 형식
    name = Column(String(200), nullable=False)  # "CPU 스케줄링 기술 최적화"
    message_ids = Column(JSON, nullable=False)  # ["msg-1", "msg-2", "msg-3"]
    question_tags = Column(JSON, nullable=False)  # ["FCFS 알고리즘", "병렬 처리"]
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    conversations = relationship("Conversation", secondary=conversation_focus, back_populates="focuses")
    
    def __repr__(self):
        return f"<Focus(id='{self.id}', name='{self.name}')>"