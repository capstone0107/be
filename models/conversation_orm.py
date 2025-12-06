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
    
    Lifecycle:
    1. 대화 시작: id만 설정, is_saved=0 (임시)
    2. 메시지 저장: messages 관계로 추가
    3. 사용자 저장: title/summary 업데이트, is_saved=1
    """
    __tablename__ = "conversations"
    
    id = Column(String(50), primary_key=True)  # "auto-1732950000000" 형식
    title = Column(String(200), nullable=True)  # 사용자 저장 시 업데이트
    summary = Column(Text, nullable=True)  # Focus 분류 시 업데이트
    is_saved = Column(Integer, default=0, nullable=False)  # 0: 임시, 1: 저장완료
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)  # 소유 사용자
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.message_order"
    )
    focuses = relationship(
        "Focus",
        secondary=conversation_focus,
        back_populates="conversations"
    )
    
    def __repr__(self):
        return f"<Conversation(id='{self.id}', user_id={self.user_id}, is_saved={self.is_saved}, messages={len(self.messages)})>"


class Message(Base):
    """
    대화 메시지를 저장하는 테이블
    
    role 구분:
    - 'user': 사용자 질문
    - 'assistant': AI 응답
    """
    __tablename__ = "messages"
    
    id = Column(String(50), primary_key=True)  # UUID 형식
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
    
    생성 시점: 사용자가 저장 버튼을 누를 때만
    
    ⭐ user_id 추가: Focus는 특정 사용자가 소유하며, 
    같은 conversation_id라도 사용자마다 다른 Focus를 가질 수 있습니다.
    """
    __tablename__ = "focuses"
    
    id = Column(String(50), primary_key=True)  # "focus-cpu-scheduling" 형식
    name = Column(String(200), nullable=False)  # "CPU 스케줄링 기초"
    message_ids = Column(JSON, nullable=False)  # ["msg-1", "msg-2", "msg-3"]
    question_tags = Column(JSON, nullable=False)  # ["FCFS", "Round Robin"]
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)  # ⭐ 소유 사용자 추가
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    conversations = relationship(
        "Conversation",
        secondary=conversation_focus,
        back_populates="focuses"
    )
    
    def __repr__(self):
        return f"<Focus(id='{self.id}', name='{self.name}', user_id={self.user_id})>"