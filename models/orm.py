from sqlalchemy import Column, Integer, String, DateTime, func
from database import Base
from sqlalchemy.dialects.mysql import JSON

class User(Base):
    """
    users 테이블과 매핑되는 SQLAlchemy ORM 모델
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    
    # 비밀번호는 평문이 아닌 해싱된 값으로 저장
    hashed_password = Column(String(255), nullable=False) 
    
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"

class Quiz(Base):
    __tablename__ = "quizzes"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    question = Column(String(500), nullable=False)  # 문제
    options = Column(JSON, nullable=False)  # ["옵션1", "옵션2", "옵션3", "옵션4"]
    correct_answer = Column(Integer, nullable=False)  # 정답 인덱스 (0, 1, 2, 3)
    explanation = Column(String(2000), nullable=True)  # 해설
    related_question = Column(String(500), nullable=False) # 유저의 연관 질문
    source_url = Column(String(500), nullable=True) # 출처 URL
    source_title = Column(String(200), nullable=True) # 출처 제목

    def __repr__(self):
        return f"<Quiz(id={self.id}, question='{self.question}')>"