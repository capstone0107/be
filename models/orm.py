from sqlalchemy import Column, Integer, String, DateTime, func
from database import Base 

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