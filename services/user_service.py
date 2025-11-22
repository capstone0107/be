from datetime import datetime, timedelta, timezone
from typing import Optional
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import jwt, JWTError

from models.orm import User as DBUser # ORM 모델
from models.user import UserCreate
from config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES

# 비밀번호 해싱 컨텍스트
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
UNLIMITED_EXPIRY_DAYS = 365 * 100

# --- 1. 비밀번호 유틸리티 ---
def get_password_hash(password: str) -> str:
    """평문 비밀번호를 해싱합니다."""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """평문 비밀번호와 해싱된 비밀번호를 비교합니다."""
    return pwd_context.verify(plain_password, hashed_password)

# --- 2. JWT 생성 ---
def create_access_token(data: dict):
    to_encode = data.copy()
    
    if ACCESS_TOKEN_EXPIRE_MINUTES is None:
        expire = datetime.now(timezone.utc) + timedelta(days=UNLIMITED_EXPIRY_DAYS)
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
    to_encode.update({"exp": expire.timestamp()})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# --- 3. CRUD 로직 ---
def get_user_by_email(db: Session, email: str):
    return db.query(DBUser).filter(DBUser.email == email).first()

def create_user(db: Session, user: UserCreate):
    # 비밀번호 해시 및 저장
    hashed_password = get_password_hash(user.password)
    db_user = DBUser(
        email=user.email,
        username=user.username,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user