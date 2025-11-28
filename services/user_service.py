from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from jose import jwt
import bcrypt

from models.orm import User as DBUser # ORM 모델
from models.user import UserCreate
from config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES

# 비밀번호 해싱 컨텍스트
UNLIMITED_EXPIRY_DAYS = 365 * 100

# --- 1. 비밀번호 유틸리티 ---
def get_password_hash(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

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