from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from jose import jwt, JWTError

from database import get_db
from models.user import UserCreate, UserLogin, UserResponse, Token
from models.orm import User as DBUser # ORM 모델
from services import user_service # 핵심 서비스 로직
from config import SECRET_KEY, ALGORITHM

router = APIRouter(prefix="/users", tags=["users"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="users/token") # 로그인 엔드포인트 URL

# --- 1. JWT 유효성 검사 및 현재 사용자 조회 의존성 ---
def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)) -> DBUser:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="인증 정보를 확인할 수 없습니다.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # 1. JWT 디코딩
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    # 2. DB에서 사용자 조회
    user = user_service.get_user_by_email(db, email=email)
    if user is None:
        raise credentials_exception
    return user

# --- 2. 회원가입 (POST /register) ---
@router.post("/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED, summary="사용자 회원가입")
def register_user_endpoint(user: UserCreate, db: Session = Depends(get_db)):
    db_user = user_service.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="이미 등록된 이메일 주소입니다.")
    return user_service.create_user(db=db, user=user)

# --- 3. 로그인 및 JWT 발급 (POST /token) ---
@router.post("/login", response_model=Token, summary="이메일/비밀번호로 로그인하여 JWT 발급")
def login_for_access_token_endpoint(user_data: UserLogin, db: Session = Depends(get_db)):
    user = user_service.get_user_by_email(db, email=user_data.email)
    
    # 1. 사용자 존재 및 비밀번호 확인
    if not user or not user_service.verify_password(user_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="이메일 또는 비밀번호가 올바르지 않습니다.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 2. JWT 생성
    access_token = user_service.create_access_token(
        data={"sub": user.email, "user_id": user.id}
    )
    return {"access_token": access_token, "token_type": "bearer"}

# --- 4. JWT를 통한 로그인 확인 (GET /me) ---
@router.get("/me", response_model=UserResponse, summary="JWT를 통해 현재 로그인 사용자 정보 확인")
def read_users_me_endpoint(current_user: Annotated[DBUser, Depends(get_current_user)]):
    """유효한 JWT 토큰이 있어야 접근 가능합니다."""
    return current_user