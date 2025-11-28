from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from datetime import datetime

# --- 요청 모델 ---

class UserCreate(BaseModel):
    """사용자 생성 요청 시 사용 (평문 비밀번호 포함)"""
    username: str = Field(..., max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8) # 평문 비밀번호


class UserLogin(BaseModel):
    """사용자 로그인 요청 시 사용"""
    email: EmailStr
    password: str

# --- 응답 모델 ---
class UserResponse(BaseModel):
    """사용자 정보를 응답할 때 사용 (비밀번호, 해시 값 제외)"""
    id: int
    username: str
    email: EmailStr
    created_at: datetime
    updated_at: datetime
    
    model_config = {
        "from_attributes": True
    }

class Token(BaseModel):
    """JWT 토큰 응답 시 사용"""
    access_token: str
    token_type: str = "bearer"