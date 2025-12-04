# middleware/auth.py
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from jose import jwt, JWTError
from sqlalchemy.orm import Session

from config import SECRET_KEY, ALGORITHM

# 인증이 필요 없는 경로들
PUBLIC_PATHS = {
    "/",
    "/health",
    "/docs",
    "/openapi.json",
    "/users/signup",
    "/users/login",
}

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 1. 공개 경로는 인증 건너뛰기
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)
        
        # 2. Authorization 헤더에서 토큰 추출
        auth_header = request.headers.get("Authorization")
        
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            
            try:
                # 3. JWT 디코딩
                payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
                email = payload.get("sub")
                user_id = payload.get("user_id")
                
                if email and user_id:
                    # 4. request.state에 사용자 정보 저장
                    request.state.user_email = email
                    request.state.user_id = user_id
                    request.state.is_authenticated = True
                    
            except JWTError:
                # 토큰이 유효하지 않아도 일단 통과 (인증 필수 엔드포인트에서 처리)
                request.state.is_authenticated = False
        else:
            request.state.is_authenticated = False
        
        return await call_next(request)