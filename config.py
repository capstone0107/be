import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

# JWT 보안 설정
SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-default-key")
ALGORITHM = "HS256"

# 만료 시간 설정 (None 또는 0일 때 무제한 처리)
expire_minutes_str = os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES")
if not expire_minutes_str or expire_minutes_str == '0':
    ACCESS_TOKEN_EXPIRE_MINUTES: Optional[int] = None # None: 무제한
else:
    try:
        ACCESS_TOKEN_EXPIRE_MINUTES = int(expire_minutes_str)
    except ValueError:
        ACCESS_TOKEN_EXPIRE_MINUTES = None