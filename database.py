import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Docker Compose 환경 변수에서 DB 연결 정보 로드
# 예시: mysql+pymysql://user:password@mysql-db/app_db
DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    # database.py가 로드될 때 환경 변수가 없으면 즉시 오류 발생
    raise ValueError("FATAL: DATABASE_URL environment variable is not set. Check .env file or environment.")

# SQLAlchemy 엔진 생성
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True # 연결이 끊어졌는지 주기적으로 확인하여 재연결 시도
)

# DB 세션 생성기
# autocommit=False: 트랜잭션 수동 커밋/롤백
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 모든 ORM 모델의 기본 클래스
Base = declarative_base()

def get_db():
    """
    FastAPI 의존성 주입(Depends)을 위한 DB 세션 제너레이터.
    요청마다 새로운 세션을 생성하고, 요청 완료 시 세션을 안전하게 닫습니다.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()