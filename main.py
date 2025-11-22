import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed, before_log, after_log

from database import engine, Base 
from models.orm import User
from routers import query, admin, analysis, search, user
from services import langchain_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(
    title="FastAPI LangChain Application",
    description="A FastAPI application with LangChain and OpenAI integration",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(query.router)
app.include_router(admin.router)
app.include_router(analysis.router)
app.include_router(search.router)
app.include_router(user.router)

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting FastAPI LangChain Application...")

    # Initialize database
    initialize_database()

    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Initialize RAG system
    langchain_service.initialize_rag_system()

@retry(
    stop=stop_after_attempt(10),  # 최대 10번 시도
    wait=wait_fixed(3),           # 매 시도 사이에 3초 대기
    before=before_log(logger, logging.INFO),
    after=after_log(logger, logging.WARNING),
    retry_error_cls=RuntimeError, # 재시도 실패 시 최종적으로 발생시킬 에러 타입 지정
    reraise=True
)
def initialize_database():
    """
    DB에 필요한 테이블을 초기화합니다. 연결 실패 시 RuntimeError를 발생시켜 앱 시작을 중단합니다.
    """
    logger.info("Attempting to connect to the database and initialize tables...")
    try:
        # DB 연결 시도 및 테이블 생성
        Base.metadata.create_all(bind=engine)
        
        # 간단한 연결 테스트
        with engine.connect():
             logger.info("Successfully connected to the database and initialized tables.")
        
    except Exception as e:
        logger.critical(f"FATAL ERROR: Failed to connect to the database or initialize tables. Details: {e}")
        
        # DB 연결 실패는 치명적인 오류이므로, 앱 시작을 막기 위해 예외 발생
        raise RuntimeError("Database connection failed. Shutting down application.") from e


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "FastAPI LangChain Application",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "chat": "/api/chat",
            "query": "/api/query",
            "embed": "/api/embed",
            "reload": "/api/reload",
            "analyze": "/api/analyze",
            "analysis_status": "/api/analysis/status"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "rag_initialized": langchain_service.is_initialized()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)