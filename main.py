import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from routers import query, admin, analysis
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

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting FastAPI LangChain Application...")
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Initialize RAG system
    langchain_service.initialize_rag_system()


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