"""
FastAPI application with LangChain and OpenAI integration.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import LangChain components
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader, DirectoryLoader

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

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    
class QueryResponse(BaseModel):
    answer: str
    source_documents: Optional[List[str]] = None

class EmbeddingRequest(BaseModel):
    text: str

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    dimension: int

class ChatRequest(BaseModel):
    message: str
    
class ChatResponse(BaseModel):
    response: str

# Global variables for vector store
vector_store = None
qa_chain = None

def initialize_rag_system():
    """Initialize the RAG system with documents from the data directory."""
    global vector_store, qa_chain
    
    try:
        # Check if OpenAI API key is set
        if not os.getenv("OPENAI_API_KEY"):
            print("Warning: OPENAI_API_KEY not set. RAG system will not be initialized.")
            return
        
        # Load documents
        loader = DirectoryLoader(
            "data/",
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        documents = loader.load()
        
        if not documents:
            print("Warning: No documents found in data/ directory.")
            return
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        
        # Create QA chain
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        
        print("RAG system initialized successfully!")
    except Exception as e:
        print(f"Error initializing RAG system: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Initialize RAG system
    initialize_rag_system()

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
            "embed": "/api/embed"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "rag_initialized": qa_chain is not None
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint using OpenAI's ChatGPT.
    """
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        response = llm.predict(request.message)
        
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query documents using RAG (Retrieval Augmented Generation).
    """
    try:
        if not qa_chain:
            raise HTTPException(
                status_code=503,
                detail="RAG system not initialized. Please ensure documents are loaded."
            )
        
        result = qa_chain({"query": request.question})
        
        # Extract source documents
        source_docs = []
        if "source_documents" in result:
            source_docs = [doc.page_content[:200] for doc in result["source_documents"]]
        
        return QueryResponse(
            answer=result["result"],
            source_documents=source_docs
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/embed", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest):
    """
    Create embeddings for a given text using OpenAI embeddings.
    """
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        embeddings = OpenAIEmbeddings()
        embedding_vector = embeddings.embed_query(request.text)
        
        return EmbeddingResponse(
            embedding=embedding_vector,
            dimension=len(embedding_vector)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reload")
async def reload_documents():
    """
    Reload documents and reinitialize the RAG system.
    """
    try:
        initialize_rag_system()
        return {
            "status": "success",
            "message": "Documents reloaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
