# FastAPI + LangChain + OpenAI Project

## Project Overview

This project is a complete FastAPI application integrated with LangChain and OpenAI for building RAG (Retrieval Augmented Generation) systems.

## What's Included

### Core Application Files
- **main.py**: Complete FastAPI application with:
  - Basic endpoints (/, /health)
  - Chat endpoint (/api/chat) using OpenAI ChatGPT
  - RAG query endpoint (/api/query) for document-based Q&A
  - Embedding endpoint (/api/embed) for text embeddings
  - Document reload endpoint (/api/reload)
  - Full CORS support
  - Automatic RAG initialization on startup

### Configuration Files
- **requirements.txt**: All necessary Python dependencies including:
  - FastAPI and Uvicorn
  - LangChain (core, openai, and community packages)
  - OpenAI client
  - ChromaDB for vector storage
  - Document loaders (pypdf, python-multipart)
  - Environment variable handling

- **.env.example**: Template for environment configuration
  - OPENAI_API_KEY (required for OpenAI features)
  - APP_NAME
  - DEBUG mode

- **.gitignore**: Proper Python .gitignore including:
  - Python cache files
  - Virtual environments
  - .env files
  - Vector database directory
  - IDE files

### Docker Configuration
- **Dockerfile**: Multi-stage Docker build with:
  - Python 3.11 slim base image
  - Optimized layer caching
  - Security best practices
  - Proper working directory setup

- **docker-compose.yml**: Complete orchestration with:
  - Port mapping (8000:8000)
  - Environment variable passing
  - Volume mounts for data and vector DB
  - Hot reload support
  - Restart policies

### Sample Data
Three comprehensive text files in the `data/` directory:
- **fastapi_info.txt**: FastAPI framework documentation
- **langchain_info.txt**: LangChain concepts and RAG implementation
- **openai_info.txt**: OpenAI API usage and best practices

These files serve as the knowledge base for the RAG system.

### Example Scripts
- **example_rag.py**: Standalone script demonstrating:
  - Document loading from the data directory
  - Text splitting and chunking
  - Embedding creation
  - Vector store initialization
  - RAG chain setup
  - Sample queries

- **test_api.py**: API testing script for:
  - Testing all endpoints
  - Verifying server health
  - Demonstrating API usage

### Documentation
- **README.md**: Comprehensive documentation including:
  - Feature overview
  - Installation instructions (local and Docker)
  - API endpoint descriptions
  - Usage examples (curl and Python)
  - Development guidelines

## Key Features Implemented

### 1. FastAPI Application Structure
- Proper application factory pattern
- CORS middleware configuration
- Pydantic models for request/response validation
- Error handling with HTTP exceptions
- Async/await support

### 2. LangChain RAG Implementation
- Document loading from directory
- Text splitting with configurable chunk size
- OpenAI embeddings integration
- ChromaDB vector store
- RetrievalQA chain with source document tracking
- Automatic initialization on startup

### 3. OpenAI Integration
- ChatGPT endpoint for conversational AI
- Embedding generation endpoint
- Proper API key configuration
- Error handling for missing credentials

### 4. Containerization
- Production-ready Dockerfile
- Docker Compose for easy deployment
- Volume mounts for data persistence
- Environment variable configuration
- Hot reload support for development

### 5. Developer Experience
- Clear code structure and comments
- Type hints throughout
- Example scripts for learning
- Test script for verification
- Comprehensive README

## How to Use

### Quick Start (Docker)
1. Copy `.env.example` to `.env` and add your OpenAI API key
2. Run `docker compose up --build`
3. Access the API at http://localhost:8000
4. View interactive docs at http://localhost:8000/docs

### Local Development
1. Create virtual environment: `python -m venv venv`
2. Activate: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`
4. Configure `.env` file with OpenAI API key
5. Run: `uvicorn main:app --reload`

### Testing
1. Start the server
2. Run: `python test_api.py`
3. Or use the interactive docs at /docs

### RAG System
The RAG system automatically:
1. Loads all .txt files from the `data/` directory
2. Splits them into chunks
3. Creates embeddings using OpenAI
4. Stores them in ChromaDB
5. Enables semantic search for Q&A

To query the system:
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is FastAPI?"}'
```

## Architecture

```
User Request
    ↓
FastAPI Endpoint
    ↓
LangChain RAG Chain
    ↓
├─→ Retriever (ChromaDB) → Relevant Documents
│                              ↓
└─→ ChatGPT (OpenAI) ←────────┘
    ↓
Response with Sources
```

## Dependencies Explained

- **fastapi**: Modern web framework
- **uvicorn**: ASGI server
- **pydantic**: Data validation
- **langchain**: LLM framework
- **langchain-openai**: OpenAI integration for LangChain
- **langchain-community**: Community-contributed components
- **openai**: OpenAI Python client
- **chromadb**: Vector database
- **tiktoken**: Token counting for OpenAI models
- **pypdf**: PDF document loading
- **python-dotenv**: Environment variable management

## Security Considerations

- API keys stored in environment variables
- .env file excluded from git
- No hardcoded secrets
- CORS configured (should be restricted in production)
- Proper error handling without exposing internals

## Next Steps

To extend this project, consider:
1. Adding authentication/authorization
2. Implementing rate limiting
3. Adding more document loaders (PDF, CSV, etc.)
4. Database integration for user management
5. Caching layer for embeddings
6. Logging and monitoring
7. Unit tests and integration tests
8. CI/CD pipeline

## Troubleshooting

### "OpenAI API key not configured"
- Ensure OPENAI_API_KEY is set in your .env file
- Restart the application after setting the key

### "RAG system not initialized"
- Check that the data/ directory contains .txt files
- Verify the OpenAI API key is valid
- Check application logs for initialization errors

### Docker build fails
- Ensure Docker is running
- Check network connectivity
- Try building without cache: `docker compose build --no-cache`

## Project Status

✅ All requirements met:
- ✅ Basic FastAPI application with sample endpoints
- ✅ LangChain RAG implementation using OpenAI embeddings and ChatGPT
- ✅ Sample document loader and embedding examples
- ✅ Example text files for document processing
- ✅ Dockerfile and docker-compose.yml for containerization
- ✅ requirements.txt with all dependencies
- ✅ .env.example for configuration
- ✅ Comprehensive documentation
- ✅ Example scripts and testing utilities
