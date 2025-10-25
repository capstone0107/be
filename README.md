# FastAPI LangChain Application

A FastAPI application with LangChain and OpenAI integration for building RAG (Retrieval Augmented Generation) systems.

## Features

- **FastAPI**: Modern, fast web framework for building APIs
- **LangChain**: Framework for developing applications with LLMs
- **OpenAI Integration**: ChatGPT and embeddings support
- **RAG System**: Document-based question answering
- **Vector Store**: Chroma for efficient document retrieval
- **Docker Support**: Full containerization with Docker and docker-compose

## Project Structure

```
.
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose configuration
├── data/                  # Sample text files for RAG
│   ├── fastapi_info.txt
│   ├── langchain_info.txt
│   └── openai_info.txt
└── chroma_db/            # Vector database (generated)
```

## Prerequisites

- Python 3.11+
- OpenAI API key
- Docker and Docker Compose (optional)

## Installation

### Local Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd be
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

5. Run the application:
```bash
uvicorn main:app --reload
```

The application will be available at `http://localhost:8000`

### Docker Setup

1. Configure environment variables:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

2. Build and run with Docker Compose:
```bash
docker-compose up --build
```

The application will be available at `http://localhost:8000`

## API Endpoints

### Health Check
```
GET /health
```
Check application health and configuration status.

### Root
```
GET /
```
Get API information and available endpoints.

### Chat
```
POST /api/chat
Content-Type: application/json

{
  "message": "Hello! Tell me about FastAPI."
}
```
Chat with OpenAI's ChatGPT.

### Query Documents (RAG)
```
POST /api/query
Content-Type: application/json

{
  "question": "What is FastAPI?"
}
```
Query documents using Retrieval Augmented Generation.

### Create Embeddings
```
POST /api/embed
Content-Type: application/json

{
  "text": "This is a sample text to embed."
}
```
Create embeddings for given text using OpenAI.

### Reload Documents
```
POST /api/reload
```
Reload documents and reinitialize the RAG system.

## Interactive API Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Usage Examples

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is LangChain?"}'

# Query documents
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key features of FastAPI?"}'

# Create embeddings
curl -X POST http://localhost:8000/api/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'
```

### Using Python

```python
import requests

# Chat endpoint
response = requests.post(
    "http://localhost:8000/api/chat",
    json={"message": "Tell me about OpenAI"}
)
print(response.json())

# Query endpoint
response = requests.post(
    "http://localhost:8000/api/query",
    json={"question": "What is RAG?"}
)
print(response.json())
```

## Adding Your Own Documents

1. Add `.txt` files to the `data/` directory
2. Call the reload endpoint or restart the application:
```bash
curl -X POST http://localhost:8000/api/reload
```

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `APP_NAME`: Application name (optional)
- `DEBUG`: Enable debug mode (optional)

## Technologies Used

- **FastAPI**: Web framework
- **LangChain**: LLM application framework
- **OpenAI**: GPT models and embeddings
- **ChromaDB**: Vector database
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation

## Development

### Running Tests
```bash
# Add tests as needed
pytest
```

### Code Formatting
```bash
# Install development dependencies
pip install black isort

# Format code
black .
isort .
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
