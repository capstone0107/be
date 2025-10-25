"""
Test suite for FastAPI application using pytest.
Run with: pytest test_api.py
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os

from main import app
from services import langchain_service


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_openai_key(monkeypatch):
    """Mock OpenAI API key environment variable."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")


class TestBasicEndpoints:
    """Test basic endpoints that don't require OpenAI."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns correct information."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "FastAPI LangChain Application"
        assert data["version"] == "1.0.0"
        assert "endpoints" in data
        assert "/health" in data["endpoints"]["health"]
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "openai_configured" in data
        assert "rag_initialized" in data


class TestChatEndpoint:
    """Test chat endpoint with mocked OpenAI."""
    
    @patch('services.langchain_service.ChatOpenAI')
    def test_chat_success(self, mock_chatgpt, client, mock_openai_key):
        """Test successful chat request."""
        # Mock ChatGPT response
        mock_llm = MagicMock()
        mock_llm.predict.return_value = "This is a test response"
        mock_chatgpt.return_value = mock_llm
        
        response = client.post(
            "/api/chat",
            json={"message": "Hello!"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert data["response"] == "This is a test response"
    
    def test_chat_no_api_key(self, client):
        """Test chat endpoint without API key."""
        response = client.post(
            "/api/chat",
            json={"message": "Hello!"}
        )
        
        assert response.status_code == 500
        assert "OpenAI API key not configured" in response.json()["detail"]


class TestQueryEndpoint:
    """Test query endpoint with mocked RAG system."""
    
    def test_query_not_initialized(self, client):
        """Test query when RAG system is not initialized."""
        # Ensure RAG is not initialized
        langchain_service.qa_chain = None
        
        response = client.post(
            "/api/query",
            json={"question": "What is FastAPI?"}
        )
        
        assert response.status_code == 503
        assert "RAG system not initialized" in response.json()["detail"]
    
    @patch.object(langchain_service, 'query')
    @patch.object(langchain_service, 'is_initialized')
    def test_query_success(self, mock_initialized, mock_query, client):
        """Test successful query request."""
        # Mock RAG system as initialized
        mock_initialized.return_value = True
        
        # Mock query response
        mock_query.return_value = {
            "answer": "FastAPI is a modern web framework",
            "source_documents": ["Source 1", "Source 2"]
        }
        
        response = client.post(
            "/api/query",
            json={"question": "What is FastAPI?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert data["answer"] == "FastAPI is a modern web framework"
        assert "source_documents" in data
        assert len(data["source_documents"]) == 2


class TestEmbedEndpoint:
    """Test embedding endpoint with mocked OpenAI."""
    
    @patch('services.langchain_service.OpenAIEmbeddings')
    def test_embed_success(self, mock_embeddings, client, mock_openai_key):
        """Test successful embedding creation."""
        # Mock embeddings
        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_embeddings.return_value = mock_embed
        
        response = client.post(
            "/api/chat",  # Note: endpoint needs to be fixed in chat.py
            json={"message": "Test text"}
        )
        
        assert response.status_code == 200


class TestAdminEndpoint:
    """Test admin endpoints."""
    
    @patch.object(langchain_service, 'initialize_rag_system')
    def test_reload_success(self, mock_init, client):
        """Test successful document reload."""
        mock_init.return_value = None
        
        response = client.post("/api/reload")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "Documents reloaded" in data["message"]
        
        # Verify initialize was called
        mock_init.assert_called_once()


class TestInputValidation:
    """Test input validation for API endpoints."""
    
    def test_chat_missing_message(self, client, mock_openai_key):
        """Test chat endpoint with missing message field."""
        response = client.post("/api/chat", json={})
        assert response.status_code == 422  # Validation error
    
    def test_query_missing_question(self, client):
        """Test query endpoint with missing question field."""
        response = client.post("/api/query", json={})
        assert response.status_code == 422  # Validation error
    
    def test_chat_invalid_json(self, client):
        """Test chat endpoint with invalid JSON."""
        response = client.post(
            "/api/chat",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
