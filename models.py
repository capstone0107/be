"""
Pydantic models for request and response validation.
"""
from pydantic import BaseModel
from typing import List, Optional


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


class ReloadResponse(BaseModel):
    status: str
    message: str
