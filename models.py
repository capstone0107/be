"""
Pydantic models for request and response validation.
"""
from pydantic import BaseModel
from typing import List, Optional


class QueryRequest(BaseModel):
    question: List[str]
    
class KnowledgeCard(BaseModel):
    summary: str
    source: str

class QueryResponse(BaseModel):
    answer: str
    cards: List[KnowledgeCard] = []

class EmbeddingRequest(BaseModel):
    text: str


class EmbeddingResponse(BaseModel):
    embedding: List[float]
    dimension: int

class ReloadResponse(BaseModel):
    status: str
    message: str
