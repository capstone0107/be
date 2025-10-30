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


# Conversation Analysis Models
class ConversationMessage(BaseModel):
    """A single message in a conversation."""
    role: str  # 'user' or 'assistant'
    content: str


class MetacognitiveInsight(BaseModel):
    """Metacognitive insight for a learning topic."""
    topic: str
    card_id: str  # CARD_TRADE_OFF, CARD_CONTEXT, CARD_PRECONDITION, CARD_EDGE_CASE
    search_keywords: List[str]


class ExternalVerification(BaseModel):
    """External verification with source."""
    topic: str
    summary: str  # Nuanced insight (max 200 chars)
    source: str  # URL to source
    follow_up_questions: List[str] = []


class AnalyzeConversationRequest(BaseModel):
    """Request to analyze a conversation."""
    messages: List[ConversationMessage]


class AnalyzeConversationResponse(BaseModel):
    """Response from conversation analysis."""
    overall_summary: str
    metacognitive_insights: List[MetacognitiveInsight]
    external_verifications: List[ExternalVerification]
    analyzed_at: Optional[str] = None
    message_count: Optional[int] = None
    error: Optional[str] = None