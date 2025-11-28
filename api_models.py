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

class GraphNode(BaseModel):
    """A node in the conversation graph."""
    id: str
    label: str
    description: str  # The "brief sentence" explaining the node
    category: Optional[str] = None # e.g., "Concept", "Problem", "Solution"
    related_message_indices: List[int] = [] # To link back to chat
    conversation_id: Optional[str] = None # For combined graphs (color coding)

class GraphEdge(BaseModel):
    """An edge connecting two nodes."""
    source: str
    target: str
    label: str # e.g., "LEADS_TO", "SOLVES_PROBLEM"

class GraphData(BaseModel):
    """The complete graph structure."""
    nodes: List[GraphNode]
    edges: List[GraphEdge]

class CombinedGraphRequest(BaseModel):
    """Request to combine multiple existing graphs."""
    graphs: List[GraphData]

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