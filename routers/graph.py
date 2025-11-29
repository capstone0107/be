"""
Graph router for handling knowledge graph generation and merging.
"""
import logging
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any

# Import the new graph-related models
from api_models import AnalyzeConversationRequest, CombinedGraphRequest, GraphData
# Import the NEW graph_service
from services.graph_service import graph_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/graph", tags=["graph"])


@router.post("", response_model=GraphData)
async def generate_conversation_graph(request: AnalyzeConversationRequest):
    """
    Generates a knowledge graph for a SINGLE conversation.
    This uses the new graph_service to analyze messages and return nodes/edges.
    """
    try:
        if not graph_service.is_available():
            raise HTTPException(status_code=503, detail="Graph service unavailable (Check OpenAI Key)")

        # Convert Pydantic models to dicts for the service
        messages = [msg.dict() for msg in request.messages]
        
        # Call the new service
        result = graph_service.generate_graph(messages)

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        # Extract the 'graph' part of the response
        graph_data = result.get("graph", {"nodes": [], "edges": []})
        
        return GraphData(**graph_data)

    except Exception as e:
        logger.error(f"Error generating graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/combined", response_model=Dict[str, Any])
async def generate_combined_graph(request: CombinedGraphRequest):
    """
    Generates a combined knowledge graph from MULTIPLE graph summaries.
    """
    try:
        if not graph_service.is_available():
            raise HTTPException(status_code=503, detail="Graph service unavailable")
        
        if not request.graphs:
            return {"nodes": [], "edges": []}

        # Prepare data for the service
        graphs_data = [g.dict() for g in request.graphs]

        # Call the new service's combined graph method
        combined_result = graph_service.generate_combined_graph(graphs_data)

        if "error" in combined_result:
            raise HTTPException(status_code=500, detail=combined_result["error"])

        return combined_result

    except Exception as e:
        logger.error(f"Error generating combined graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))