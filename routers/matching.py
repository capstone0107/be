"""
Matching router for LLM-based presentation-note matching.
"""
import json
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
from typing import Optional

from services.matching_service import matching_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/matching", tags=["matching"])


def load_mock_data():
    """Load mock presentation and notes data."""
    try:
        mock_dir = Path("mock_data")
        
        # Load presentation
        with open(mock_dir / "presentation.json", "r", encoding="utf-8") as f:
            presentation = json.load(f)
        
        # Load notes
        with open(mock_dir / "notes.json", "r", encoding="utf-8") as f:
            notes = json.load(f)
        
        return presentation, notes
    
    except FileNotFoundError as e:
        logger.error(f"Mock data file not found: {e}")
        raise HTTPException(
            status_code=404,
            detail="Mock data files not found. Ensure mock_data/ directory exists."
        )
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse mock data: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to parse mock data JSON files."
        )


@router.get("/status")
async def get_matching_status():
    """
    Check if the matching service is available.
    
    Returns:
        Status of the matching service
    """
    return {
        "available": matching_service.is_available(),
        "service": "llm_matching",
        "model": "gpt-4o-mini"
    }


@router.post("/test")
async def test_matching(section_id: Optional[str] = None):
    """
    Test LLM-based matching with mock data.
    
    Args:
        section_id: Optional section ID to match. If not provided, matches all sections.
        
    Returns:
        Matching results for the requested section(s)
    """
    try:
        if not matching_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="Matching service not available. Check OpenAI API key."
            )
        
        # Load mock data
        presentation, notes = load_mock_data()
        
        # If specific section requested
        if section_id:
            section = next(
                (s for s in presentation["sections"] if s["section_id"] == section_id),
                None
            )
            
            if not section:
                raise HTTPException(
                    status_code=404,
                    detail=f"Section {section_id} not found"
                )
            
            logger.info(f"Matching section {section_id}: {section['title']}")
            matches = matching_service.match_section_to_all_notes(section, notes)
            
            # Categorize results
            high = [m for m in matches if m.get("relevance") == "높음"]
            medium = [m for m in matches if m.get("relevance") == "중간"]
            low = [m for m in matches if m.get("relevance") == "낮음"]
            none = [m for m in matches if m.get("relevance") == "없음"]
            
            return {
                "section_id": section["section_id"],
                "section_title": section["title"],
                "total_notes_tested": len(notes),
                "matches": {
                    "high": high,
                    "medium": medium,
                    "low": low,
                    "none": none
                },
                "summary": {
                    "high_count": len(high),
                    "medium_count": len(medium),
                    "low_count": len(low),
                    "none_count": len(none)
                }
            }
        
        # Match all sections
        else:
            logger.info("Matching all sections")
            results = matching_service.match_presentation_to_notes(presentation, notes)
            return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in test matching: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/mock-data")
async def get_mock_data():
    """
    Get the mock data (presentation and notes) for reference.
    
    Returns:
        Mock presentation and notes data
    """
    try:
        presentation, notes = load_mock_data()
        return {
            "presentation": presentation,
            "notes": notes,
            "stats": {
                "total_sections": len(presentation.get("sections", [])),
                "total_notes": len(notes),
                "notes_by_author": {
                    "준": len([n for n in notes if n["author"] == "준"]),
                    "민수": len([n for n in notes if n["author"] == "민수"]),
                    "수진": len([n for n in notes if n["author"] == "수진"])
                }
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting mock data: {e}")
        raise HTTPException(status_code=500, detail=str(e))