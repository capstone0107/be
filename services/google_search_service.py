"""
Google Search service for external web search.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)


class GoogleSearchResult:
    """Google search result model"""
    def __init__(self, title: str, link: str, snippet: str):
        self.title = title
        self.link = link
        self.snippet = snippet
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "title": self.title,
            "link": self.link,
            "snippet": self.snippet
        }


class GoogleSearchService:
    """Service for Google Custom Search API"""
    
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.cse_id = os.getenv("GOOGLE_CSE_ID")
        self.service = None
        
        if self.api_key and self.cse_id:
            try:
                self.service = build("customsearch", "v1", developerKey=self.api_key)
                logger.info("Google Search service initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Google Search service: {e}")
        else:
            logger.warning("Google Search credentials not configured (GOOGLE_API_KEY or GOOGLE_CSE_ID missing)")
    
    def is_available(self) -> bool:
        """Check if Google Search is available"""
        return self.service is not None
    
    def search(self, query: str, num_results: int = 5) -> List[GoogleSearchResult]:
        """
        Perform Google search
        
        Args:
            query: Search query
            num_results: Number of results to return (max 10)
            
        Returns:
            List of search results
        """
        if not self.is_available():
            logger.warning("Google Search service not available")
            return []
        
        try:
            # Execute search
            result = self.service.cse().list(
                q=query,
                cx=self.cse_id,
                num=min(num_results, 10)  # API limit is 10
            ).execute()
            
            # Parse results
            search_results = []
            if "items" in result:
                for item in result["items"]:
                    search_results.append(
                        GoogleSearchResult(
                            title=item.get("title", ""),
                            link=item.get("link", ""),
                            snippet=item.get("snippet", "")
                        )
                    )
            
            logger.info(f"Found {len(search_results)} results for query: {query}")
            return search_results
            
        except HttpError as e:
            logger.error(f"Google Search API error: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")
            return []


# Global service instance
google_search_service = GoogleSearchService()