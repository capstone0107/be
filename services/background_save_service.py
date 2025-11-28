import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from services.focus_service import focus_service
from database import SessionLocal

logger = logging.getLogger(__name__)


class BackgroundConversationService:
    """Service for saving conversations in the background."""
    
    def __init__(self):
        self.save_queue = asyncio.Queue()
        self.is_processing = False
    
    async def save_conversation_async(
        self,
        conversation_id: str,
        messages: List[Dict[str, Any]],
        auto_title: Optional[str] = None
    ):
        """
        Save conversation in the background without blocking the response.
        
        Args:
            conversation_id: Conversation ID
            messages: List of messages
            auto_title: Auto-generated title (optional)
        """
        try:
            logger.info(f"Background save initiated for conversation {conversation_id}")
            
            # Create database session
            db = SessionLocal()
            
            try:
                # Step 1: Classify conversation
                # logger.info(f"Classifying conversation {conversation_id}...")
                # classification_result = focus_service.classify_conversation(
                #     conversation_id=conversation_id,
                #     messages=messages
                # )
                
                # # Check for errors
                # if "error" in classification_result:
                #     logger.error(f"Classification failed: {classification_result.get('message')}")
                #     return
                
                # # Step 2: Generate title if not provided
                # if not auto_title:
                #     auto_title = self._generate_title(messages, classification_result)
                
                # Step 3: Save to database
                logger.info(f"Saving conversation {conversation_id} to database...")
                conversation = focus_service.save_conversation_with_focuses(
                    conversation_id=conversation_id,
                    # title=auto_title,
                    messages=messages,
                    # classification_result=None,
                    db=db
                )
                
                # logger.info(
                #     f"✅ Successfully saved conversation {conversation_id} "
                #     f"with {len(messages)} messages and "
                #     f"{len(classification_result.get('focuses', []))} focuses"
                # )
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Background save failed for {conversation_id}: {e}")
            # Don't raise - this is background task
    
    def _generate_title(
        self,
        messages: List[Dict[str, Any]],
        classification_result: Dict[str, Any]
    ) -> str:
        """
        Generate auto title from conversation.
        
        Args:
            messages: Message list
            classification_result: Classification result
            
        Returns:
            Generated title
        """
        # Use conversation summary if available
        summary = classification_result.get("conversation_summary")
        if summary:
            return summary
        
        # Use first user message as fallback
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                # Truncate to 50 characters
                return content[:50] + "..." if len(content) > 50 else content
        
        # Default
        return f"대화 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    def save_in_background(
        self,
        conversation_id: str,
        messages: List[Dict[str, Any]],
        auto_title: Optional[str] = None
    ):
        """
        Queue a conversation for background saving.
        
        This is a synchronous wrapper that creates a background task.
        
        Args:
            conversation_id: Conversation ID
            messages: List of messages
            auto_title: Auto-generated title (optional)
        """
        # Create background task (fire and forget)
        asyncio.create_task(
            self.save_conversation_async(
                conversation_id=conversation_id,
                messages=messages,
                auto_title=auto_title
            )
        )
        
        logger.info(f"Queued conversation {conversation_id} for background save")


# Global service instance
background_save_service = BackgroundConversationService()