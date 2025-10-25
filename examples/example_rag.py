"""
Example script demonstrating how to use the LangChain RAG system directly.
This can be run independently of the FastAPI server.
"""
import os
import sys
from dotenv import load_dotenv
import logging

# Add parent directory to path to import services
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.langchain_service import LangChainService

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__) # Create a logger for this module


def main():
    """Main function to demonstrate LangChain RAG."""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set in environment variables.")
        print("Please create a .env file with your OpenAI API key.")
        return
    
    # Initialize service
    service = LangChainService()
    
    print("Loading documents from data/ directory...")
    service.initialize_rag_system()
    
    if not service.is_initialized():
        print("Failed to initialize RAG system.")
        return
    
    print("\n" + "="*50)
    print("RAG System Ready!")
    print("="*50 + "\n")
    
    # Example queries
    queries = [
        "운영체제 학습을 위해 내용 자세히 알려줘.",
    ]
    
    for question in queries:
        print(f"Question: {question}")
        result = service.query(question)

        print(f"Answer: {result['answer']}")
        print(result['source_documents'][2])
        print(f"Sources: {len(result['source_documents'])} documents retrieved")
        print("-" * 50 + "\n")


if __name__ == "__main__":
    main()
