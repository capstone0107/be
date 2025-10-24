"""
LangChain service for RAG system.
"""
import os
import logging
from typing import Optional, Dict, Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, TextLoader

logger = logging.getLogger(__name__)


class LangChainService:
    """Service class for managing LangChain RAG operations."""
    
    def __init__(self):
        self.vector_store: Optional[Chroma] = None
        self.qa_chain: Optional[RetrievalQA] = None
        
    def initialize_rag_system(self, data_dir: str = "data/") -> None:
        """
        Initialize the RAG system with documents from the data directory.
        
        Args:
            data_dir: Directory containing text documents
        """
        try:
            # Check if OpenAI API key is set
            if not os.getenv("OPENAI_API_KEY"):
                logger.warning("OPENAI_API_KEY not set. RAG system will not be initialized.")
                return
            
            # Load documents
            loader = DirectoryLoader(
                data_dir,
                glob="**/*.txt",
                loader_cls=TextLoader
            )
            documents = loader.load()
            
            if not documents:
                logger.warning(f"No documents found in {data_dir} directory.")
                return
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            
            # Create embeddings and vector store
            embeddings = OpenAIEmbeddings()
            self.vector_store = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory="./chroma_db"
            )
            
            # Create QA chain
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )
            
            logger.info("RAG system initialized successfully!")
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            raise
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system with a question.
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary with answer and source documents
            
        Raises:
            ValueError: If RAG system is not initialized
        """
        if not self.qa_chain:
            raise ValueError("RAG system not initialized")
        
        result = self.qa_chain({"query": question})
        
        # Extract source documents
        source_docs = []
        if "source_documents" in result:
            source_docs = [doc.page_content[:200] for doc in result["source_documents"]]
        
        return {
            "answer": result["result"],
            "source_documents": source_docs
        }
    
    def is_initialized(self) -> bool:
        """Check if the RAG system is initialized."""
        return self.qa_chain is not None
    
    def create_embedding(self, text: str) -> list:
        """
        Create embeddings for a given text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding
        """
        embeddings = OpenAIEmbeddings()
        return embeddings.embed_query(text)
    
    def chat(self, message: str, temperature: float = 0.7) -> str:
        """
        Direct chat with ChatGPT.
        
        Args:
            message: Message to send
            temperature: Temperature for response generation
            
        Returns:
            Response from ChatGPT
        """
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature)
        return llm.predict(message)


# Global service instance
langchain_service = LangChainService()
