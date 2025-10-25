"""
LangChain service for RAG system.
"""
import os
import logging
from typing import Optional, Dict, Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import DirectoryLoader, TextLoader

logger = logging.getLogger(__name__)


class LangChainService:
    """Service class for managing LangChain RAG operations."""
    
    def __init__(self):
        self.vector_store: Optional[Chroma] = None
        self.retriever = None
        self.llm = None
        
        # 커스텀 프롬프트 설정
        self.prompt = ChatPromptTemplate.from_template("""
            당신은 친절한 AI 어시스턴트입니다.

            아래 문서를 참고하여 질문에 답변하세요.
            문서에 관련 정보가 있으면 문서 기반으로 답변하고, 외부 출처(검색 등)를 참고하지 마세요.
            문서 내용이 있다면, 지식 카드를 만들어 답변에 포함해야 합니다.
            답변에는 답변(answer), 요약(summary), 출처 URL(source) 섹션이 JSON 형식으로 포함되어야 합니다.
            지식 카드의 summary와 answer 부분에는 최소 약 40글자의 내용과 필요에 따라 수식이 포함되어야합니다.
            지식카드의 출처가 최대한 겹치지 않게 문서 내용을 분류하고 분리하여 정리해야합니다. 지식카드의 개수는 최소 1개, 최대 3개가 될 수 있습니다.

            반드시 다음 JSON 형식으로만 답변하세요:
            {{
                "answer": "여기에 전체 답변이 들어갑니다.",
                "cards": [
                    {{
                        "summary": "첫 번째 출처의 내용 요약 (최소 40글자)",
                        "source": "첫 번째 문서에 포함된 정확한 출처 URL"
                    }},
                    {{
                        "summary": "두 번째 출처의 내용 요약 (최소 40글자)",
                        "source": "두 번째 문서에 포함된 정확한 출처 URL (첫 번째와 달라야 함)"
                    }},
                    {{
                        "summary": "세 번째 출처의 내용 요약 (최소 40글자)",
                        "source": "세 번째 문서에 포함된 정확한 출처 URL (위 두 개와 달라야 함)"
                    }}
                ]
            }}
            
            문서:
            {context}

            질문: {question}

            JSON 응답:
        """)
        
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

            # Create embeddings and vector store
            if(os.path.exists("./chroma_db")):
                self.vector_store = Chroma(
                    persist_directory="./chroma_db",
                    embedding_function=OpenAIEmbeddings()
                )
                logger.info("기존의 벡터 DB 로드")
            else: 
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
                embeddings = OpenAIEmbeddings()

                self.vector_store = Chroma.from_documents(
                    documents=splits,
                    embedding=embeddings, # 임베딩 객체 전달
                    persist_directory="./chroma_db"
                )
            
            # Create retriever
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 3}
            )
            
            # Create LLM
            qa_model = os.getenv("QA_MODEL", "gpt-3.5-turbo")
            self.llm = ChatOpenAI(model_name=qa_model, temperature=0)
            
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
        if not self.retriever or not self.llm:
            raise ValueError("RAG system not initialized")
        
        # Helper function to format documents
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Create RAG chain using LCEL
        rag_chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Get answer from RAG chain
        answer = rag_chain.invoke(question)
        
        # Get source documents separately
        source_docs = self.retriever.get_relevant_documents(question)
        source_contents = [doc.page_content[:200] for doc in source_docs]
        
        return {
            "answer": answer,
            "source_documents": source_contents
        }
    
    def is_initialized(self) -> bool:
        """Check if the RAG system is initialized."""
        return self.retriever is not None and self.llm is not None
    
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
        llm_model = os.getenv("CHAT_MODEL", "gpt-3.5-turbo")
        llm = ChatOpenAI(model_name=llm_model, temperature=temperature)
        return llm.predict(message)


# Global service instance
langchain_service = LangChainService()