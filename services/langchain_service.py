"""
LangChain service for RAG system.
"""
import os
import re
import logging
from typing import Optional, Dict, Any, List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.schema import Document

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
            {{{{
                "answer": "여기에 전체 답변이 들어갑니다.",
                "cards": [
                    {{{{
                        "summary": "첫 번째 출처의 내용 요약 (최소 40글자)",
                        "source": "첫 번째 문서에 포함된 정확한 출처 URL"
                    }}}},
                    {{{{
                        "summary": "두 번째 출처의 내용 요약 (최소 40글자)",
                        "source": "두 번째 문서에 포함된 정확한 출처 URL (첫 번째와 달라야 함)"
                    }}}},
                    {{{{
                        "summary": "세 번째 출처의 내용 요약 (최소 40글자)",
                        "source": "세 번째 문서에 포함된 정확한 출처 URL (위 두 개와 달라야 함)"
                    }}}}
                ]
            }}}}

            문서:
            {context}

            대화 기록:
            {chat_history}

            현재 질문: {question}

            JSON 응답:
        """)
    
    def extract_metadata_from_text(self, text: str, file_path: str) -> Dict[str, Any]:
        """
        텍스트에서 메타데이터 추출
        
        Args:
            text: 문서 텍스트
            file_path: 파일 경로
            
        Returns:
            메타데이터 딕셔너리
        """
        metadata = {
            "source_file": file_path,
            "source_url": None,
            "title": None
        }
        
        # 출처 URL 추출 (마지막에 있는 "출처 : URL" 패턴)
        source_pattern = r'출처\s*:\s*(https?://[^\s\n]+)'
        source_match = re.search(source_pattern, text)
        if source_match:
            metadata["source_url"] = source_match.group(1)
        
        # 제목 추출 (첫 번째 줄 또는 파일명)
        lines = text.split('\n')
        if lines:
            first_line = lines[0].strip()
            if first_line and len(first_line) < 200:  # 제목은 짧을 것
                metadata["title"] = first_line
        
        # 제목이 없으면 파일명 사용
        if not metadata["title"]:
            metadata["title"] = os.path.basename(file_path).replace('.txt', '')
        
        return metadata
    
    def preprocess_documents(self, documents: List[Document]) -> List[Document]:
        """
        문서 전처리: 출처 정보 추출 및 메타데이터 추가
        
        Args:
            documents: 원본 문서 리스트
            
        Returns:
            전처리된 문서 리스트
        """
        processed_docs = []
        
        for doc in documents:
            # 텍스트에서 메타데이터 추출
            extracted_metadata = self.extract_metadata_from_text(
                doc.page_content, 
                doc.metadata.get('source', '')
            )
            
            # 기존 메타데이터와 병합
            doc.metadata.update(extracted_metadata)
            
            # 출처 정보를 텍스트 끝에서 제거 (중복 방지)
            content = doc.page_content
            content = re.sub(r'\n*출처\s*:\s*https?://[^\s\n]+\s*$', '', content)
            doc.page_content = content.strip()
            
            processed_docs.append(doc)
            
            logger.info(f"Processed: {doc.metadata.get('title')} - {doc.metadata.get('source_url')}")
        
        return processed_docs
        
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
                
                # 문서 전처리: 메타데이터 추출
                documents = self.preprocess_documents(documents)
                
                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                splits = text_splitter.split_documents(documents)
                
                # 청크에도 메타데이터가 상속됨
                logger.info(f"Created {len(splits)} chunks from {len(documents)} documents")
                
                embeddings = OpenAIEmbeddings()

                self.vector_store = Chroma.from_documents(
                    documents=splits,
                    embedding=embeddings,
                    persist_directory="./chroma_db"
                )
            
            # Create retriever with MMR
            self.retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 5,
                    "fetch_k": 15,
                    "lambda_mult": 0.5
                }
            )
            
            # Create LLM
            qa_model = os.getenv("QA_MODEL", "gpt-3.5-turbo")
            self.llm = ChatOpenAI(model_name=qa_model, temperature=0)
            
            logger.info("RAG system initialized successfully!")
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            raise
    
    def query(self, question: List[str]) -> Dict[str, Any]:
        """
        Query the RAG system with a question.
        
        Args:
            question: List of questions (conversation history)
            
        Returns:
            Dictionary with answer
            
        Raises:
            ValueError: If RAG system is not initialized
        """
        if not self.retriever or not self.llm:
            raise ValueError("RAG system not initialized")
        
        # 가장 최근 질문만 retriever에 사용
        latest_question = question[-1] if question else ""
        
        # 대화 기록 포맷팅 (최근 질문 제외)
        chat_history = ""
        if len(question) > 1:
            for i, q in enumerate(question[:-1], 1):
                role = "사용자" if i % 2 == 1 else "AI"
                chat_history += f"{role}: {q}\n"
        
        # Helper function to format documents with metadata
        def format_docs(docs):
            formatted = []
            for i, doc in enumerate(docs, 1):
                source_url = doc.metadata.get('source_url', '출처없음')
                title = doc.metadata.get('title', '제목없음')
                
                formatted.append(
                    f"=== 문서 {i}: {title} ===\n"
                    f"[출처: {source_url}]\n"
                    f"{doc.page_content}\n"
                    f"=== 문서 {i} 끝 ==="
                )
            
            return "\n\n".join(formatted)
        
        # 문서 검색
        source_documents = self.retriever.invoke(latest_question)
        
        # Create RAG chain using LCEL
        rag_chain = (
            {
                "context": lambda _: format_docs(source_documents),
                "chat_history": lambda _: chat_history,
                "question": lambda _: latest_question
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Get answer from RAG chain
        answer = rag_chain.invoke({})
        
        return {
            "answer": answer,
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


# Global service instance
langchain_service = LangChainService()