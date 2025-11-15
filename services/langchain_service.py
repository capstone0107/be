"""
LangChain service for RAG system with Google Search integration.
"""
import os
import re
import json
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

from services.google_search_service import google_search_service

logger = logging.getLogger(__name__)


class LangChainService:
    """Service class for managing LangChain RAG operations with Google Search."""
    
    def __init__(self):
        self.vector_store: Optional[Chroma] = None
        self.retriever = None
        self.llm = None
        self.search_query_llm = None  # 검색 쿼리 생성용 별도 LLM
        
        # 커스텀 프롬프트 설정 (Google Search 결과 포함)
        self.prompt = ChatPromptTemplate.from_template("""
            당신은 정확한 답변을 하는 AI 어시스턴트입니다.

            아래 내부 문서와 웹 검색 결과를 참고하여 질문에 답변하세요.
            
            ### 규칙:
            1. 내부 문서에 관련 정보가 충분하면 내부 문서를 우선 사용하세요.
            2. 내부 문서가 불충분하거나 최신 정보가 필요하면 웹 검색 결과를 활용하세요.
            3. 출처를 명확히 구분하세요 (내부 문서 vs 웹 검색).
            4. 반드시 순수한 JSON만 출력하세요. 마크다운 코드 블록(``` 또는 ```json)을 사용하지 마세요.

            ### 내부 문서:
            {context}
            
            ### 웹 검색 결과:
            {search_results}

            ### 대화 기록:
            {chat_history}

            ### 현재 질문:
            {question}
            
            다음 JSON 형식으로만 답변하세요:
            {{{{
                "answer": "여기에 전체 답변, 충분히 디테일하게.",
                "cards": [
                    {{{{
                        "summary": "첫 번째 출처의 내용 요약",
                        "source": "첫 번째 문서의 정확한 출처 URL 또는 파일명"
                    }}}},
                    {{{{
                        "summary": "두 번째 출처의 내용 요약",
                        "source": "두 번째 문서의 정확한 출처 URL 또는 파일명"
                    }}}},
                    {{{{
                        "summary": "세 번째 출처의 내용 요약",
                        "source": "세 번째 문서의 정확한 출처 URL 또는 파일명"
                    }}}}
                ]
            }}}}

            중요:
            - 지식 카드는 실제 사용한 출처 기반으로만 작성
            - 각 카드의 출처(source)는 서로 달라야 함
            - 내부 문서는 파일명, 웹 검색은 URL로 표시
        """)
        
        # 검색 쿼리 최적화 프롬프트
        self.query_optimization_prompt = ChatPromptTemplate.from_template("""
            당신은 검색 쿼리 최적화 전문가입니다.
            사용자의 질문을 구글 검색에 최적화된 검색어로 변환하세요.

            ### 규칙:
            1. 핵심 키워드만 추출 (3-5개 단어)
            2. 불필요한 조사, 어미 제거
            3. 영어 기술 용어는 영어로 유지
            4. 검색 의도를 명확히 반영
            5. 너무 일반적이거나 모호한 단어 제거

            ### 예시:
            - 질문: "실시간 운영체제에서 우선순위 스케줄링이 왜 중요해?"
            - 검색어: "실시간 운영체제 우선순위 스케줄링 중요성"
            
            - 질문: "What are the benefits of using Docker?"
            - 검색어: "Docker benefits advantages"

            ### 사용자 질문:
            {question}

            ### 최적화된 검색어만 출력하세요 (다른 설명 없이):
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
            
            # Create retriever with similarity score threshold
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 5,
                    "score_threshold": 0.9  # 임계값
                }
            )
            
            # Create main LLM for answer generation
            qa_model = os.getenv("QA_MODEL", "gpt-3.5-turbo")
            self.llm = ChatOpenAI(model_name=qa_model, temperature=0)
            
            # Create separate LLM for search query optimization (lighter model)
            self.search_query_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            
            logger.info("RAG system initialized successfully with Google Search support!")
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            raise
    
    def generate_search_query(self, question: str, chat_history: str = "") -> Optional[str]:
        """
        Generate optimized search query from user question.
        
        Args:
            question: User's question
            chat_history: Optional conversation context
            
        Returns:
            Optimized search query or None if generation fails
        """
        if not self.search_query_llm:
            logger.warning("Search query LLM not initialized")
            return None
        
        try:
            # Create query optimization chain
            query_chain = (
                self.query_optimization_prompt
                | self.search_query_llm
                | StrOutputParser()
            )
            
            # Generate optimized query
            optimized_query = query_chain.invoke({"question": question})
            optimized_query = optimized_query.strip()
            
            logger.info(f"Original question: {question}")
            logger.info(f"Optimized search query: {optimized_query}")
            
            return optimized_query
            
        except Exception as e:
            logger.error(f"Error generating search query: {e}")
            return None
    
    def perform_web_search(self, query: str, num_results: int = 3) -> str:
        """
        Perform web search using Google Search service.
        
        Args:
            query: Search query
            num_results: Number of results to fetch
            
        Returns:
            Formatted search results as string
        """
        # Check if Google Search is available
        if not google_search_service.is_available():
            logger.warning("Google Search service not available")
            return "웹 검색 서비스를 사용할 수 없습니다."
        
        try:
            # Perform search
            results = google_search_service.search(query, num_results)
            
            if not results:
                logger.info(f"No search results found for query: {query}")
                return "검색 결과가 없습니다."
            
            # Format results
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append(
                    f"=== 웹 검색 결과 {i} ===\n"
                    f"제목: {result.title}\n"
                    f"출처: {result.link}\n"
                    f"요약: {result.snippet}\n"
                )
            
            logger.info(f"Found {len(results)} web search results")
            return "\n\n".join(formatted_results)
            
        except Exception as e:
            logger.error(f"Error performing web search: {e}")
            return "웹 검색 중 오류가 발생했습니다."
    
    def query(self, question: List[str]) -> Dict[str, Any]:
        """
        Query the RAG system with Google Search support.
        
        Args:
            question: List of questions (conversation history)
            
        Returns:
            Dictionary with answer and cards
            
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
        
        # Step 1: 내부 문서 검색
        logger.info("Step 1: Searching internal documents...")
        source_documents = self.retriever.invoke(latest_question)
        logger.info(f"Found {len(source_documents)} internal documents")
        
        # Step 2: 검색 쿼리 생성
        logger.info("Step 2: Generating optimized search query...")
        search_query = self.generate_search_query(latest_question, chat_history)
        
        # Step 3: 웹 검색 수행 (검색 쿼리가 생성된 경우에만)
        web_search_results = "웹 검색이 비활성화되었습니다."
        if search_query:
            logger.info("Step 3: Performing web search...")
            web_search_results = self.perform_web_search(search_query, num_results=3)
        else:
            logger.info("Step 3: Skipping web search (query generation failed)")
        
        # Helper function to format documents with metadata
        def format_docs(docs):
            if not docs:
                return "관련 내부 문서를 찾을 수 없습니다."
            
            formatted = []
            for i, doc in enumerate(docs, 1):
                source_url = doc.metadata.get('source_url', '출처없음')
                title = doc.metadata.get('title', '제목없음')
                
                formatted.append(
                    f"=== 내부 문서 {i}: {title} ===\n"
                    f"[출처: {source_url}]\n"
                    f"{doc.page_content}\n"
                    f"=== 문서 {i} 끝 ==="
                )
            
            return "\n\n".join(formatted)
        
        # Step 4: RAG chain 생성 (내부 문서 + 웹 검색 결과)
        logger.info("Step 4: Creating RAG chain with internal + web search results...")
        rag_chain = (
            {
                "context": lambda _: format_docs(source_documents),
                "search_results": lambda _: web_search_results,
                "chat_history": lambda _: chat_history,
                "question": lambda _: latest_question
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Step 5: 답변 생성
        logger.info("Step 5: Generating answer...")
        answer = rag_chain.invoke({})
        
        # JSON 파싱 및 에러 처리
        try:
            # JSON 파싱 시도
            parsed_answer = json.loads(answer)
            
            # 필수 필드 검증
            if "answer" not in parsed_answer:
                logger.error("Response missing 'answer' field")
                parsed_answer["answer"] = "답변을 찾을 수 없습니다."
            
            if "cards" not in parsed_answer:
                parsed_answer["cards"] = []
            
            logger.info(f"Successfully generated answer with {len(parsed_answer.get('cards', []))} cards")
            return parsed_answer
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Raw response (first 1000 chars): {answer[:1000]}")
            
            # JSON이 아닌 일반 텍스트로 응답한 경우 처리
            return {
                "answer": answer,
                "cards": []
            }
        except Exception as e:
            logger.error(f"Unexpected error in query: {e}")
            return {
                "answer": "오류가 발생했습니다.",
                "cards": []
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