"""
LangChain service for RAG system.
"""
import os
import logging
import json
from typing import Optional, Dict, Any, List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class LangChainService:
    """Service class for managing LangChain RAG operations."""
    
    def __init__(self):
        self.vector_store: Optional[Chroma] = None
        self.retriever = None
        self.llm = None
        self.rag_chain = None  # Add rag_chain placeholder
        
        # 커스텀 프롬프트 설정 (Original prompt is good)
        self.prompt = ChatPromptTemplate.from_template("""
            당신은 친절한 AI 어시B턴트입니다.

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
        
    @staticmethod
    def _format_docs_with_metadata(docs: List[Document]) -> str:
        """
        Formats documents for the prompt, including their source metadata.
        This is the fix for Part 1.
        """
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'N/A')  # extract source from retrieved chunks
            formatted.append(f"--- Document {i+1} (Source: {source}) ---\n{doc.page_content}")
            
        return "\n\n".join(formatted)

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
                # When loader.load() runs, it creates Documents
                # with metadata={"source": "path/to/file.txt"}
                documents = loader.load()
                
                if not documents:
                    logger.warning(f"No documents found in {data_dir} directory.")
                    return
                
                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                # When splits runs, it copies the metadata to all chunks
                splits = text_splitter.split_documents(documents)
                embeddings = OpenAIEmbeddings()

                # The vector store saves the chunks AND their metadata
                self.vector_store = Chroma.from_documents(
                    documents=splits,
                    embedding=embeddings, # 임베딩 객체 전달
                    persist_directory="./chroma_db"
                )
            
            # --- DIVERSITY FIX (Part 2) ---
            # Increase 'k' to give the LLM more options to choose from.
            self.retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 6,          # Retrieve 6 docs to give the LLM options
                    "fetch_k": 20    # Fetch 20 docs for MMR to select from
                }
            )
            '''
            [
                <Doc 1: chunk from data/doc_A.txt, content: "E=mc^2 is...">,
                <Doc 2: chunk from data/doc_B.txt, content: "H2O is water...">,
                <Doc 3: chunk from data/doc_C.txt, content: "DNA has A, T, C, G...">,
                <Doc 4: chunk from data/doc_A.txt, content: "F=ma is Newton's...">,
                <Doc 5: chunk from data/doc_B.txt, content: "C6H12O6 is glucose...">,
                <Doc 6: chunk from data/doc_A.txt, content: "PV=nRT is the ideal...">
            ]
            '''

            # Create LLM
            qa_model = os.getenv("QA_MODEL", "gpt-3.5-turbo")
            self.llm = ChatOpenAI(model_name=qa_model, temperature=0)
            
            # --- LCEL CHAIN FIX (Part 1) ---
            
            # This Runnable retrieves the docs (with metadata)
            retriever_with_docs = RunnablePassthrough.assign(
                docs=self.retriever
            )
            
            # This Runnable uses our new function to create the context string
            # This is where the URL is revealed to the LLM
            context_creator = RunnablePassthrough.assign(
                context=(lambda x: self._format_docs_with_metadata(x["docs"]))
            )
            '''
            --- Document 1 (Source: data/doc_A.txt) ---
            E=mc^2 is the mass-energy equivalence.

            --- Document 2 (Source: data/doc_B.txt) ---
            H2O is the chemical formula for water.

            --- Document 3 (Source: data/doc_C.txt) ---
            DNA has four bases: Adenine (A), Thymine (T), Cytosine (C), and Guanine (G).

            --- Document 4 (Source: data/doc_A.txt) ---
            F=ma is Newton's second law of motion.

            --- Document 5 (Source: data/doc_B.txt) ---
            C6H12O6 is the chemical formula for glucose.

            --- Document 6 (Source: data/doc_A.txt) ---
            PV=nRT is the ideal gas law.
            '''

            # This defines the payload that will be sent to the final prompt
            rag_chain_payload = {
                "context": (lambda x: x["context"]),
                "question": (lambda x: x["question"]),
            }
            
            # This is the full chain
            self.rag_chain = (
                retriever_with_docs  # Input: question (str)
                | context_creator    # Output: {"question":..., "docs":..., "context":...}
                | {
                    # "answer" branch: runs the LLM to get the JSON string
                    "answer": rag_chain_payload | self.prompt | self.llm | StrOutputParser(),
                    # "source_documents" branch: just passes the raw docs list through
                    "source_documents": (lambda x: x["docs"]) 
                  }
            )
            '''
            below is an example look of method "query"'s return value inside the try block
            {
                "answer_json": {
                    "answer": "과학에는 다양한 유형의 공식이 있습니다...",
                    "cards": [
                        { "summary": "...", "source": "data/doc_A.txt" },
                        { "summary": "...", "source": "data/doc_B.txt" },
                        { "summary": "...", "source": "data/doc_C.txt" }
                    ]
                },
                "retrieved_sources": [
                    { "content_snippet": "E=mc^2...", "source": "data/doc_A.txt" },
                    { "content_snippet": "H2O is water...", "source": "data/doc_B.txt" },
                    { "content_snippet": "DNA has A, T, C, G...", "source": "data/doc_C.txt" },
                    { "content_snippet": "F=ma is Newton's...", "source": "data/doc_A.txt" },
                    { "content_snippet": "C6H12O6 is glucose...", "source": "data/doc_B.txt" },
                    { "content_snippet": "PV=nRT is the ideal...", "source": "data/doc_A.txt" }
                ]
            }
            '''
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
        if not self.rag_chain: # Check if the chain is initialized
            raise ValueError("RAG system not initialized. Call initialize_rag_system() first.")
        
        # --- REVISED QUERY METHOD ---
        # Invoke the pre-defined LCEL chain
        result = self.rag_chain.invoke(question)
        
        # 'result' is now a dictionary: {"answer": "...", "source_documents": [Doc, ...]}
        
        # Format the source documents for a clean return value
        # This shows ALL 6 documents that were retrieved
        source_info = [
            {
                "content_snippet": doc.page_content[:200] + "...", 
                "source": doc.metadata.get("source", "N/A") # Get source from metadata
            }
            for doc in result["source_documents"] # These are the 6 docs from the retriever
        ]
        
        # Attempt to parse the JSON answer for cleaner output
        try:
            # result["answer"] is the JSON string from the LLM
            # The LLM *itself* has already selected the diverse sources
            # and put them in its 'cards'
            parsed_answer = json.loads(result["answer"])
            return {
                "answer_json": parsed_answer,
                "retrieved_sources": source_info # All 6 sources passed to the LLM
            }
        except json.JSONDecodeError:
            # Fallback if LLM output is not valid JSON
            logger.warning("LLM output was not valid JSON.")
            return {
                "answer_json": {"answer": "Error: LLM did not return valid JSON.", "cards": []},
                "raw_answer": result["answer"],
                "retrieved_sources": source_info
            }
    
    def is_initialized(self) -> bool:
        """Check if the RAG system is initialized."""
        return self.rag_chain is not None and self.llm is not None
    
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
