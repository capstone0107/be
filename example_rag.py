"""
Example script demonstrating how to use the LangChain RAG system directly.
This can be run independently of the FastAPI server.
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Load environment variables
load_dotenv()

def main():
    """Main function to demonstrate LangChain RAG."""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set in environment variables.")
        print("Please create a .env file with your OpenAI API key.")
        return
    
    print("Loading documents from data/ directory...")
    
    # Load documents
    loader = DirectoryLoader(
        "data/",
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()
    
    print(f"Loaded {len(documents)} documents")
    
    # Split documents into chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    print(f"Created {len(splits)} text chunks")
    
    # Create embeddings and vector store
    print("Creating embeddings and vector store...")
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    # Create QA chain
    print("Initializing QA chain...")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    print("\n" + "="*50)
    print("RAG System Ready!")
    print("="*50 + "\n")
    
    # Example queries
    queries = [
        "What is FastAPI?",
        "What are the key features of LangChain?",
        "How does RAG work?",
        "What models does OpenAI offer?"
    ]
    
    for query in queries:
        print(f"Question: {query}")
        result = qa_chain({"query": query})
        print(f"Answer: {result['result']}")
        print(f"Sources: {len(result['source_documents'])} documents retrieved")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    main()
