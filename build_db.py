import os
import shutil
import logging
import pandas as pd  # <-- 1. Import pandas
from dotenv import load_dotenv
# 2. Import DataFrameLoader
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Get the absolute path to the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Build absolute paths relative to the script's directory
CSV_FILE_PATH = os.path.join(SCRIPT_DIR, "data", "os_docs.csv")
DB_DIRECTORY = os.path.join(SCRIPT_DIR, "chroma_db")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def check_api_key():
    """Check if the OpenAI API key is set in the environment."""
    if not os.getenv("OPENAI_API_KEY"):
        logging.error("OPENAI_API_KEY environment variable not set.")
        logging.error("Make sure you have a .env file with OPENAI_API_KEY='sk-...'")
        raise EnvironmentError("OPENAI_API_KEY environment variable not set.")
    logging.info("OpenAI API key loaded successfully.")

def load_documents():
    """Load documents using pandas and DataFrameLoader."""
    if not os.path.exists(CSV_FILE_PATH):
        logging.error(f"CSV file not found at path: {CSV_FILE_PATH}")
        raise FileNotFoundError(f"CSV file not found at path: {CSV_FILE_PATH}")
    
    # 3. Use pandas to read the CSV
    logging.info(f"Loading CSV from {CSV_FILE_PATH} using pandas...")
    try:
        df = pd.read_csv(CSV_FILE_PATH)
    except Exception as e:
        logging.error(f"Error reading CSV file with pandas: {e}")
        raise
    
    # 4. Use DataFrameLoader
    # This is the stable way to specify the content column
    loader = DataFrameLoader(
        df,
        page_content_column="content" # <-- This is the correct argument for DataFrameLoader
    )
    
    logging.info("Loading documents from DataFrame...")
    documents = loader.load()
    
    # 5. Manually add metadata (url, title, etc.) because DataFrameLoader doesn't have
    #    'source_column' or 'metadata_columns' arguments.
    #    We loop through the docs and the dataframe rows together.
    
    # Get all rows as a list of dictionaries
    all_rows_metadata = df.to_dict(orient="records")
    
    final_documents = []
    for i, doc in enumerate(documents):
        # The 'doc' only has page_content. We need to add metadata.
        # Get the corresponding row's data
        metadata_row = all_rows_metadata[i]
        
        # We want to use 'url' as the 'source'
        if "url" in metadata_row:
            doc.metadata["source"] = metadata_row.get("url")
            
        # Add other useful metadata
        if "title" in metadata_row:
            doc.metadata["title"] = metadata_row.get("title")
        if "category" in metadata_row:
            doc.metadata["category"] = metadata_row.get("category")
        if "page_id" in metadata_row:
            doc.metadata["page_id"] = metadata_row.get("page_id")
        
        final_documents.append(doc)

    logging.info(f"Loaded {len(final_documents)} documents and enriched with metadata.")
    return final_documents

def split_documents(documents):
    """
    Split documents using a semantic, two-stage approach:
    1. First, split by Markdown headers (`== ... ==`).
    2. Then, use RecursiveCharacterTextSplitter as a fallback
       for any sections that are still too large.
    """
    logging.info("Splitting documents using MarkdownHeaderTextSplitter...")
    
    headers_to_split_on = [
        ("==", "Header 2"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, 
        strip_headers=True
    )
    
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )

    all_chunks = []
    for doc in documents:
        content = doc.page_content
        markdown_chunks = markdown_splitter.split_text(content)
        
        chunks_with_metadata = []
        for chunk in markdown_chunks:
            # We merge this with the original document's metadata (source, title, etc.)
            new_metadata = doc.metadata.copy() 
            new_metadata.update(chunk.metadata)
            
            new_chunk_doc = (
                chunk.page_content,
                new_metadata
            )
            chunks_with_metadata.append(new_chunk_doc)
        
        # Fallback: if any markdown_chunk is *still* too big, split it further
        docs_to_split = [Document(page_content=content, metadata=meta) for content, meta in chunks_with_metadata]
        final_chunks = fallback_splitter.split_documents(docs_to_split)
        
        all_chunks.extend(final_chunks)

    logging.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents.")
    return all_chunks

# ... (all your imports and other functions are the same) ...
# ... (check_api_key, load_documents, split_documents) ...

def build_vector_db(chunks):
    """
    Create and persist the Chroma vector database by
    processing the chunks in small batches.
    """
    logging.info("Initializing OpenAI embeddings...")
    embeddings = OpenAIEmbeddings() 

    logging.info("Building Chroma vector database in batches...")
    
    # Define a batch size (e.g., 500 chunks at a time)
    # You can make this number smaller (like 200) if it fails again
    batch_size = 500 
    
    # Get the first batch to create the database
    first_batch = chunks[:batch_size]
    
    if not first_batch:
        logging.error("No chunks to process. Exiting.")
        return

    logging.info(f"Creating DB with first batch ({len(first_batch)} chunks)...")
    # 1. Create the database with the first batch
    vector_db = Chroma.from_documents(
        documents=first_batch,
        embedding=embeddings,
        persist_directory=DB_DIRECTORY
    )

    # 2. Now, add the rest of the chunks in batches
    for i in range(batch_size, len(chunks), batch_size):
        # Get the next batch
        batch = chunks[i:i+batch_size]
        
        logging.info(f"Adding batch {i//batch_size + 1} ({len(batch)} chunks) to Chroma...")
        
        # Add the batch to the existing database
        vector_db.add_documents(documents=batch)
    
    # 3. Persist all the additions
    logging.info("Persisting all batches to disk...")
    vector_db.persist()

    logging.info(f"Successfully built and persisted vector DB at {DB_DIRECTORY}")
    return vector_db

# ... (your main function is the same) ...

def main():
    """Main function to build the vector database."""
    load_dotenv()
    
    try:
        check_api_key()
        
        if os.path.exists(DB_DIRECTORY):
            logging.warning(f"Database directory '{DB_DIRECTORY}' already exists.")
            logging.warning("To rebuild, manually delete the directory.")
            return

        documents = load_documents()
        if not documents:
            logging.warning("No documents loaded, stopping build.")
            return
            
        chunks = split_documents(documents)

        # --- Inspection Code (Optional) ---
        logging.info(f"--- 🔍 Inspecting First 5 Chunks ---")
        for i, chunk in enumerate(chunks[:5]):
            logging.info(f"--- CHUNK {i+1} / {len(chunks)} ---")
            logging.info(f"  METADATA: {chunk.metadata}")
            content_snippet = chunk.page_content.replace('\n', ' ')[0:150]
            logging.info(f"  CONTENT (Snippet): {content_snippet}...")
            logging.info(f"  CONTENT (Length): {len(chunk.page_content)} characters")
        logging.info("--- End of Inspection ---")
        # --- End of Inspection ---
        
        build_vector_db(chunks)
        
        logging.info("Database build process completed successfully.")
        
    except Exception as e:
        logging.error(f"An error occurred during the build process: {e}")

if __name__ == "__main__":
    main()

