import os
import logging
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Get the absolute path to the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Build absolute paths relative to the script's directory
DB_DIRECTORY = os.path.join(SCRIPT_DIR, "chroma_db")
ITEMS_TO_INSPECT = 10

def inspect_database():
    """Loads the Chroma database and prints a sample of its contents."""
    load_dotenv()  # Load the .env file
    
    if not os.getenv("OPENAI_API_KEY"): # Quick check
        logging.error("OPENAI_API_KEY not found. Make sure .env file is correct.")
        return

    if not os.path.exists(DB_DIRECTORY):
        logging.error(f"Database directory not found: {DB_DIRECTORY}")
        logging.error("Please run the 'build_db.py' script first.")
        return

    try:
        logging.info(f"Loading database from {DB_DIRECTORY}...")
        # We need the embedding function to tell Chroma how to create
        # query vectors (even though we're just browsing).
        embeddings = OpenAIEmbeddings()
        
        vector_db = Chroma(
            persist_directory=DB_DIRECTORY,
            embedding_function=embeddings
        )

        logging.info("Database loaded successfully.")
        
        total_items = vector_db._collection.count()
        if total_items == 0:
            logging.warning("The database is empty.")
            return
            
        logging.info(f"Total items in database: {total_items}")

        # --- This is how you "browse" the DB ---
        logging.info(f"--- Inspecting first {ITEMS_TO_INSPECT} items ---")
        
        results = vector_db.get(
            limit=ITEMS_TO_INSPECT,
            include=["metadatas", "documents"] 
        )
        
        metadatas = results.get('metadatas', [])
        documents = results.get('documents', [])

        for i in range(len(documents)):
            meta = metadatas[i]
            doc = documents[i].replace('\n', ' ')[0:150] # Get a snippet
            
            logging.info(f"\n--- ITEM {i+1} ---")
            logging.info(f"  SOURCE (URL): {meta.get('source')}") # <-- THIS IS THE FIX
            logging.info(f"  TITLE: {meta.get('title')}")
            logging.info(f"  HEADER: {meta.get('Header 2', 'N/A')}")
            logging.info(f"  CONTENT: {doc}...")

    except Exception as e:
        logging.error(f"An error occurred while inspecting the database: {e}")

if __name__ == "__main__":
    inspect_database()