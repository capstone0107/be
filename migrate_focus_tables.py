"""
Database migration script to create focus-related tables.
Run this script to create Conversation, Message, and Focus tables.

Usage:
    python migrate_focus_tables.py
"""
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import engine, Base
from models.conversation_orm import Conversation, Message, Focus, conversation_focus

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_existing_tables():
    """Check which tables already exist."""
    from sqlalchemy import inspect
    
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    
    logger.info("Existing tables in database:")
    for table in existing_tables:
        logger.info(f"  - {table}")
    
    return existing_tables


def migrate():
    """Create all focus-related tables."""
    try:
        logger.info("=" * 60)
        logger.info("Starting Focus Tables Migration")
        logger.info("=" * 60)
        
        # Check existing tables
        existing_tables = check_existing_tables()
        
        # Tables to create
        new_tables = ["conversations", "messages", "focuses", "conversation_focus"]
        
        # Check which tables need to be created
        tables_to_create = [t for t in new_tables if t not in existing_tables]
        
        if not tables_to_create:
            logger.info("All focus tables already exist. No migration needed.")
            return
        
        logger.info(f"Tables to create: {', '.join(tables_to_create)}")
        
        # Create tables
        logger.info("Creating tables...")
        Base.metadata.create_all(bind=engine)
        
        logger.info("✅ Successfully created tables:")
        for table in new_tables:
            logger.info(f"  ✓ {table}")
        
        logger.info("=" * 60)
        logger.info("Migration completed successfully!")
        logger.info("=" * 60)
        
        # Verify tables were created
        logger.info("\nVerifying tables...")
        final_tables = check_existing_tables()
        
        for table in new_tables:
            if table in final_tables:
                logger.info(f"  ✓ {table} - verified")
            else:
                logger.warning(f"  ✗ {table} - NOT FOUND!")
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error("Migration FAILED!")
        logger.error("=" * 60)
        logger.error(f"Error: {e}")
        raise


def rollback():
    """Drop all focus-related tables (use with caution!)."""
    try:
        logger.warning("=" * 60)
        logger.warning("WARNING: This will DROP all focus tables!")
        logger.warning("=" * 60)
        
        response = input("Are you sure? Type 'yes' to confirm: ")
        if response.lower() != 'yes':
            logger.info("Rollback cancelled.")
            return
        
        from sqlalchemy import MetaData
        
        metadata = MetaData()
        metadata.reflect(bind=engine)
        
        tables_to_drop = ["conversation_focus", "messages", "focuses", "conversations"]
        
        logger.info("Dropping tables...")
        for table_name in tables_to_drop:
            if table_name in metadata.tables:
                table = metadata.tables[table_name]
                logger.info(f"  Dropping {table_name}...")
                table.drop(engine)
                logger.info(f"  ✓ Dropped {table_name}")
        
        logger.info("Rollback completed successfully!")
        
    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Focus tables migration script")
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback (drop) all focus tables"
    )
    
    args = parser.parse_args()
    
    if args.rollback:
        rollback()
    else:
        migrate()