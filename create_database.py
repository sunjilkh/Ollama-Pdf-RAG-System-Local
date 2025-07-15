from langchain_chroma import Chroma
from embedding import get_embedding_function_with_fallback
from assign_ids import assign_unique_ids
from split import chunks
import os
import chromadb


def create_or_update_database(chunks_with_ids, persist_directory="db"):
    """
    Create or update ChromaDB vector database with document chunks.

    Args:
        chunks_with_ids: List of document chunks with unique IDs
        persist_directory: Directory to persist the database

    Returns:
        ChromaDB vector store instance
    """
    print("Creating/updating vector database...")

    # Get embedding function
    try:
        embedding_function = get_embedding_function_with_fallback()
        print("Embedding function initialized successfully")
    except Exception as e:
        print(f"Error initializing embedding function: {e}")
        return None

    # Create or load existing database with proper ChromaDB client configuration
    print(f"Setting up ChromaDB in directory: {persist_directory}")
    
    try:
        # Try with PersistentClient for newer ChromaDB versions
        client = chromadb.PersistentClient(path=persist_directory)
        
        db = Chroma(
            client=client,
            embedding_function=embedding_function,
        )
        
    except Exception as client_error:
        print(f"PersistentClient failed: {client_error}")
        # Fallback to legacy configuration
        db = Chroma(
            persist_directory=persist_directory, 
            embedding_function=embedding_function
        )

    # Get existing IDs to avoid duplicates
    try:
        existing_data = db.get()
        existing_ids = set(existing_data["ids"]) if existing_data["ids"] else set()
        print(f"Found {len(existing_ids)} existing documents in database")
    except Exception as e:
        print(f"Database appears to be empty or new: {e}")
        existing_ids = set()

    # Filter out chunks that already exist
    new_chunks = [
        chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids
    ]

    print(f"Total chunks: {len(chunks_with_ids)}")
    print(f"Existing chunks: {len(existing_ids)}")
    print(f"New chunks to add: {len(new_chunks)}")

    if new_chunks:
        print("Adding new chunks to database...")
        try:
            # Extract IDs for the new chunks
            new_ids = [chunk.metadata["id"] for chunk in new_chunks]

            # Add documents to database
            db.add_documents(new_chunks, ids=new_ids)

            # Note: ChromaDB automatically persists data in newer versions
            print(f"✅ Successfully added {len(new_chunks)} new chunks to database")
        except Exception as e:
            print(f"❌ Error adding chunks to database: {e}")
            return None
    else:
        print("No new chunks to add - database is up to date")

    return db


def test_database_query(db, test_query="What is an algorithm?"):
    """Test the database with a sample query."""
    if db is None:
        print("Database is not available for testing")
        return False

    try:
        print(f"\nTesting database with query: '{test_query}'")

        # Perform similarity search
        results = db.similarity_search(test_query, k=3)

        print(f"Found {len(results)} relevant documents:")
        for i, result in enumerate(results, 1):
            print(f"\n  Result {i}:")
            print(f"    ID: {result.metadata.get('id', 'N/A')}")
            print(f"    Page: {result.metadata.get('page', 'N/A')}")
            print(f"    Content preview: {result.page_content[:150]}...")

        return True
    except Exception as e:
        print(f"Error testing database: {e}")
        return False


def get_database_stats(db):
    """Get statistics about the database."""
    if db is None:
        print("Database is not available")
        return

    try:
        data = db.get()
        total_docs = len(data["ids"]) if data["ids"] else 0

        print(f"\n📊 Database Statistics:")
        print(f"  Total documents: {total_docs}")
        print(f"  Database directory: {db._persist_directory}")

        if total_docs > 0:
            # Show some example IDs
            example_ids = data["ids"][:5] if len(data["ids"]) >= 5 else data["ids"]
            print(f"  Example document IDs: {example_ids}")

    except Exception as e:
        print(f"Error getting database stats: {e}")


if __name__ == "__main__":
    print("Starting vector database creation process...")

    # Get chunks with IDs
    print("Processing chunks...")
    chunks_with_ids = assign_unique_ids(chunks)

    # Create/update database
    db = create_or_update_database(chunks_with_ids)

    if db:
        # Get database statistics
        get_database_stats(db)

        # Test the database
        test_success = test_database_query(db)

        if test_success:
            print("\n✅ Vector database created and tested successfully!")
            print("Database is ready for querying and retrieval.")
        else:
            print("\n⚠️ Database created but testing failed.")
    else:
        print("\n❌ Failed to create vector database.")
