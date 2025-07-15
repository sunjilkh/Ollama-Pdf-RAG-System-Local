from langchain_chroma import Chroma
from embedding import get_embedding_function_with_fallback
import os
import time
import threading
from typing import Optional, List, Dict, Any
from functools import lru_cache
from config import DATABASE_DIRECTORY, RETRIEVAL_COUNT
import chromadb


class OptimizedDatabaseManager:
    """Singleton database manager with caching and optimization"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(OptimizedDatabaseManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            self.initialized = True
            self.db_cache = None
            self.embedding_function = None
            self.persist_directory = DATABASE_DIRECTORY
            self.query_cache = {}
            self.cache_hits = 0
            self.cache_misses = 0
            self._preload_database()

    def _preload_database(self):
        """Preload database in background for faster access"""

        def preload():
            try:
                print("🔄 Preloading database...")
                self.get_database()
                print("✅ Database preloaded successfully")
            except Exception as e:
                print(f"⚠️  Database preload failed: {e}")

        # Run preload in background
        threading.Thread(target=preload, daemon=True).start()

    def get_embedding_function(self):
        """Get cached embedding function"""
        if self.embedding_function is None:
            self.embedding_function = get_embedding_function_with_fallback()
        return self.embedding_function

    def get_database(self):
        """Get cached database instance"""
        if self.db_cache is None:
            self.db_cache = self._load_database()
        return self.db_cache

    def _load_database(self):
        """Load database with optimizations and proper ChromaDB client configuration"""
        try:
            if not os.path.exists(self.persist_directory):
                print(f"Database directory '{self.persist_directory}' does not exist.")
                return None

            embedding_function = self.get_embedding_function()

            # Create ChromaDB client with proper configuration for newer versions
            try:
                # Try with PersistentClient for newer ChromaDB versions
                client = chromadb.PersistentClient(path=self.persist_directory)

                db = Chroma(
                    client=client,
                    embedding_function=embedding_function,
                )

            except Exception as client_error:
                print(f"PersistentClient failed: {client_error}")
                # Fallback to legacy configuration
                db = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=embedding_function,
                )

            # Quick validation
            data = db.get()
            if not data["ids"]:
                print("Database exists but is empty.")
                return None

            print(f"✅ Database loaded with {len(data['ids'])} documents")
            return db

        except Exception as e:
            print(f"❌ Error loading database: {e}")
            return None

    def query_database_cached(self, query: str, k: int = RETRIEVAL_COUNT) -> List[Any]:
        """Query database with result caching"""

        # Create cache key
        cache_key = f"{query}:{k}"

        # Check cache first
        if cache_key in self.query_cache:
            self.cache_hits += 1
            return self.query_cache[cache_key]

        # Cache miss - query database
        self.cache_misses += 1
        db = self.get_database()

        if db is None:
            return []

        try:
            # Optimized similarity search
            results = db.similarity_search(query, k=k)

            # Cache the results
            self.query_cache[cache_key] = results

            # Limit cache size
            if len(self.query_cache) > 200:
                # Remove oldest entries
                oldest_key = next(iter(self.query_cache))
                del self.query_cache[oldest_key]

            return results

        except Exception as e:
            print(f"❌ Error querying database: {e}")
            return []

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_queries = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_queries if total_queries > 0 else 0

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.query_cache),
        }

    def clear_cache(self):
        """Clear all caches"""
        self.query_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        print("🧹 Cache cleared")

    def warmup_cache(self):
        """Warm up cache with common queries"""
        common_queries = [
            "algorithm",
            "sorting",
            "data structure",
            "binary search",
            "complexity",
        ]

        for query in common_queries:
            self.query_database_cached(query, k=3)

        print(f"🔥 Cache warmed up with {len(common_queries)} common queries")


# Global optimized database manager instance
db_manager = OptimizedDatabaseManager()


def load_database(persist_directory="db"):
    """
    Load existing ChromaDB vector database using optimized manager.

    Args:
        persist_directory: Directory where database is persisted

    Returns:
        ChromaDB vector store instance or None if failed
    """
    return db_manager.get_database()


def query_database(db, query, k=RETRIEVAL_COUNT):
    """
    Query the vector database and return relevant documents using optimized caching.

    Args:
        db: ChromaDB vector store instance (can be None, will use optimized manager)
        query: Search query string
        k: Number of documents to retrieve

    Returns:
        List of relevant document chunks
    """
    # Use optimized cached query
    results = db_manager.query_database_cached(query, k=k)

    if results:
        print(f"Found {len(results)} relevant documents")
    else:
        print("No relevant documents found")

    return results


def query_database_with_scores(db, query, k=5):
    """
    Query the vector database and return documents with similarity scores.

    Args:
        db: ChromaDB vector store instance
        query: Search query string
        k: Number of documents to retrieve

    Returns:
        List of tuples: (document, similarity_score)
    """
    if db is None:
        print("Database is not available")
        return []

    try:
        # Search for relevant documents with scores
        results = db.similarity_search_with_score(query, k=k)
        print(f"Found {len(results)} relevant documents with scores")
        return results

    except Exception as e:
        print(f"Error querying database with scores: {e}")
        return []


def get_citation_info(metadata):
    """
    Extract citation information from document metadata.

    Args:
        metadata (dict): Document metadata

    Returns:
        str: Formatted citation string
    """
    file_name = metadata.get(
        "file_name", metadata.get("source_file", "Unknown Document")
    )
    page_number = metadata.get("page_number", metadata.get("page", "Unknown"))

    # Clean up file name (remove extension and path)
    if file_name and file_name != "Unknown Document":
        clean_name = os.path.basename(file_name).replace(".pdf", "")
    else:
        clean_name = "Unknown Document"

    return f"{clean_name}, Page {page_number}"


def generate_optimized_prompt_template(query, results, max_context_length=1200):
    """
    Generate optimized prompt template with reduced length for faster inference.
    """
    if not results:
        return f"Answer briefly: {query}\n\nAnswer:"

    # Extract only the most essential information
    contexts = []
    current_length = 0

    for result in results:
        content = result.page_content.strip()

        # Extract key sentences (first 150 chars)
        if len(content) > 150:
            # Try to find a good breaking point
            content = content[:150]
            last_period = content.rfind(".")
            if last_period > 100:
                content = content[: last_period + 1]
            else:
                content = content + "..."

        # Simple page reference
        page = result.metadata.get("page", "?")
        source_info = f"[{page}] {content}"

        if current_length + len(source_info) < max_context_length:
            contexts.append(source_info)
            current_length += len(source_info)
        else:
            break

    context = "\n".join(contexts)

    # Ultra-concise prompt
    prompt = f"""Context: {context}

Q: {query}
A:"""

    return prompt


def generate_prompt_template(query, results, max_context_length=4000):
    """
    Generate a prompt template with retrieved context and user question.

    Args:
        query: User query string
        results: List of relevant document chunks
        max_context_length: Maximum length of context to include

    Returns:
        Formatted prompt string
    """
    if not results:
        return f"""
I don't have any relevant information to answer your question.

Question: {query}

Please ask a question about the content in the uploaded documents.
"""

    # Combine context from all results
    contexts = []
    current_length = 0
    citation_info = []

    for i, result in enumerate(results):
        content = result.page_content.strip()

        # Get citation information
        citation = get_citation_info(result.metadata)
        citation_info.append(citation)

        source_info = f"[Source: {citation}]"

        # Check if adding this context would exceed the limit
        new_content = f"{source_info}\n{content}\n\n"
        if current_length + len(new_content) > max_context_length:
            break

        contexts.append(new_content)
        current_length += len(new_content)

    context = "".join(contexts).strip()

    # Create unique citations list
    unique_citations = list(
        dict.fromkeys(citation_info)
    )  # Preserve order, remove duplicates
    citations_text = "\n".join([f"- {citation}" for citation in unique_citations])

    prompt_template = f"""Use the following context to answer the question. If the answer cannot be found in the context, say "I don't have enough information to answer this question based on the provided context."

Always cite your sources using the page numbers provided in the context.

Context:
{context}

Question: {query}

Answer: [Provide your answer here and cite the relevant pages]

Sources:
{citations_text}"""

    return prompt_template


def format_search_results(results, query):
    """
    Format search results for display with proper citations.

    Args:
        results: List of search results
        query: Original search query

    Returns:
        str: Formatted results string
    """
    if not results:
        return "No relevant results found."

    formatted_results = []
    formatted_results.append(f"🔍 Search Results for: '{query}'")
    formatted_results.append("=" * 50)

    for i, result in enumerate(results, 1):
        citation = get_citation_info(result.metadata)
        content_preview = (
            result.page_content[:200] + "..."
            if len(result.page_content) > 200
            else result.page_content
        )

        formatted_results.append(f"\n📄 Result {i}: {citation}")
        formatted_results.append(f"Content: {content_preview}")
        formatted_results.append("-" * 30)

    return "\n".join(formatted_results)


def interactive_query_session(db):
    """Run an interactive query session."""
    if db is None:
        print("Database is not available")
        return

    print("Interactive query session started")
    print("Type 'quit' to exit")

    while True:
        try:
            query = input("\n🔍 Enter your query: ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not query:
                continue

            # Query the database
            results = query_database(db, query, k=5)

            if results:
                print(f"\n✅ Found {len(results)} relevant documents")

                # Display formatted results
                formatted_results = format_search_results(results, query)
                print(formatted_results)

                # Generate and display prompt
                prompt = generate_prompt_template(query, results)
                print(f"\n📝 Generated prompt for LLM:")
                print(prompt)
            else:
                print("❌ No relevant documents found")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def test_sample_queries(db):
    """Test the database with sample queries."""
    if db is None:
        print("Database is not available for testing")
        return

    test_queries = [
        "What is an algorithm?",
        "How does quicksort work?",
        "What is the time complexity of binary search?",
        "Explain dynamic programming",
        "What is a graph data structure?",
    ]

    print("🧪 Testing database with sample queries...")
    print("=" * 50)

    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        results = query_database(db, query, k=3)

        if results:
            print(f"✅ Found {len(results)} results")
            for i, result in enumerate(results, 1):
                citation = get_citation_info(result.metadata)
                preview = (
                    result.page_content[:100] + "..."
                    if len(result.page_content) > 100
                    else result.page_content
                )
                print(f"   {i}. {citation}")
                print(f"      {preview}")
        else:
            print("❌ No results found")

        print("-" * 30)


if __name__ == "__main__":
    # Load database
    db = load_database()

    if db:
        print("Choose an option:")
        print("1. Interactive query session")
        print("2. Run sample queries")

        choice = input("Enter choice (1 or 2): ").strip()

        if choice == "1":
            interactive_query_session(db)
        elif choice == "2":
            test_sample_queries(db)
        else:
            print("Invalid choice, running sample queries...")
            test_sample_queries(db)
    else:
        print("Failed to load database. Please run create_database.py first.")
