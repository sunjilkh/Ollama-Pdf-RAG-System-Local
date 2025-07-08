from langchain_chroma import Chroma
from embedding import get_embedding_function_with_fallback
import os


def load_database(persist_directory="db"):
    """
    Load existing ChromaDB vector database.

    Args:
        persist_directory: Directory where database is persisted

    Returns:
        ChromaDB vector store instance or None if failed
    """
    try:
        if not os.path.exists(persist_directory):
            print(f"Database directory '{persist_directory}' does not exist.")
            print("Please run create_database.py first to create the vector database.")
            return None

        embedding_function = get_embedding_function_with_fallback()
        db = Chroma(
            persist_directory=persist_directory, embedding_function=embedding_function
        )

        # Test if database has content
        data = db.get()
        if not data["ids"]:
            print("Database exists but is empty.")
            return None

        print(f"✅ Loaded database with {len(data['ids'])} documents")
        return db

    except Exception as e:
        print(f"Error loading database: {e}")
        return None


def query_database(db, query, k=5):
    """
    Query the database and retrieve relevant chunks.

    Args:
        db: ChromaDB vector store instance
        query: User query string
        k: Number of top results to retrieve

    Returns:
        List of relevant document chunks
    """
    if db is None:
        print("Database is not available")
        return []

    try:
        print(f"Querying database for: '{query}'")
        print(f"Retrieving top {k} relevant chunks...")

        # Perform similarity search
        results = db.similarity_search(query, k=k)

        print(f"Found {len(results)} relevant chunks")
        return results

    except Exception as e:
        print(f"Error querying database: {e}")
        return []


def query_database_with_scores(db, query, k=5):
    """
    Query database and return results with similarity scores.

    Args:
        db: ChromaDB vector store instance
        query: User query string
        k: Number of top results to retrieve

    Returns:
        List of tuples (document, score)
    """
    if db is None:
        print("Database is not available")
        return []

    try:
        print(f"Querying database with scores for: '{query}'")

        # Perform similarity search with scores
        results = db.similarity_search_with_score(query, k=k)

        print(f"Found {len(results)} relevant chunks with scores")
        return results

    except Exception as e:
        print(f"Error querying database with scores: {e}")
        return []


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

    for i, result in enumerate(results):
        content = result.page_content.strip()
        source_info = f"[Source: Page {result.metadata.get('page', 'Unknown')}]"

        # Check if adding this context would exceed the limit
        new_content = f"{source_info}\n{content}\n\n"
        if current_length + len(new_content) > max_context_length:
            break

        contexts.append(new_content)
        current_length += len(new_content)

    context = "".join(contexts).strip()

    prompt_template = f"""Use the following context to answer the question. If the answer cannot be found in the context, say "I don't have enough information to answer this question based on the provided context."

Context:
{context}

Question: {query}

Answer:"""

    return prompt_template


def interactive_query_session(db):
    """Run an interactive query session."""
    if db is None:
        print("Database is not available for interactive session")
        return

    print("\n🔍 Interactive Query Session")
    print("Enter your questions (type 'quit' to exit):")

    while True:
        try:
            query = input("\nQ: ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not query:
                continue

            # Query database
            results = query_database_with_scores(db, query, k=3)

            if not results:
                print("No relevant information found.")
                continue

            # Generate prompt
            docs_only = [doc for doc, score in results]
            prompt = generate_prompt_template(query, docs_only)

            print("\n" + "=" * 50)
            print("GENERATED PROMPT FOR LLM:")
            print("=" * 50)
            print(prompt)
            print("=" * 50)

            # Show similarity scores
            print("\nRelevance Scores:")
            for i, (doc, score) in enumerate(results, 1):
                print(
                    f"  Result {i}: {score:.4f} (Page {doc.metadata.get('page', 'N/A')})"
                )

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def test_sample_queries(db):
    """Test the system with sample queries."""
    sample_queries = [
        "What is an algorithm?",
        "How does merge sort work?",
        "What is the time complexity of quick sort?",
        "Explain binary search trees",
        "What is dynamic programming?",
    ]

    print("\n🧪 Testing Sample Queries:")
    print("=" * 50)

    for query in sample_queries:
        print(f"\nQuery: {query}")
        results = query_database(db, query, k=2)

        if results:
            prompt = generate_prompt_template(query, results, max_context_length=1000)
            print(f"Generated prompt length: {len(prompt)} characters")
            print(f"Context sources: Pages {[r.metadata.get('page') for r in results]}")
        else:
            print("No relevant results found")

        print("-" * 30)


if __name__ == "__main__":
    print("Starting query database system...")

    # Load database
    db = load_database()

    if db:
        # Test with sample queries
        test_sample_queries(db)

        # Start interactive session
        interactive_query_session(db)
    else:
        print(
            "Please ensure the vector database is created first by running create_database.py"
        )
