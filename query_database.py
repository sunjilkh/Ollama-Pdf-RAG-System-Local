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

        print(f"✅ Database loaded successfully with {len(data['ids'])} documents")
        return db

    except Exception as e:
        print(f"❌ Error loading database: {e}")
        return None


def query_database(db, query, k=5):
    """
    Query the vector database and return relevant documents.

    Args:
        db: ChromaDB vector store instance
        query: Search query string
        k: Number of documents to retrieve

    Returns:
        List of relevant document chunks
    """
    if db is None:
        print("Database is not available")
        return []

    try:
        # Search for relevant documents
        results = db.similarity_search(query, k=k)
        print(f"Found {len(results)} relevant documents")
        return results

    except Exception as e:
        print(f"Error querying database: {e}")
        return []


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
