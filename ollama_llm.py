import requests
import json
from query_database import load_database, query_database, generate_prompt_template


def query_ollama(
    prompt,
    model="phi3",
    max_tokens=512,
    temperature=0.7,
    base_url="http://localhost:11434",
):
    """
    Query Ollama API to generate a response.

    Args:
        prompt: The prompt to send to the model
        model: Model name (e.g., 'phi3', 'mistral', 'codellama')
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0.0-1.0)
        base_url: Ollama server URL

    Returns:
        Generated response text or None if error
    """
    try:
        # Ollama API endpoint
        url = f"{base_url}/api/generate"

        # Request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,  # We want a complete response, not streaming
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }

        print(f"Querying Ollama model: {model}")
        print(f"Prompt length: {len(prompt)} characters")

        # Send request
        response = requests.post(url, json=payload, timeout=60)

        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip()
        else:
            print(f"Error: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return None

    except requests.exceptions.ConnectionError:
        print(
            "Error: Could not connect to Ollama. Make sure Ollama is running on localhost:11434"
        )
        print("Start Ollama with: ollama serve")
        return None
    except requests.exceptions.Timeout:
        print(
            "Error: Request timed out. The model might be too slow or the prompt too long."
        )
        return None
    except Exception as e:
        print(f"Error querying Ollama: {e}")
        return None


def get_available_models(base_url="http://localhost:11434"):
    """Get list of available models from Ollama."""
    try:
        url = f"{base_url}/api/tags"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            return models
        else:
            print(f"Error getting models: HTTP {response.status_code}")
            return []

    except Exception as e:
        print(f"Error getting available models: {e}")
        return []


def test_ollama_connection(model="phi3"):
    """Test connection to Ollama with a simple query."""
    test_prompt = "What is 2+2? Answer briefly."

    print(f"Testing Ollama connection with model: {model}")
    response = query_ollama(test_prompt, model=model, max_tokens=50)

    if response:
        print(f"✅ Ollama is working! Test response: {response}")
        return True
    else:
        print("❌ Ollama connection failed")
        return False


def query_with_fallback_models(
    prompt, preferred_models=None, max_tokens=512, temperature=0.7
):
    """
    Query Ollama with fallback to different models if preferred model fails.

    Args:
        prompt: The prompt to send
        preferred_models: List of models to try in order
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Tuple of (response, model_used) or (None, None) if all fail
    """
    if preferred_models is None:
        preferred_models = ["phi3", "mistral", "llama2", "codellama", "phi"]

    # Get available models
    available_models = get_available_models()

    if not available_models:
        print("No models available in Ollama")
        return None, None

    print(f"Available models: {available_models}")

    # Filter preferred models to only those available
    models_to_try = [
        model
        for model in preferred_models
        if any(model in available for available in available_models)
    ]

    # If no preferred models available, use the first available model
    if not models_to_try:
        models_to_try = [available_models[0]]

    print(f"Trying models in order: {models_to_try}")

    for model in models_to_try:
        print(f"Attempting with model: {model}")
        response = query_ollama(
            prompt, model=model, max_tokens=max_tokens, temperature=temperature
        )

        if response:
            print(f"✅ Success with model: {model}")
            return response, model
        else:
            print(f"❌ Failed with model: {model}")

    print("All models failed")
    return None, None


def run_rag_query(query, db=None, k=5, max_tokens=512):
    """
    Complete RAG pipeline: query database, generate prompt, get LLM response.

    Args:
        query: User question
        db: Vector database (will load if None)
        k: Number of relevant chunks to retrieve
        max_tokens: Maximum tokens for LLM response

    Returns:
        Dictionary with query results and metadata
    """
    # Load database if not provided
    if db is None:
        db = load_database()
        if db is None:
            return {
                "query": query,
                "answer": "Database not available. Please create the vector database first.",
                "success": False,
            }

    print(f"\n🔍 Processing query: '{query}'")

    # Retrieve relevant documents
    results = query_database(db, query, k=k)

    if not results:
        return {
            "query": query,
            "answer": "No relevant information found in the database.",
            "success": False,
            "sources": [],
        }

    # Generate prompt
    prompt = generate_prompt_template(query, results)

    # Get LLM response
    answer, model_used = query_with_fallback_models(prompt, max_tokens=max_tokens)

    if answer:
        sources = [
            {"page": doc.metadata.get("page"), "id": doc.metadata.get("id")}
            for doc in results
        ]
        return {
            "query": query,
            "answer": answer,
            "model_used": model_used,
            "success": True,
            "sources": sources,
            "num_sources": len(sources),
        }
    else:
        return {
            "query": query,
            "answer": "Failed to generate response from LLM.",
            "success": False,
            "sources": [],
        }


def interactive_rag_session():
    """Run an interactive RAG session."""
    print("🤖 Interactive RAG Chat Session")
    print("Loading database...")

    db = load_database()
    if db is None:
        print("❌ Could not load database. Please run create_database.py first.")
        return

    # Test Ollama connection
    if not test_ollama_connection():
        print("❌ Ollama is not working. Please ensure:")
        print("1. Ollama is installed and running (ollama serve)")
        print("2. You have a model installed (ollama pull phi3)")
        return

    print("\n✅ System ready! Ask questions about the documents.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            query = input("Q: ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not query:
                continue

            # Process query
            result = run_rag_query(query, db=db)

            # Display results
            print(f"\nA: {result['answer']}")

            if result["success"] and result.get("sources"):
                print(
                    f"\n📚 Sources: {result['num_sources']} relevant chunks from pages {[s['page'] for s in result['sources']]}"
                )
                print(f"🤖 Model used: {result.get('model_used', 'Unknown')}")

            print("-" * 50)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    print("Starting Ollama LLM interface...")

    # Test connection first
    available_models = get_available_models()
    if available_models:
        print(f"Available models: {available_models}")

        # Test with first available model
        test_model = available_models[0]
        if test_ollama_connection(test_model):
            print("✅ Ollama is ready!")

            # Run interactive session
            interactive_rag_session()
        else:
            print("❌ Ollama connection test failed")
    else:
        print("❌ No models available. Please install a model with: ollama pull phi3")
