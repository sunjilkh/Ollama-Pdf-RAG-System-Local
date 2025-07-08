from langchain_ollama import OllamaEmbeddings


def get_embedding_function():
    """
    Returns an Ollama embedding function for generating text embeddings.

    Popular embedding models for Ollama include:
    - nomic-embed-text (recommended for general use)
    - mxbai-embed-large (good performance)
    - all-minilm (lightweight)
    - llama2 (basic fallback)
    """
    return OllamaEmbeddings(
        model="nomic-embed-text",  # You can change this to your preferred model
        # base_url="http://localhost:11434"  # Default Ollama URL
    )


def get_embedding_function_with_fallback():
    """
    Returns an embedding function with fallback models if the primary model isn't available.
    """
    models_to_try = ["nomic-embed-text", "mxbai-embed-large", "all-minilm", "llama2"]

    for model in models_to_try:
        try:
            embedding_function = OllamaEmbeddings(model=model)
            # Test if the model works
            embedding_function.embed_query("test")
            print(f"Successfully using model: {model}")
            return embedding_function
        except Exception as e:
            print(f"Model {model} not available: {e}")
            continue

    raise Exception(
        "No working Ollama embedding models found. Please ensure Ollama is running and has embedding models installed."
    )


# Test function to verify embeddings are working
def test_embeddings():
    """Test the embedding function with sample text."""
    try:
        print("Testing embedding function with fallback...")
        embedding_function = get_embedding_function_with_fallback()
        test_text = "This is a test sentence for embedding."

        print(f"Input text: {test_text}")

        # Generate embedding
        embedding = embedding_function.embed_query(test_text)

        print(f"Embedding generated successfully!")
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")

        return True
    except Exception as e:
        print(f"Error testing embeddings: {e}")
        print("\nTo fix this, you need to:")
        print("1. Install Ollama: https://ollama.ai/")
        print("2. Pull an embedding model: 'ollama pull nomic-embed-text'")
        print("3. Ensure Ollama is running: 'ollama serve'")
        return False


if __name__ == "__main__":
    test_embeddings()
