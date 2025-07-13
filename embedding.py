from langchain_ollama import OllamaEmbeddings
from langdetect import detect
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import warnings

warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

# Global variables for Bangla model (load once)
_bangla_tokenizer = None
_bangla_model = None


def load_bangla_model():
    """
    Load BanglaBERT model for Bangla text embedding.
    This function loads the model only once and reuses it.
    """
    global _bangla_tokenizer, _bangla_model

    if _bangla_tokenizer is None or _bangla_model is None:
        print("Loading BanglaBERT model for Bangla text embedding...")
        try:
            _bangla_tokenizer = AutoTokenizer.from_pretrained(
                "sagorsarker/bangla-bert-base"
            )
            _bangla_model = AutoModel.from_pretrained("sagorsarker/bangla-bert-base")
            print("✅ BanglaBERT model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading BanglaBERT model: {e}")
            print("Falling back to English model for Bangla text")
            return None, None

    return _bangla_tokenizer, _bangla_model


def embed_bangla(text):
    """
    Generate embedding for Bangla text using BanglaBERT.

    Args:
        text (str): Bangla text to embed

    Returns:
        np.ndarray: 768-dimensional embedding vector
    """
    tokenizer, model = load_bangla_model()

    if tokenizer is None or model is None:
        # Fallback to English embedding if Bangla model fails
        print("Using English model as fallback for Bangla text")
        return embed_english(text)

    try:
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Use mean pooling of the last hidden states
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()

        return embeddings.numpy()

    except Exception as e:
        print(f"Error generating Bangla embedding: {e}")
        print("Falling back to English embedding")
        return embed_english(text)


def embed_english(text):
    """
    Generate embedding for English text using Ollama's nomic-embed-text.

    Args:
        text (str): English text to embed

    Returns:
        np.ndarray: Embedding vector
    """
    try:
        embedding_function = get_embedding_function_with_fallback()
        embedding = embedding_function.embed_query(text)
        return np.array(embedding)
    except Exception as e:
        print(f"Error generating English embedding: {e}")
        raise


def detect_language(text):
    """
    Detect the language of the given text.

    Args:
        text (str): Text to analyze

    Returns:
        str: Language code ('en' for English, 'bn' for Bangla, etc.)
    """
    try:
        # Clean the text - remove extra whitespace and empty lines
        cleaned_text = " ".join(text.split())

        # Skip very short texts
        if len(cleaned_text) < 10:
            return "en"  # Default to English for short texts

        # Detect language
        language = detect(cleaned_text)
        return language

    except Exception as e:
        print(f"Language detection failed: {e}")
        return "en"  # Default to English if detection fails


def get_mixed_language_embedding(text):
    """
    Generate embedding for text using language-appropriate model.

    Args:
        text (str): Text to embed (any language)

    Returns:
        np.ndarray: Embedding vector
    """
    # Detect language
    language = detect_language(text)

    # Choose appropriate embedding model
    if language == "bn":
        print(f"Detected Bangla text, using BanglaBERT")
        return embed_bangla(text)
    else:
        print(f"Detected {language} text, using English model")
        return embed_english(text)


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


def test_mixed_language_embeddings():
    """Test mixed language embedding functionality."""
    print("\n🧪 Testing Mixed Language Embeddings...")

    # Test texts in different languages
    test_texts = [
        "This is an English sentence about algorithms.",
        "এটি একটি বাংলা বাক্য যা কম্পিউটার বিজ্ঞান সম্পর্কে।",
        "Machine learning is a subset of artificial intelligence.",
        "গণিত এবং পদার্থবিদ্যা খুবই গুরুত্বপূর্ণ বিষয়।",
    ]

    results = []

    for i, text in enumerate(test_texts):
        print(f"\n--- Test {i+1} ---")
        print(f"Text: {text}")

        try:
            # Detect language
            lang = detect_language(text)
            print(f"Detected language: {lang}")

            # Generate embedding
            embedding = get_mixed_language_embedding(text)
            print(f"✅ Embedding generated successfully!")
            print(f"Embedding dimension: {len(embedding)}")
            print(f"First 5 values: {embedding[:5]}")

            results.append(
                {
                    "text": text,
                    "language": lang,
                    "embedding_dim": len(embedding),
                    "success": True,
                }
            )

        except Exception as e:
            print(f"❌ Error: {e}")
            results.append(
                {
                    "text": text,
                    "language": "unknown",
                    "embedding_dim": 0,
                    "success": False,
                }
            )

    # Summary
    print("\n📊 Test Summary:")
    successful = sum(1 for r in results if r["success"])
    print(f"Successful: {successful}/{len(results)}")

    for r in results:
        status = "✅" if r["success"] else "❌"
        print(f"{status} {r['language']}: {r['embedding_dim']}D")

    return results


if __name__ == "__main__":
    # Test both embedding functions
    print("🔍 Testing original embedding function:")
    test_embeddings()

    print("\n" + "=" * 50)

    # Test mixed language embedding
    test_mixed_language_embeddings()
