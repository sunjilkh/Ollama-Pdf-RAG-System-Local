import requests
import json
import time
import threading
from typing import Dict, Optional, Any
from functools import lru_cache
from query_database import load_database, query_database, generate_prompt_template
from translator import process_query_with_translation
from config import PREFERRED_MODEL, FALLBACK_MODELS, MAX_TOKENS, TEMPERATURE, TIMEOUT


class OptimizedModelManager:
    """Singleton model manager with caching and optimization"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(OptimizedModelManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            self.initialized = True
            self.base_url = "http://localhost:11434"
            self.preferred_model = PREFERRED_MODEL
            self.fallback_models = FALLBACK_MODELS
            self.model_cache = {}
            self.available_models = None
            self.model_warmed_up = False
            self._warm_up_models()

    def _warm_up_models(self):
        """Warm up models in background to reduce first-query latency"""

        def warm_up():
            try:
                # Test connection and warm up preferred model
                self.get_available_models()
                if self.available_models:
                    # Send a small warm-up query
                    self.query_ollama_optimized(
                        "Test",
                        model=self.preferred_model,
                        max_tokens=1,
                        temperature=0.1,
                    )
                    self.model_warmed_up = True
                    print("🔥 Model warm-up completed")
            except Exception as e:
                print(f"⚠️  Model warm-up failed: {e}")

        # Run warm-up in background
        threading.Thread(target=warm_up, daemon=True).start()

    @lru_cache(maxsize=1)
    def get_available_models(self):
        """Cache available models list"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json()
                model_names = [model["name"] for model in models.get("models", [])]
                self.available_models = model_names
                return model_names
        except Exception as e:
            print(f"⚠️  Failed to get available models: {e}")
        return []

    def query_ollama_optimized(
        self,
        prompt: str,
        model: str = None,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
        timeout: int = TIMEOUT,
    ) -> Optional[str]:
        """Optimized Ollama query with reduced parameters for speed"""

        model = model or self.preferred_model

        # Check cache first (for identical prompts)
        cache_key = f"{model}:{hash(prompt)}:{max_tokens}:{temperature}"
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]

        try:
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "num_ctx": 1536,  # Reduced context window for speed
                    "repeat_last_n": 32,  # Reduced for speed
                    "repeat_penalty": 1.05,  # Reduced for speed
                    "top_k": 20,  # Reduced for speed
                    "top_p": 0.8,  # Reduced for speed
                },
            }

            response = requests.post(url, json=payload, timeout=timeout)

            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "").strip()

                # Cache the result
                self.model_cache[cache_key] = answer

                # Limit cache size
                if len(self.model_cache) > 100:
                    # Remove oldest entries
                    oldest_key = next(iter(self.model_cache))
                    del self.model_cache[oldest_key]

                return answer
            else:
                print(f"❌ HTTP {response.status_code}: {response.text}")
                return None

        except requests.exceptions.Timeout:
            print(f"⏱️  Request timed out after {timeout}s")
            return None
        except Exception as e:
            print(f"❌ Error querying {model}: {e}")
            return None

    def query_with_smart_fallback(
        self,
        prompt: str,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
    ) -> tuple[Optional[str], Optional[str]]:
        """Smart fallback with optimized parameters"""

        # Get available models
        available = self.get_available_models()
        if not available:
            return None, None

        # Filter models to only available ones
        models_to_try = [
            model
            for model in self.fallback_models
            if any(model in available_model for available_model in available)
        ]

        if not models_to_try and available:
            models_to_try = [available[0]]

        for model in models_to_try:
            result = self.query_ollama_optimized(
                prompt, model=model, max_tokens=max_tokens, temperature=temperature
            )

            if result:
                return result, model

        return None, None

    def is_connection_healthy(self) -> bool:
        """Quick health check for Ollama connection"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


# Global optimized model manager instance
model_manager = OptimizedModelManager()


def query_ollama(
    prompt,
    model=PREFERRED_MODEL,  # Updated to use config
    max_tokens=MAX_TOKENS,  # Updated to use config
    temperature=TEMPERATURE,  # Updated to use config
    base_url="http://localhost:11434",
):
    """
    Query Ollama API to generate a response using optimized model manager.

    This function now uses the OptimizedModelManager for improved performance.
    """
    return model_manager.query_ollama_optimized(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=TIMEOUT,
    )


def get_available_models(base_url="http://localhost:11434"):
    """Get list of available models from Ollama using optimized manager."""
    return model_manager.get_available_models()


def test_ollama_connection(model=PREFERRED_MODEL):
    """Test connection to Ollama with a simple query using optimized manager."""
    test_prompt = "What is 2+2? Answer briefly."

    try:
        response = model_manager.query_ollama_optimized(
            test_prompt, model=model, max_tokens=50
        )
        if response:
            print(f"✅ Ollama connection successful with model: {model}")
            return True
        else:
            print(f"❌ Ollama query failed with model: {model}")
            return False
    except Exception as e:
        print(f"❌ Ollama connection error: {e}")
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
        preferred_models = [
            "qwen2:1.5b",
            "phi3",
            "mistral",
            "llama2",
        ]  # Updated to match optimized system

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


def generate_optimized_prompt_template(query, results, max_context_length=1500):
    """
    Generate optimized prompt template with reduced length for faster inference.
    """
    if not results:
        return f"Answer briefly: {query}\n\nAnswer:"

    # Combine context from results with optimization
    contexts = []
    current_length = 0

    for result in results:
        content = result.page_content.strip()

        # Take only the most relevant part (first 200 chars)
        if len(content) > 200:
            content = content[:200] + "..."

        # Simple page reference
        page = result.metadata.get("page", "N/A")
        source_info = f"[Page {page}] {content}"

        if current_length + len(source_info) < max_context_length:
            contexts.append(source_info)
            current_length += len(source_info)
        else:
            break

    context = "\n".join(contexts)

    # Shorter, more direct prompt
    prompt = f"""Based on the context below, answer the question briefly and cite page numbers.

Context:
{context}

Question: {query}

Answer:"""

    return prompt


def query_with_optimized_fallback(
    prompt, preferred_models=None, max_tokens=MAX_TOKENS, temperature=TEMPERATURE
):
    """
    Optimized query with fallback to different models with faster parameters.
    Now uses the OptimizedModelManager.
    """
    return model_manager.query_with_smart_fallback(
        prompt=prompt, max_tokens=max_tokens, temperature=temperature
    )


def optimized_rag_query(query: str, k: int = 3) -> Dict[str, Any]:
    """
    Fully optimized RAG query with maximum performance integration
    """
    from query_database import db_manager, generate_optimized_prompt_template

    start_time = time.time()

    # Empty Query Handling - Add proper validation
    if not query or not query.strip():
        return {
            "query": query,
            "answer": "Empty query provided. Please ask a specific question.",
            "success": False,
            "sources": [],
            "processing_time": time.time() - start_time,
            "translation_info": {
                "original_query": query,
                "processed_query": query,
                "language_detected": "unknown",
                "translation_needed": False,
                "translation_result": None,
                "success": False,
                "error": "Empty query",
            },
        }

    # Optimize language detection
    query_clean = query.strip().lower()

    # Skip translation for English (major optimization)
    if query_clean.replace(" ", "").isascii() and not any(
        char in query for char in "অআইউএওকখগঘচছজঝটঠড"
    ):
        # English query - skip translation
        processed_query = query
        translation_info = {
            "original_query": query,
            "processed_query": query,
            "language_detected": "english",
            "translation_needed": False,
            "translation_result": None,
            "success": True,
        }
        print(f"🔤 English query - skipping translation")
    else:
        # Non-English query - use translation
        translation_info = process_query_with_translation(query)
        if not translation_info["success"]:
            return {
                "query": query,
                "answer": f"Query processing failed: {translation_info.get('error', 'Unknown error')}",
                "success": False,
                "sources": [],
                "processing_time": time.time() - start_time,
                "translation_info": translation_info,
            }
        processed_query = translation_info["processed_query"]

    # Use cached database query
    results = db_manager.query_database_cached(processed_query, k=k)

    if not results:
        return {
            "query": query,
            "answer": "No relevant information found.",
            "success": False,
            "sources": [],
            "processing_time": time.time() - start_time,
            "translation_info": translation_info,
        }

    # Generate ultra-optimized prompt
    prompt = generate_optimized_prompt_template(query, results)

    # Use optimized model query with reduced parameters for speed
    answer, model_used = model_manager.query_with_smart_fallback(
        prompt,
        max_tokens=180,  # Further reduced for speed
        temperature=0.1,  # Further reduced for speed
    )

    processing_time = time.time() - start_time

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
            "processing_time": processing_time,
            "prompt_length": len(prompt),
            "translation_info": translation_info,
        }
    else:
        return {
            "query": query,
            "answer": "Failed to generate response from LLM.",
            "success": False,
            "sources": [],
            "processing_time": processing_time,
            "translation_info": translation_info,
        }


def run_rag_query(query, db=None, k=3, max_tokens=256):
    """
    Main RAG pipeline - now fully optimized and integrated.

    This function uses all optimizations:
    - Optimized model management with caching
    - Optimized database management with caching
    - Optimized translation handling
    - Optimized prompt generation

    Args:
        query: User question (in any supported language)
        db: Vector database (ignored, uses optimized manager)
        k: Number of relevant chunks to retrieve
        max_tokens: Maximum tokens for LLM response

    Returns:
        Dictionary with query results and metadata
    """
    # Use fully optimized pipeline
    return optimized_rag_query(query, k=k)


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
        print(
            "2. You have a model installed (ollama pull qwen2:1.5b or ollama pull phi3)"
        )
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

            # Display translation info if available
            if result.get("translation_info") and result["translation_info"].get(
                "translation_needed"
            ):
                translation_info = result["translation_info"]
                print(
                    f"\n🌐 Translation: '{translation_info['original_query']}' → '{translation_info['processed_query']}'"
                )
                print(f"🔤 Detected language: {translation_info['language_detected']}")

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
        print(
            "❌ No models available. Please install a model with: ollama pull qwen2:1.5b"
        )
