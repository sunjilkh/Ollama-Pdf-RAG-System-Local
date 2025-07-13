#!/usr/bin/env python3
"""
Optimized model manager with caching and performance improvements
"""

import time
import threading
from typing import Dict, Optional, Any
from functools import lru_cache
import requests
import json


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
            self.preferred_model = "qwen2:1.5b"
            self.fallback_models = ["qwen2:1.5b", "phi3", "mistral", "llama2"]
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
        max_tokens: int = 256,  # Reduced from 512
        temperature: float = 0.3,  # Reduced from 0.7 for faster inference
        timeout: int = 30,  # Reduced from 60
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
                    "num_ctx": 2048,  # Reduced context window
                    "repeat_last_n": 64,  # Reduce repetition checking
                    "repeat_penalty": 1.1,  # Slightly reduce penalty
                    "top_k": 40,  # Reduce top-k for speed
                    "top_p": 0.9,  # Reduce top-p for speed
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
        self, prompt: str, max_tokens: int = 256, temperature: float = 0.3
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


# Global instance
model_manager = OptimizedModelManager()


def generate_optimized_prompt(query: str, results: list, max_length: int = 1500) -> str:
    """Generate optimized prompt with reduced length for faster inference"""

    if not results:
        return f"""Answer this question briefly: {query}

If you don't know the answer, just say "I don't have enough information to answer this question."

Answer:"""

    # Extract key information more efficiently
    contexts = []
    current_length = 0

    for result in results:
        content = result.page_content.strip()

        # Take only the most relevant part (first 200 chars)
        if len(content) > 200:
            content = content[:200] + "..."

        # Add page info
        page = result.metadata.get("page", "N/A")
        source_info = f"[Page {page}] {content}"

        if current_length + len(source_info) < max_length:
            contexts.append(source_info)
            current_length += len(source_info)
        else:
            break

    context = "\n".join(contexts)

    # Shorter, more direct prompt
    prompt = f"""Based on the following context, answer the question. Be concise and cite page numbers.

Context:
{context}

Question: {query}

Answer:"""

    return prompt


def optimized_rag_query(query: str, db, k: int = 3) -> Dict[str, Any]:
    """Optimized RAG query with performance improvements"""

    start_time = time.time()

    # Skip translation for English (major optimization)
    if query.lower().replace(" ", "").isascii() and not any(
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
        print(f"🔤 English query detected - skipping translation")
    else:
        # Non-English query - use translation
        from translator import process_query_with_translation

        translation_info = process_query_with_translation(query)
        if not translation_info["success"]:
            return {
                "query": query,
                "answer": f"Query processing failed: {translation_info.get('error', 'Unknown error')}",
                "success": False,
                "sources": [],
                "processing_time": time.time() - start_time,
            }
        processed_query = translation_info["processed_query"]

    # Database query with reduced k for speed
    from query_database import query_database

    results = query_database(db, processed_query, k=k)

    if not results:
        return {
            "query": query,
            "original_query": query,
            "search_query": processed_query,
            "answer": "No relevant information found in the database.",
            "success": False,
            "sources": [],
            "translation_info": translation_info,
            "processing_time": time.time() - start_time,
        }

    # Generate optimized prompt
    prompt = generate_optimized_prompt(query, results)

    # Use optimized model query
    answer, model_used = model_manager.query_with_smart_fallback(
        prompt,
        max_tokens=256,  # Reduced for speed
        temperature=0.3,  # Reduced for speed
    )

    processing_time = time.time() - start_time

    if answer:
        sources = [
            {"page": doc.metadata.get("page"), "id": doc.metadata.get("id")}
            for doc in results
        ]
        return {
            "query": query,
            "original_query": query,
            "search_query": processed_query,
            "answer": answer,
            "model_used": model_used,
            "success": True,
            "sources": sources,
            "num_sources": len(sources),
            "translation_info": translation_info,
            "processing_time": processing_time,
            "prompt_length": len(prompt),
        }
    else:
        return {
            "query": query,
            "original_query": query,
            "search_query": processed_query,
            "answer": "Failed to generate response from LLM.",
            "success": False,
            "sources": [],
            "translation_info": translation_info,
            "processing_time": processing_time,
        }


# Test function
def test_optimized_performance():
    """Test optimized performance"""
    from query_database import load_database

    print("🚀 Testing Optimized Performance...")

    # Test database loading
    start = time.time()
    db = load_database()
    db_time = time.time() - start
    print(f"📊 Database load time: {db_time:.3f}s")

    if not db:
        print("❌ Database failed to load")
        return

    # Test queries
    test_queries = [
        "what is algorithm",
        "how does sorting work",
        "explain binary search",
    ]

    total_time = 0
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        start = time.time()
        result = optimized_rag_query(query, db)
        query_time = time.time() - start
        total_time += query_time

        print(f"⏱️  Time: {query_time:.3f}s")
        print(f"✅ Success: {result['success']}")
        if result["success"]:
            print(f"📝 Answer: {result['answer'][:100]}...")

    avg_time = total_time / len(test_queries)
    print(f"\n📈 OPTIMIZED PERFORMANCE:")
    print(f"  ⏱️  Average time: {avg_time:.3f}s")
    print(f"  🎯 Target: <5.0s")
    print(
        f"  🔥 Status: {'✅ EXCELLENT' if avg_time < 2 else '✅ GOOD' if avg_time < 5 else '❌ NEEDS MORE WORK'}"
    )


if __name__ == "__main__":
    test_optimized_performance()
