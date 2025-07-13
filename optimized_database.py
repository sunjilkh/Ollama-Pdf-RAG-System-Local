#!/usr/bin/env python3
"""
Optimized database manager with caching and performance improvements
"""

import time
import threading
from typing import Optional, List, Dict, Any
from functools import lru_cache
import os


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
            self.persist_directory = "db"
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
            from embedding import get_embedding_function_with_fallback

            self.embedding_function = get_embedding_function_with_fallback()
        return self.embedding_function

    def get_database(self):
        """Get cached database instance"""
        if self.db_cache is None:
            self.db_cache = self._load_database()
        return self.db_cache

    def _load_database(self):
        """Load database with optimizations"""
        try:
            if not os.path.exists(self.persist_directory):
                print(f"Database directory '{self.persist_directory}' does not exist.")
                return None

            from langchain_chroma import Chroma

            embedding_function = self.get_embedding_function()

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

    def query_database_cached(self, query: str, k: int = 3) -> List[Any]:
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
            "time complexity",
            "binary search",
            "graph",
            "tree",
            "dynamic programming",
        ]

        print("🔥 Warming up query cache...")
        for query in common_queries:
            self.query_database_cached(query, k=3)

        print(f"✅ Cache warmed up with {len(common_queries)} queries")


# Global instance
db_manager = OptimizedDatabaseManager()


def generate_optimized_prompt_v2(
    query: str, results: List[Any], max_length: int = 1200
) -> str:
    """Generate super optimized prompt with minimal length"""

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

        if current_length + len(source_info) < max_length:
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


def optimized_rag_query_v2(query: str, k: int = 3) -> Dict[str, Any]:
    """Ultra-optimized RAG query with maximum performance"""

    start_time = time.time()

    # Skip translation for English (major optimization)
    if query.lower().replace(" ", "").isascii() and not any(
        char in query for char in "অআইউএওকখগঘচছজঝটঠড"
    ):
        # English query - skip translation
        processed_query = query
        translation_needed = False
        print(f"🔤 English query - skipping translation")
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
        translation_needed = True

    # Use cached database query
    results = db_manager.query_database_cached(processed_query, k=k)

    if not results:
        return {
            "query": query,
            "answer": "No relevant information found.",
            "success": False,
            "sources": [],
            "processing_time": time.time() - start_time,
        }

    # Generate ultra-optimized prompt
    prompt = generate_optimized_prompt_v2(query, results)

    # Use optimized model query
    from optimized_models import model_manager

    answer, model_used = model_manager.query_with_smart_fallback(
        prompt,
        max_tokens=200,  # Further reduced for speed
        temperature=0.2,  # Further reduced for speed
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
            "translation_needed": translation_needed,
        }
    else:
        return {
            "query": query,
            "answer": "Failed to generate response from LLM.",
            "success": False,
            "sources": [],
            "processing_time": processing_time,
            "translation_needed": translation_needed,
        }


def test_optimized_database():
    """Test optimized database performance"""

    print("🚀 Testing Optimized Database Performance...")

    # Test database loading
    start = time.time()
    db = db_manager.get_database()
    db_time = time.time() - start
    print(f"📊 Database load time: {db_time:.3f}s")

    if not db:
        print("❌ Database failed to load")
        return

    # Test cached queries
    test_queries = [
        "what is algorithm",
        "sorting algorithm",
        "binary search",
        "data structure",
        "graph algorithms",
    ]

    print("\n🔍 Testing cached queries...")

    total_time = 0
    for i, query in enumerate(test_queries, 1):
        print(f"\n🎯 Query {i}: {query}")
        start = time.time()
        result = optimized_rag_query_v2(query)
        query_time = time.time() - start
        total_time += query_time

        print(f"⏱️  Time: {query_time:.3f}s")
        print(f"✅ Success: {result['success']}")
        if result["success"]:
            print(f"📝 Answer: {result['answer'][:80]}...")

    # Test cache performance
    print(f"\n📈 Cache Performance:")
    cache_stats = db_manager.get_cache_stats()
    print(f"  🎯 Cache hit rate: {cache_stats['hit_rate']:.1%}")
    print(f"  📊 Cache size: {cache_stats['cache_size']}")

    avg_time = total_time / len(test_queries)
    print(f"\n🏆 FINAL PERFORMANCE:")
    print(f"  ⏱️  Average time: {avg_time:.3f}s")
    print(f"  🎯 Target: <5.0s")
    print(
        f"  🔥 Status: {'✅ EXCELLENT' if avg_time < 2 else '✅ GOOD' if avg_time < 5 else '❌ NEEDS MORE WORK'}"
    )


if __name__ == "__main__":
    test_optimized_database()
