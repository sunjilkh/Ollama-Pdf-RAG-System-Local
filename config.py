#!/usr/bin/env python3
"""
Centralized configuration for BanglaRAG system
All optimizations are now fully integrated into the main project files
"""

# Model Configuration
PREFERRED_MODEL = "qwen2:1.5b"  # Fast and efficient
FALLBACK_MODELS = ["qwen2:1.5b", "phi3", "mistral", "llama2"]

# Performance Settings (Optimized for speed)
MAX_TOKENS = 180  # Reduced for faster inference
TEMPERATURE = 0.1  # Reduced for faster, more focused responses
TIMEOUT = 25  # Reduced for faster failure detection

# Database Configuration
DATABASE_DIRECTORY = "db"
RETRIEVAL_COUNT = 3  # Optimized retrieval count

# Translation Settings
SKIP_TRANSLATION_FOR_ENGLISH = True
CACHE_TRANSLATIONS = True

# Voice Input Settings
DEFAULT_RECORDING_DURATION = 5
WHISPER_MODEL_SIZE = "base"

# System Settings
OLLAMA_BASE_URL = "http://localhost:11434"
ENABLE_CACHING = True
ENABLE_OPTIMIZATION = True

print("📋 BanglaRAG Configuration Loaded")
print(f"🤖 Preferred Model: {PREFERRED_MODEL}")
print(f"🔄 Fallback Models: {FALLBACK_MODELS}")
print(f"⚡ Performance Mode: Fully Integrated & Optimized")
print(f"🚀 All optimizations active in main project files")
