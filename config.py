#!/usr/bin/env python3
"""
Centralized configuration for BanglaRAG system
"""

# Model Configuration
PREFERRED_MODEL = "qwen2:1.5b"  # Fast and efficient
FALLBACK_MODELS = ["qwen2:1.5b", "phi3", "mistral", "llama2"]

# Performance Settings
MAX_TOKENS = 256
TEMPERATURE = 0.3
TIMEOUT = 30

# Database Configuration
DATABASE_DIRECTORY = "db"
RETRIEVAL_COUNT = 3  # Number of documents to retrieve

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
print(f"⚡ Performance Mode: {'Enabled' if ENABLE_OPTIMIZATION else 'Disabled'}")
