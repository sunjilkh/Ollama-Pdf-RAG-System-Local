#!/usr/bin/env python3
"""
🚀 BanglaRAG System Launcher
Simple launcher with all available options
"""

import sys
import os
from config import PREFERRED_MODEL, FALLBACK_MODELS


def print_banner():
    """Print system banner"""
    print("=" * 60)
    print("🚀 BANGLARAG SYSTEM LAUNCHER")
    print("📚 Mixed-Language RAG with Optimized Performance")
    print("🌐 English & Bangla | 🎙️ Voice & Text | ⚡ Optimized")
    print("=" * 60)
    print(f"🤖 Primary Model: {PREFERRED_MODEL}")
    print(f"🔄 Fallback Models: {', '.join(FALLBACK_MODELS)}")
    print("=" * 60)


def main():
    """Main launcher"""
    print_banner()

    print("\n📋 AVAILABLE LAUNCHERS:")
    print("1. 🎛️  Full Menu System          → python main.py")
    print(
        '2. 💬 Direct Chat Session       → python -c "from ollama_llm import interactive_rag_session; interactive_rag_session()"'
    )
    print("3. 🧪 Test Optimized System     → python test_optimized_system.py")
    print("4. 🎤 Voice Input Session       → python voice_input.py --interactive")
    print("5. 📊 Quick Test Suite          → python test_rag.py")
    print("6. 🔧 This Launcher             → python launch.py")

    print("\n🎯 RECOMMENDED FOR DAILY USE:")
    print("   → python main.py (Option 4: Interactive Chat)")

    print("\n📖 QUICK SETUP:")
    print("1. Install model: ollama pull qwen2:1.5b")
    print("2. Create database: python main.py → Option 2")
    print("3. Start chatting: python main.py → Option 4")

    choice = input("\n🚀 Launch directly? (1-6 or Enter to exit): ").strip()

    if choice == "1":
        os.system("python main.py")
    elif choice == "2":
        from ollama_llm import interactive_rag_session

        interactive_rag_session()
    elif choice == "3":
        os.system("python test_optimized_system.py")
    elif choice == "4":
        os.system("python voice_input.py --interactive")
    elif choice == "5":
        os.system("python test_rag.py")
    elif choice == "6":
        print("🔄 Restarting launcher...")
        main()
    else:
        print("👋 Goodbye!")


if __name__ == "__main__":
    main()
