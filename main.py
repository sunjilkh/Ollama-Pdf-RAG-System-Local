#!/usr/bin/env python3
"""
Main BanglaRAG System Runner
Provides a unified interface to the complete BanglaRAG (Retrieval-Augmented Generation) system
with support for mixed-language (Bangla & English) text and voice input.
"""

import sys
import os
from datetime import datetime


def print_banner():
    """Print system banner."""
    print("=" * 60)
    print("🎤 BANGLARAG SYSTEM")
    print("📚 Mixed-Language RAG with Voice Input Support")
    print("🌐 Supporting English & Bangla | 🎙️ Voice & Text Input")
    print("=" * 60)


def check_dependencies():
    """Check if all required components are available."""
    issues = []

    # Check if database exists
    if not os.path.exists("db"):
        issues.append("Vector database not found. Run option 2 to create it.")

    # Check if PDF exists
    pdf_files = [f for f in os.listdir(".") if f.endswith(".pdf")]
    if not pdf_files:
        issues.append("No PDF files found in current directory.")

    # Check Ollama connection
    try:
        from embedding import get_embedding_function_with_fallback

        get_embedding_function_with_fallback()
        print("✅ Embedding function working")
    except Exception as e:
        issues.append(f"Embedding function failed: {e}")

    # Check if we can load database
    try:
        from query_database import load_database

        db = load_database()
        if db:
            print("✅ Vector database accessible")
        else:
            issues.append("Cannot load vector database")
    except Exception as e:
        issues.append(f"Database loading failed: {e}")

    # Check Ollama LLM
    try:
        from ollama_llm import get_available_models, test_ollama_connection

        models = get_available_models()
        if models:
            print(f"✅ Ollama models available: {models}")
            # Filter out embedding models and test with a generative model
            generative_models = [
                m
                for m in models
                if not any(embed in m.lower() for embed in ["embed", "embedding"])
            ]
            if generative_models:
                if test_ollama_connection(generative_models[0]):
                    print("✅ Ollama LLM working")
                else:
                    issues.append("Ollama LLM connection failed")
            else:
                issues.append(
                    "No generative Ollama models available (only embedding models found)"
                )
        else:
            issues.append("No Ollama models available")
    except Exception as e:
        issues.append(f"Ollama check failed: {e}")

    return issues


def run_document_processing():
    """Run the complete document processing pipeline."""
    print("\n🔄 Starting Document Processing Pipeline...")

    try:
        print("Step 1: Loading PDF documents...")
        from loader import documents

        print(f"✅ Loaded {len(documents)} pages")

        print("Step 2: Splitting documents into chunks...")
        from split import chunks

        print(f"✅ Created {len(chunks)} chunks")

        print("Step 3: Assigning unique IDs...")
        from assign_ids import assign_unique_ids

        chunks_with_ids = assign_unique_ids(chunks.copy())
        print(f"✅ Assigned IDs to {len(chunks_with_ids)} chunks")

        print("Step 4: Creating vector database...")
        from create_database import create_or_update_database

        db = create_or_update_database(chunks_with_ids)

        if db:
            print("✅ Document processing completed successfully!")
            return True
        else:
            print("❌ Database creation failed")
            return False

    except Exception as e:
        print(f"❌ Error in document processing: {e}")
        return False


def run_interactive_chat():
    """Run interactive chat session."""
    print("\n💬 Starting Interactive Chat Session...")

    try:
        from ollama_llm import interactive_rag_session

        interactive_rag_session()
    except KeyboardInterrupt:
        print("\nChat session ended by user.")
    except Exception as e:
        print(f"❌ Error in chat session: {e}")


def run_database_query_test():
    """Test database querying functionality."""
    print("\n🔍 Testing Database Query Functionality...")

    try:
        from query_database import load_database, test_sample_queries

        db = load_database()
        if db:
            test_sample_queries(db)
            print("✅ Database query test completed")
        else:
            print("❌ Could not load database")
    except Exception as e:
        print(f"❌ Error in database query test: {e}")


def run_system_tests():
    """Run comprehensive system tests."""
    print("\n🧪 Running System Tests...")

    try:
        from test_rag import RAGTester
        from query_database import load_database

        db = load_database()
        if not db:
            print("❌ Database not available for testing")
            return

        tester = RAGTester(db)
        print("Running test suite...")
        tester.run_test_suite()
        report = tester.generate_test_report()

        if report:
            pass_rate = report["test_summary"]["pass_rate"]
            if pass_rate >= 70:
                print("✅ System tests PASSED (≥70% pass rate)")
            else:
                print("⚠️ System tests PARTIALLY PASSED (<70% pass rate)")

    except Exception as e:
        print(f"❌ Error in system tests: {e}")


def show_system_status():
    """Show current system status."""
    print("\n📊 SYSTEM STATUS")
    print("-" * 40)

    # PDF files
    pdf_files = [f for f in os.listdir(".") if f.endswith(".pdf")]
    print(f"📄 PDF Files: {len(pdf_files)}")
    for pdf in pdf_files:
        size_mb = os.path.getsize(pdf) / (1024 * 1024)
        print(f"   - {pdf} ({size_mb:.1f} MB)")

    # Database status
    if os.path.exists("db"):
        try:
            from query_database import load_database

            db = load_database()
            if db:
                data = db.get()
                doc_count = len(data["ids"]) if data["ids"] else 0
                print(f"🗄️ Vector Database: {doc_count} documents")
            else:
                print("🗄️ Vector Database: Exists but empty")
        except:
            print("🗄️ Vector Database: Exists but error loading")
    else:
        print("🗄️ Vector Database: Not created")

    # Ollama status
    try:
        from ollama_llm import get_available_models

        models = get_available_models()
        print(f"🤖 Ollama Models: {len(models)} available")
        for model in models[:3]:  # Show first 3
            print(f"   - {model}")
    except:
        print("🤖 Ollama Models: Not available")


def run_voice_input_session():
    """Run interactive voice input session."""
    print("\n🎤 Starting Voice Input Session...")

    try:
        # Check if voice input module is available
        from voice_input import interactive_voice_session

        print("✅ Voice input module loaded successfully")
        print("\n📋 Instructions:")
        print("- Press Enter to start recording")
        print("- Speak clearly in English or Bangla")
        print("- Default recording duration: 5 seconds")
        print("- Type 'quit' to exit")

        interactive_voice_session()

    except ImportError as e:
        print(f"❌ Voice input module not available: {e}")
        print("\n📋 To enable voice input:")
        print("1. Install dependencies: pip install openai-whisper pyaudio")
        print("2. Ensure microphone is connected")
    except Exception as e:
        print(f"❌ Error in voice input session: {e}")


def run_voice_input_test():
    """Run voice input demonstration and tests."""
    print("\n🧪 Running Voice Input Tests...")

    try:
        from test_voice_demo import main as run_voice_demo

        print("✅ Voice demo module loaded successfully")
        run_voice_demo()

    except ImportError as e:
        print(f"❌ Voice demo module not available: {e}")
    except Exception as e:
        print(f"❌ Error running voice tests: {e}")


def process_single_voice_query():
    """Process a single voice query."""
    print("\n🎤 Single Voice Query")
    print("=" * 40)

    try:
        from voice_input import process_voice_query, display_voice_query_results

        print("🎙️ Preparing to record audio...")
        print("📍 When ready, you'll have 5 seconds to speak your question")

        input("Press Enter when ready to start recording...")

        # Process voice query
        result = process_voice_query(duration=5)

        # Display results
        display_voice_query_results(result)

    except ImportError as e:
        print(f"❌ Voice input module not available: {e}")
    except Exception as e:
        print(f"❌ Error processing voice query: {e}")


def run_mixed_language_demo():
    """Demonstrate mixed language capabilities."""
    print("\n🌐 Mixed Language Capabilities Demo")
    print("=" * 50)

    try:
        from ollama_llm import run_rag_query
        from query_database import load_database

        # Load database
        db = load_database()
        if not db:
            print("❌ Database not available")
            return

        # Demo queries in different languages
        demo_queries = [
            {
                "text": "What is an algorithm?",
                "language": "English",
                "description": "Basic algorithm definition",
            },
            {
                "text": "How does quicksort work?",
                "language": "English",
                "description": "Algorithm explanation",
            },
            {
                "text": "অ্যালগরিদম কি?",
                "language": "Bangla",
                "description": "Algorithm definition in Bangla",
            },
        ]

        print("🔍 Testing queries in different languages:")

        for i, query_info in enumerate(demo_queries, 1):
            print(f"\n🎯 Query {i} ({query_info['language']}):")
            print(f"📝 Text: {query_info['text']}")
            print(f"📋 Description: {query_info['description']}")

            result = run_rag_query(query_info["text"], db=db)

            if result["success"]:
                print(f"✅ Success!")
                print(f"💬 Answer: {result['answer'][:150]}...")
                if result.get("sources"):
                    print(f"📚 Sources: {len(result['sources'])} pages")
            else:
                print(f"❌ Failed: {result['answer']}")

            print("-" * 30)

    except ImportError as e:
        print(f"❌ Required modules not available: {e}")
    except Exception as e:
        print(f"❌ Error in mixed language demo: {e}")


def main_menu():
    """Display main menu and handle user selection."""
    while True:
        print_banner()

        print("\n📋 MAIN MENU")
        print("1. 📊 Show System Status")
        print("2. 🔄 Process Documents (Create/Update Database)")
        print("3. 🔍 Test Database Queries")
        print("4. 💬 Interactive Chat Session")
        print("5. 🎤 Voice Input Session")
        print("6. 🎙️ Single Voice Query")
        print("7. 🌐 Mixed Language Demo")
        print("8. 🧪 Run System Tests")
        print("9. 🔬 Voice Input Tests")
        print("10. 🛠️ Check Dependencies")
        print("11. 🚪 Exit")

        choice = input("\nSelect option (1-11): ").strip()

        if choice == "1":
            show_system_status()
        elif choice == "2":
            run_document_processing()
        elif choice == "3":
            run_database_query_test()
        elif choice == "4":
            run_interactive_chat()
        elif choice == "5":
            run_voice_input_session()
        elif choice == "6":
            process_single_voice_query()
        elif choice == "7":
            run_mixed_language_demo()
        elif choice == "8":
            run_system_tests()
        elif choice == "9":
            run_voice_input_test()
        elif choice == "10":
            print("\n🛠️ Checking Dependencies...")
            issues = check_dependencies()
            if issues:
                print("❌ Issues found:")
                for issue in issues:
                    print(f"   - {issue}")
            else:
                print("✅ All dependencies working correctly!")
        elif choice == "11":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please select 1-11.")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    print(f"Starting RAG System at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check if running from correct directory
    if not os.path.exists("loader.py"):
        print("❌ Error: Please run this script from the RAG system directory")
        print("   (The directory containing loader.py and other RAG files)")
        sys.exit(1)

    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n👋 RAG System terminated by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
