#!/usr/bin/env python3
"""
Main RAG System Runner
Provides a unified interface to the complete RAG (Retrieval-Augmented Generation) system.
"""

import sys
import os
from datetime import datetime


def print_banner():
    """Print system banner."""
    print("=" * 60)
    print("🤖 CHAT PDF RAG SYSTEM")
    print("📚 Retrieval-Augmented Generation for PDF Documents")
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


def main_menu():
    """Display main menu and handle user selection."""
    while True:
        print_banner()

        print("\n📋 MAIN MENU")
        print("1. 📊 Show System Status")
        print("2. 🔄 Process Documents (Create/Update Database)")
        print("3. 🔍 Test Database Queries")
        print("4. 💬 Interactive Chat Session")
        print("5. 🧪 Run System Tests")
        print("6. 🛠️ Check Dependencies")
        print("7. 🚪 Exit")

        choice = input("\nSelect option (1-7): ").strip()

        if choice == "1":
            show_system_status()
        elif choice == "2":
            run_document_processing()
        elif choice == "3":
            run_database_query_test()
        elif choice == "4":
            run_interactive_chat()
        elif choice == "5":
            run_system_tests()
        elif choice == "6":
            print("\n🛠️ Checking Dependencies...")
            issues = check_dependencies()
            if issues:
                print("❌ Issues found:")
                for issue in issues:
                    print(f"   - {issue}")
            else:
                print("✅ All dependencies working correctly!")
        elif choice == "7":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please select 1-7.")

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
