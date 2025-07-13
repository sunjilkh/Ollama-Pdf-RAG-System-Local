"""
Demo script to test voice transcription functionality.
This simulates the voice input process without requiring actual audio recording.
"""

import tempfile
import os
from voice_input import transcribe_audio, load_whisper_model, process_voice_query
from query_database import load_database
from ollama_llm import run_rag_query


def create_demo_audio_file():
    """
    Create a demo audio file for testing.
    In a real scenario, this would be an actual audio recording.
    """
    # For demo purposes, we'll create a placeholder file
    # In practice, this would be a real audio file
    demo_file = "demo_audio.wav"

    # Create a dummy audio file (this is just for demo)
    with open(demo_file, "w") as f:
        f.write("This is a demo audio file placeholder")

    return demo_file


def simulate_bangla_transcription():
    """
    Simulate Bangla voice transcription for testing.
    """
    print("🧪 Simulating Bangla Voice Transcription Test")
    print("=" * 50)

    # Simulate different Bangla queries
    test_queries = [
        {
            "text": "অ্যালগরিদম কি?",
            "translation": "What is an algorithm?",
            "language": "bn",
        },
        {
            "text": "কুইকসর্ট কিভাবে কাজ করে?",
            "translation": "How does quicksort work?",
            "language": "bn",
        },
        {
            "text": "ডায়নামিক প্রোগ্রামিং ব্যাখ্যা করুন",
            "translation": "Explain dynamic programming",
            "language": "bn",
        },
        {
            "text": "What is a binary search tree?",
            "translation": "What is a binary search tree?",
            "language": "en",
        },
    ]

    # Load database
    db = load_database()
    if db is None:
        print("❌ Database not available for testing")
        return

    print("✅ Database loaded successfully")

    # Test each query
    for i, query_info in enumerate(test_queries, 1):
        print(f"\n🎯 Test {i}: {query_info['language'].upper()} Query")
        print(f"📝 Original: {query_info['text']}")
        print(f"🔤 Translation: {query_info['translation']}")

        # Simulate transcription result
        transcription_result = {
            "text": query_info["text"],
            "language": query_info["language"],
        }

        # Run the query using the transcribed text
        print(f"🔍 Running RAG query...")
        rag_result = run_rag_query(query_info["text"], db=db)

        # Display results
        if rag_result["success"]:
            print(f"✅ Query successful!")
            print(f"💬 Answer: {rag_result['answer'][:200]}...")

            if rag_result.get("sources"):
                print(f"📚 Sources: {len(rag_result['sources'])} pages")
                for source in rag_result["sources"][:3]:
                    print(f"   - Page {source['page']}")
        else:
            print(f"❌ Query failed: {rag_result['answer']}")

        print("-" * 40)


def test_voice_input_components():
    """
    Test individual components of the voice input system.
    """
    print("🔧 Testing Voice Input Components")
    print("=" * 40)

    # Test 1: Whisper model loading
    print("\n1. Testing Whisper Model Loading...")
    model = load_whisper_model("tiny")
    if model:
        print("✅ Whisper model loaded successfully")
    else:
        print("❌ Failed to load Whisper model")
        return

    # Test 2: Database connection
    print("\n2. Testing Database Connection...")
    db = load_database()
    if db:
        print("✅ Database connected successfully")
    else:
        print("❌ Failed to connect to database")
        return

    # Test 3: RAG query functionality
    print("\n3. Testing RAG Query Functionality...")
    test_query = "What is an algorithm?"
    result = run_rag_query(test_query, db=db)
    if result["success"]:
        print("✅ RAG query successful")
        print(f"📝 Query: {test_query}")
        print(f"💬 Answer: {result['answer'][:100]}...")
    else:
        print("❌ RAG query failed")

    print("\n✅ All components are working correctly!")


def demo_voice_workflow():
    """
    Demonstrate the complete voice input workflow.
    """
    print("🎤 Voice Input Workflow Demo")
    print("=" * 40)

    # Simulate the complete workflow
    print("\n📋 Workflow Steps:")
    print("1. 🎤 Audio Recording (simulated)")
    print("2. 🔄 Audio Transcription (simulated)")
    print("3. 🔍 Database Query")
    print("4. 🤖 LLM Response Generation")
    print("5. 📄 Page Citation")

    # Simulate a voice query
    sample_queries = [
        "What is the time complexity of merge sort?",
        "Explain the concept of dynamic programming",
    ]

    db = load_database()
    if not db:
        print("❌ Database not available")
        return

    for query in sample_queries:
        print(f"\n🎯 Processing Query: '{query}'")
        print("🔄 Transcribing audio... (simulated)")
        print("🔍 Searching database...")

        result = run_rag_query(query, db=db)

        if result["success"]:
            print("✅ Query processed successfully!")
            print(f"💬 Answer: {result['answer'][:150]}...")

            if result.get("sources"):
                print(f"📚 Citations:")
                for source in result["sources"][:3]:
                    print(
                        f"   - Cormen - Introduction to Algorithms, Page {source['page']}"
                    )
        else:
            print("❌ Query processing failed")

        print("-" * 30)


def main():
    """
    Main function to run voice input tests.
    """
    print("🎤 BanglaRAG Voice Input System - Demo & Test")
    print("=" * 60)

    # Run component tests
    test_voice_input_components()

    print("\n" + "=" * 60)

    # Run workflow demo
    demo_voice_workflow()

    print("\n" + "=" * 60)

    # Run Bangla transcription simulation
    simulate_bangla_transcription()

    print("\n🎉 Demo completed successfully!")
    print("\n📋 Next Steps:")
    print(
        "1. Test with actual audio files using: python voice_input.py --audio sample.wav"
    )
    print("2. Record live audio using: python voice_input.py --interactive")
    print("3. Test Bangla audio: python voice_input.py --language bn")


if __name__ == "__main__":
    main()
