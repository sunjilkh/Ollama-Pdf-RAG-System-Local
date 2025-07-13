import whisper
import pyaudio
import wave
import tempfile
import os
import argparse
import sys
from pathlib import Path
from query_database import load_database
from ollama_llm import run_rag_query
import warnings

warnings.filterwarnings("ignore")

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5  # Default recording time

# Global whisper model
_whisper_model = None


def load_whisper_model(model_size="base"):
    """
    Load Whisper model for transcription.

    Args:
        model_size (str): Model size - "tiny", "base", "small", "medium", "large"

    Returns:
        whisper.Whisper: Loaded model
    """
    global _whisper_model

    if _whisper_model is None:
        print(f"🔄 Loading Whisper '{model_size}' model...")
        try:
            _whisper_model = whisper.load_model(model_size)
            print(f"✅ Whisper model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading Whisper model: {e}")
            return None

    return _whisper_model


def record_audio(duration=RECORD_SECONDS, output_file="temp_audio.wav"):
    """
    Record audio from microphone.

    Args:
        duration (int): Recording duration in seconds
        output_file (str): Output file path

    Returns:
        str: Path to recorded audio file
    """
    try:
        # Initialize PyAudio
        p = pyaudio.PyAudio()

        print(f"🎤 Recording audio for {duration} seconds...")
        print("📍 Speak now!")

        # Open stream
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        frames = []

        # Record audio
        for i in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK)
            frames.append(data)

        print("🔴 Recording finished!")

        # Stop and close stream
        stream.stop_stream()
        stream.close()
        p.terminate()

        # Save audio file
        with wave.open(output_file, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b"".join(frames))

        print(f"💾 Audio saved to: {output_file}")
        return output_file

    except Exception as e:
        print(f"❌ Error recording audio: {e}")
        return None


def transcribe_audio(audio_file, model_size="base", language=None):
    """
    Transcribe audio file using Whisper.

    Args:
        audio_file (str): Path to audio file
        model_size (str): Whisper model size
        language (str): Language code (e.g., 'en', 'bn') or None for auto-detection

    Returns:
        dict: Transcription results
    """
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return None

    model = load_whisper_model(model_size)
    if model is None:
        return None

    try:
        print(f"🔄 Transcribing audio...")

        # Transcribe with or without language specification
        if language:
            result = model.transcribe(audio_file, language=language)
        else:
            result = model.transcribe(audio_file)

        print(f"✅ Transcription completed!")
        return result

    except Exception as e:
        print(f"❌ Error transcribing audio: {e}")
        return None


def process_voice_query(audio_file=None, duration=5, model_size="base", language=None):
    """
    Complete voice query processing: record -> transcribe -> search -> respond.

    Args:
        audio_file (str): Path to existing audio file (if None, will record new)
        duration (int): Recording duration if recording new audio
        model_size (str): Whisper model size
        language (str): Language code or None for auto-detection

    Returns:
        dict: Complete query results
    """
    # Step 1: Get audio file
    if audio_file is None:
        audio_file = record_audio(duration)
        if audio_file is None:
            return {"error": "Failed to record audio"}

    # Step 2: Transcribe audio
    transcription = transcribe_audio(audio_file, model_size, language)
    if transcription is None:
        return {"error": "Failed to transcribe audio"}

    query_text = transcription["text"].strip()
    detected_language = transcription.get("language", "unknown")

    print(f"\n🎯 Transcribed Query: '{query_text}'")
    print(f"🌐 Detected Language: {detected_language}")

    if not query_text:
        return {"error": "No speech detected"}

    # Step 3: Load database and run RAG query
    print(f"\n🔍 Searching database for relevant information...")
    db = load_database()
    if db is None:
        return {"error": "Database not available"}

    # Step 4: Run RAG query
    rag_result = run_rag_query(query_text, db=db)

    # Step 5: Combine results
    result = {
        "query": query_text,
        "detected_language": detected_language,
        "transcription": transcription,
        "rag_result": rag_result,
        "success": True,
    }

    return result


def display_voice_query_results(result):
    """
    Display voice query results in a formatted way.

    Args:
        result (dict): Voice query results
    """
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return

    print("\n" + "=" * 60)
    print("🎤 VOICE QUERY RESULTS")
    print("=" * 60)

    print(f"📝 Query: {result['query']}")
    print(f"🌐 Language: {result['detected_language']}")

    if result["rag_result"]["success"]:
        print(f"\n💬 Answer:")
        print(result["rag_result"]["answer"])

        if result["rag_result"].get("sources"):
            print(f"\n📚 Sources:")
            for source in result["rag_result"]["sources"]:
                print(f"   - Page {source['page']}")

        print(f"\n🤖 Model: {result['rag_result'].get('model_used', 'Unknown')}")
    else:
        print(f"\n❌ RAG Query Failed: {result['rag_result']['answer']}")


def interactive_voice_session():
    """
    Run an interactive voice query session.
    """
    print("🎤 Interactive Voice Query Session")
    print("=" * 50)
    print("Commands:")
    print("  - Press Enter to start recording")
    print("  - Type 'quit' to exit")
    print("  - Type 'help' for commands")

    while True:
        try:
            user_input = input(
                "\n🎤 Press Enter to record (or 'quit' to exit): "
            ).strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break

            if user_input.lower() == "help":
                print("\nAvailable commands:")
                print("  - Enter: Start recording")
                print("  - quit/exit/q: Exit the session")
                print("  - help: Show this help")
                continue

            # Process voice query
            result = process_voice_query(duration=5)
            display_voice_query_results(result)

        except KeyboardInterrupt:
            print("\n👋 Session ended by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def test_voice_transcription():
    """
    Test voice transcription with sample audio or live recording.
    """
    print("🧪 Testing Voice Transcription")
    print("=" * 40)

    # Test with different languages
    test_languages = [
        {"name": "English", "code": "en"},
        {"name": "Bangla", "code": "bn"},
        {"name": "Auto-detect", "code": None},
    ]

    for lang_info in test_languages:
        print(f"\n🎯 Testing {lang_info['name']} transcription...")

        # Record audio
        audio_file = record_audio(duration=3)
        if audio_file:
            # Transcribe
            result = transcribe_audio(audio_file, language=lang_info["code"])
            if result:
                print(f"✅ Transcription: '{result['text']}'")
                print(f"🌐 Detected Language: {result.get('language', 'unknown')}")
            else:
                print("❌ Transcription failed")

            # Clean up
            if os.path.exists(audio_file):
                os.remove(audio_file)
        else:
            print("❌ Recording failed")

        input("Press Enter to continue to next test...")


def main():
    """
    Main function to handle command line arguments and run voice input.
    """
    parser = argparse.ArgumentParser(description="Voice Input for BanglaRAG System")
    parser.add_argument("--audio", type=str, help="Path to audio file to transcribe")
    parser.add_argument(
        "--duration", type=int, default=5, help="Recording duration in seconds"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size",
    )
    parser.add_argument("--language", type=str, help="Language code (e.g., 'en', 'bn')")
    parser.add_argument("--test", action="store_true", help="Run transcription tests")
    parser.add_argument(
        "--interactive", action="store_true", help="Run interactive voice session"
    )

    args = parser.parse_args()

    if args.test:
        test_voice_transcription()
    elif args.interactive:
        interactive_voice_session()
    elif args.audio:
        # Process existing audio file
        result = process_voice_query(
            audio_file=args.audio, model_size=args.model, language=args.language
        )
        display_voice_query_results(result)
    else:
        # Record and process new audio
        result = process_voice_query(
            duration=args.duration, model_size=args.model, language=args.language
        )
        display_voice_query_results(result)


if __name__ == "__main__":
    main()
