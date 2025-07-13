from langdetect import detect
from deep_translator import GoogleTranslator


def detect_language(text):
    """
    Detect the language of the input text.

    Args:
        text (str): Input text to detect language

    Returns:
        str: Language code ('en' for English, 'bn' for Bangla, etc.)
    """
    try:
        # Remove extra whitespace
        text = text.strip()

        if not text:
            return "en"  # Default to English for empty text

        # Detect language
        lang = detect(text)

        # Map common language codes
        if lang == "bn":
            return "bangla"
        elif lang == "en":
            return "english"
        else:
            # For other languages, return the code
            return lang

    except Exception as e:
        print(f"⚠️ Language detection failed: {e}")
        return "english"  # Default to English on error


def translate_to_english(text, source_lang="auto"):
    """
    Translate text to English using Google Translate.

    Args:
        text (str): Text to translate
        source_lang (str): Source language code ('auto' for auto-detection)

    Returns:
        dict: Translation result with original text, translated text, and metadata
    """
    try:
        # Map language codes for deep-translator
        if source_lang == "auto":
            source_lang = "bn"  # Default to Bengali for auto-detection

        # Initialize translator
        translator = GoogleTranslator(source=source_lang, target="en")

        # Perform translation
        translated_text = translator.translate(text)

        return {
            "original_text": text,
            "translated_text": translated_text,
            "source_language": source_lang,
            "target_language": "en",
            "confidence": None,  # deep-translator doesn't provide confidence
            "success": True,
        }

    except Exception as e:
        print(f"⚠️ Translation failed: {e}")
        return {
            "original_text": text,
            "translated_text": text,  # Return original text if translation fails
            "source_language": "unknown",
            "target_language": "en",
            "confidence": None,
            "success": False,
            "error": str(e),
        }


def process_query_with_translation(query):
    """
    Process a query with automatic language detection and translation.

    Args:
        query (str): User query in any supported language

    Returns:
        dict: Processing result with original query, processed query, and metadata
    """
    try:
        # Clean the query
        query = query.strip()

        if not query:
            return {
                "original_query": query,
                "processed_query": query,
                "language_detected": "unknown",
                "translation_needed": False,
                "translation_result": None,
                "success": False,
                "error": "Empty query",
            }

        # Detect language
        detected_lang = detect_language(query)

        # Check if translation is needed
        if detected_lang == "bangla":
            print(f"🔄 Bangla query detected: '{query}'")
            print("🌐 Translating to English...")

            # Translate to English
            translation_result = translate_to_english(query, source_lang="bn")

            if translation_result["success"]:
                processed_query = translation_result["translated_text"]
                print(f"✅ Translation successful: '{processed_query}'")
            else:
                print(
                    f"❌ Translation failed: {translation_result.get('error', 'Unknown error')}"
                )
                processed_query = query  # Use original query if translation fails

            return {
                "original_query": query,
                "processed_query": processed_query,
                "language_detected": detected_lang,
                "translation_needed": True,
                "translation_result": translation_result,
                "success": True,
            }

        else:
            # No translation needed for English or other languages
            print(f"🔤 {detected_lang.title()} query detected: '{query}'")

            return {
                "original_query": query,
                "processed_query": query,
                "language_detected": detected_lang,
                "translation_needed": False,
                "translation_result": None,
                "success": True,
            }

    except Exception as e:
        print(f"⚠️ Query processing failed: {e}")
        return {
            "original_query": query,
            "processed_query": query,
            "language_detected": "unknown",
            "translation_needed": False,
            "translation_result": None,
            "success": False,
            "error": str(e),
        }


def test_translation_service():
    """
    Test the translation service with sample queries.
    """
    print("🧪 Testing Translation Service")
    print("=" * 40)

    test_queries = [
        "What is an algorithm?",  # English
        "এলগরিদম কি?",  # Bangla: "What is algorithm?"
        "অ্যালগরিদম কিভাবে কাজ করে?",  # Bangla: "How does algorithm work?"
        "How does quicksort work?",  # English
        "ডেটা স্ট্রাকচার কি?",  # Bangla: "What is data structure?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: {query}")

        result = process_query_with_translation(query)

        if result["success"]:
            print(f"✅ Language: {result['language_detected']}")
            print(f"🔄 Translation needed: {result['translation_needed']}")
            print(f"📝 Processed query: '{result['processed_query']}'")

            if result["translation_result"]:
                print(
                    f"🌐 Translation confidence: {result['translation_result'].get('confidence', 'N/A')}"
                )
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")

        print("-" * 30)


if __name__ == "__main__":
    test_translation_service()
