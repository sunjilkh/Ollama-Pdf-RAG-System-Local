from langchain_text_splitters import RecursiveCharacterTextSplitter
from langdetect import detect
from loader import documents
import warnings

warnings.filterwarnings("ignore")

# Try to import Indic NLP for Bangla text processing
try:
    from indicnlp.tokenize import sentence_tokenize

    INDIC_NLP_AVAILABLE = True
    print("✅ Indic NLP library available for Bangla text processing")
except ImportError:
    INDIC_NLP_AVAILABLE = False
    print(
        "⚠️  Indic NLP library not available - falling back to English splitter for Bangla"
    )


def detect_document_language(documents):
    """
    Detect the primary language of the document collection.

    Args:
        documents: List of document pages

    Returns:
        str: Primary language code ('en', 'bn', etc.)
    """
    if not documents:
        return "en"

    # Sample text from multiple pages for better detection
    sample_text = ""
    pages_to_sample = min(3, len(documents))  # Sample first 3 pages

    for i in range(pages_to_sample):
        page_content = documents[i].page_content
        # Take first 500 characters from each page
        sample_text += page_content[:500] + " "

    # Clean and detect language
    try:
        cleaned_text = " ".join(sample_text.split())
        if len(cleaned_text) < 50:
            return "en"  # Default to English for very short texts

        language = detect(cleaned_text)
        print(f"🔍 Detected document language: {language}")
        return language

    except Exception as e:
        print(f"Language detection failed: {e}")
        return "en"  # Default to English


def split_bangla_text(text, chunk_size=1000, chunk_overlap=100):
    """
    Split Bangla text using Indic NLP sentence tokenizer.

    Args:
        text (str): Bangla text to split
        chunk_size (int): Target chunk size in characters
        chunk_overlap (int): Overlap between chunks

    Returns:
        list: List of text chunks
    """
    if not INDIC_NLP_AVAILABLE:
        print("⚠️  Indic NLP not available, using English splitter")
        return split_english_text(text, chunk_size, chunk_overlap)

    try:
        # Split into sentences using Indic NLP
        sentences = sentence_tokenize.sentence_split(text, lang="bn")

        # Combine sentences into chunks
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                # Save current chunk
                chunks.append(current_chunk.strip())

                # Start new chunk with overlap
                if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                    # Take last part of current chunk for overlap
                    overlap_text = current_chunk[-chunk_overlap:]
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    except Exception as e:
        print(f"Error in Bangla text splitting: {e}")
        print("Falling back to English text splitter")
        return split_english_text(text, chunk_size, chunk_overlap)


def split_english_text(text, chunk_size=1000, chunk_overlap=100):
    """
    Split English text using RecursiveCharacterTextSplitter.

    Args:
        text (str): English text to split
        chunk_size (int): Target chunk size in characters
        chunk_overlap (int): Overlap between chunks

    Returns:
        list: List of text chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)


def create_language_aware_chunks(documents, chunk_size=1000, chunk_overlap=100):
    """
    Create chunks from documents using language-appropriate splitting.

    Args:
        documents: List of document pages
        chunk_size (int): Target chunk size in characters
        chunk_overlap (int): Overlap between chunks

    Returns:
        list: List of document chunks with metadata
    """
    print("🔄 Starting language-aware document splitting...")
    print(f"Total documents to split: {len(documents)}")

    # Detect primary language
    primary_language = detect_document_language(documents)

    # Choose splitting method based on language
    if primary_language == "bn":
        print("📝 Using Bangla-aware splitting with Indic NLP")
        split_function = split_bangla_text
    else:
        print("📝 Using English splitting with RecursiveCharacterTextSplitter")
        split_function = split_english_text

    # Process all documents
    all_chunks = []
    total_characters = 0

    for doc in documents:
        # Split the document content
        text_chunks = split_function(doc.page_content, chunk_size, chunk_overlap)

        # Create document chunks with metadata
        for chunk_text in text_chunks:
            # Create new document chunk
            from langchain_core.documents import Document

            chunk = Document(
                page_content=chunk_text,
                metadata=doc.metadata.copy(),  # Preserve original metadata
            )
            all_chunks.append(chunk)
            total_characters += len(chunk_text)

    print(f"✅ Documents split into {len(all_chunks)} chunks")
    print(
        f"📊 Average characters per chunk: {total_characters // len(all_chunks) if all_chunks else 0}"
    )
    print(f"🌐 Primary language: {primary_language}")

    return all_chunks


# Create chunks using language-aware splitting
chunks = create_language_aware_chunks(documents)

# Display preview
if chunks:
    print(f"\n📋 First chunk preview (first 200 characters):")
    print(chunks[0].page_content[:200] + "...")
    print(f"📄 First chunk metadata: {chunks[0].metadata}")

    # Show language distribution
    print(f"\n📊 Chunk Statistics:")
    print(f"   Total chunks: {len(chunks)}")
    print(
        f"   Shortest chunk: {min(len(chunk.page_content) for chunk in chunks)} characters"
    )
    print(
        f"   Longest chunk: {max(len(chunk.page_content) for chunk in chunks)} characters"
    )
    print(
        f"   Average chunk: {sum(len(chunk.page_content) for chunk in chunks) // len(chunks)} characters"
    )
else:
    print("⚠️  No chunks created - check if documents are loaded properly")


# Test function for Bangla chunking
def test_bangla_chunking():
    """Test Bangla text chunking with sample text."""
    print("\n🧪 Testing Bangla Chunking...")

    sample_bangla_text = """
    এটি একটি বাংলা অনুচ্ছেদ। এখানে কয়েকটি বাক্য রয়েছে। প্রতিটি বাক্য আলাদা আলাদা ভাবে প্রক্রিয়া করা হবে।
    কম্পিউটার বিজ্ঞান একটি আকর্ষণীয় বিষয়। এটি গণিত এবং প্রযুক্তির সংমিশ্রণ। আমরা এখানে বিভিন্ন অ্যালগরিদম নিয়ে আলোচনা করব।
    মেশিন লার্নিং কৃত্রিম বুদ্ধিমত্তার একটি শাখা। এটি ডেটা থেকে শেখার প্রক্রিয়া। আমরা বিভিন্ন মডেল ব্যবহার করে সমস্যা সমাধান করি।
    """

    print(f"Sample text: {sample_bangla_text[:100]}...")

    # Test language detection
    try:
        lang = detect(sample_bangla_text)
        print(f"Detected language: {lang}")
    except Exception as e:
        print(f"Language detection failed: {e}")

    # Test Bangla chunking
    chunks = split_bangla_text(sample_bangla_text, chunk_size=200, chunk_overlap=50)

    print(f"\n📊 Bangla Chunking Results:")
    print(f"   Number of chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"\n   Chunk {i+1} ({len(chunk)} chars):")
        print(f"   {chunk}")

    if len(chunks) > 3:
        print(f"\n   ... and {len(chunks) - 3} more chunks")


if __name__ == "__main__":
    print("🚀 Running Bangla-aware text splitting test...")
    test_bangla_chunking()
