from split import chunks
import os


def assign_unique_ids(chunks):
    """
    Assign deterministic unique IDs to each chunk based on source path, page number, and chunk index.

    Args:
        chunks: List of document chunks from text splitter

    Returns:
        List of chunks with unique IDs assigned to metadata
    """
    print("Assigning unique IDs to chunks...")

    # Keep track of chunk index per page
    page_chunk_counters = {}

    for doc in chunks:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", 0)

        # Clean source path for ID (remove special characters)
        source_clean = (
            os.path.basename(source)
            .replace(".pdf", "")
            .replace(" ", "_")
            .replace("-", "_")
        )

        # Create page key for tracking chunks per page
        page_key = f"{source_clean}_page{page}"

        # Initialize or increment chunk counter for this page
        if page_key not in page_chunk_counters:
            page_chunk_counters[page_key] = 0
        else:
            page_chunk_counters[page_key] += 1

        chunk_index = page_chunk_counters[page_key]

        # Generate unique ID
        chunk_id = f"{source_clean}_page{page}_chunk{chunk_index}"
        doc.metadata["id"] = chunk_id

    print(f"Assigned unique IDs to {len(chunks)} chunks")

    # Show some examples
    print("\nExample chunk IDs:")
    for i, chunk in enumerate(chunks[:5]):
        print(f"  Chunk {i+1}: {chunk.metadata['id']}")

    return chunks


def verify_unique_ids(chunks):
    """Verify that all chunk IDs are unique."""
    ids = [chunk.metadata.get("id") for chunk in chunks]
    unique_ids = set(ids)

    print(f"\nID Verification:")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Unique IDs: {len(unique_ids)}")
    print(f"  All IDs unique: {len(ids) == len(unique_ids)}")

    if len(ids) != len(unique_ids):
        print("  WARNING: Duplicate IDs found!")
        # Find duplicates
        seen = set()
        duplicates = set()
        for id in ids:
            if id in seen:
                duplicates.add(id)
            seen.add(id)
        print(f"  Duplicate IDs: {duplicates}")
        return False

    return True


if __name__ == "__main__":
    print("Starting chunk ID assignment process...")

    # Assign IDs to chunks
    chunks_with_ids = assign_unique_ids(chunks)

    # Verify uniqueness
    is_unique = verify_unique_ids(chunks_with_ids)

    if is_unique:
        print("\n✅ All chunk IDs are unique and ready for database storage!")
    else:
        print("\n❌ Issues found with chunk IDs!")

    # Export for next step
    print(f"\nChunks with IDs are ready for vector database creation.")
    print(
        f"Variable 'chunks_with_ids' contains {len(chunks_with_ids)} processed chunks."
    )
