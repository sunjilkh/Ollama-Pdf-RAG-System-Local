from langchain_text_splitters import RecursiveCharacterTextSplitter
from loader import documents

print("Starting document splitting...")
print(f"Total documents to split: {len(documents)}")

# Create text splitter with chunk size of 1000 and overlap of 100
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Split the documents into chunks
chunks = text_splitter.split_documents(documents)

print(f"Documents split into {len(chunks)} chunks")
print(
    f"Average characters per chunk: {sum(len(chunk.page_content) for chunk in chunks) // len(chunks) if chunks else 0}"
)

# Preview first chunk
if chunks:
    print(f"\nFirst chunk preview (first 200 characters):")
    print(chunks[0].page_content[:200] + "...")
    print(f"First chunk metadata: {chunks[0].metadata}")
