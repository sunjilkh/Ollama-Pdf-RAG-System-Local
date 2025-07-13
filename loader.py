from langchain_community.document_loaders import PyPDFLoader
import os

data_folder = "."  # Current directory
documents = []

print("Looking for PDF files in the current directory...")

for filename in os.listdir(data_folder):
    if filename.endswith(".pdf"):
        print(f"Found PDF: {filename}")
        print(f"Loading {filename}...")
        loader = PyPDFLoader(os.path.join(data_folder, filename))
        pdf_documents = loader.load()

        # Enhance metadata for better citation support
        for i, doc in enumerate(pdf_documents):
            # Add or update page metadata for citation
            doc.metadata["file_name"] = filename
            doc.metadata["page_number"] = (
                i + 1
            )  # Human-readable page numbers start from 1
            doc.metadata["source_file"] = filename

            # Keep original page metadata if it exists
            if "page" not in doc.metadata:
                doc.metadata["page"] = i

        documents.extend(pdf_documents)
        print(f"Loaded {len(pdf_documents)} pages from {filename}")

print(f"\nTotal documents loaded: {len(documents)}")
if documents:
    print(f"First document preview (first 200 characters):")
    print(documents[0].page_content[:200] + "...")

    # Display enhanced metadata
    print(f"\n📄 Enhanced metadata for first document:")
    print(f"   File name: {documents[0].metadata.get('file_name', 'Unknown')}")
    print(f"   Page number: {documents[0].metadata.get('page_number', 'Unknown')}")
    print(f"   Source file: {documents[0].metadata.get('source_file', 'Unknown')}")
    print(f"   Total pages: {documents[0].metadata.get('total_pages', 'Unknown')}")


def get_page_citation(metadata):
    """
    Generate a citation string from document metadata.

    Args:
        metadata (dict): Document metadata

    Returns:
        str: Citation string
    """
    file_name = metadata.get("file_name", "Unknown Document")
    page_number = metadata.get("page_number", "Unknown")

    # Clean up file name (remove extension)
    clean_name = (
        file_name.replace(".pdf", "") if file_name.endswith(".pdf") else file_name
    )

    return f"{clean_name}, Page {page_number}"


# Test citation generation
if documents:
    print(f"\n📚 Sample citation: {get_page_citation(documents[0].metadata)}")
