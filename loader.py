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
        documents.extend(pdf_documents)
        print(f"Loaded {len(pdf_documents)} pages from {filename}")

print(f"\nTotal documents loaded: {len(documents)}")
if documents:
    print(f"First document preview (first 200 characters):")
    print(documents[0].page_content[:] + "...")


