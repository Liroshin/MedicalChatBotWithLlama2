from dotenv import load_dotenv
import os
from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Load and prepare the data
print("[INFO] Loading PDF files...")
documents = load_pdf("data/")

print("[INFO] Splitting documents into chunks...")
text_chunks = text_split(documents)

print("[INFO] Downloading HuggingFace embeddings...")
embeddings = download_hugging_face_embeddings()  # Or use HuggingFaceEmbeddings()

# Convert to list of text content
texts = [t.page_content for t in text_chunks]

# Create Pinecone Vector Store
index_name = "medical"

print(f"[INFO] Creating or using existing Pinecone index: {index_name}")
docsearch = PineconeVectorStore.from_texts(
    texts=texts,
    embedding=embeddings,
    index_name=index_name
)

print("[SUCCESS] Pinecone Vector Store is ready.")
