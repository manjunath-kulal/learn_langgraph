import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# -----------------------------
# PATH SETUP (robust, no hardcoding)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "resumes")
INDEX_PATH = os.path.join(BASE_DIR, "resume_index")

# -----------------------------
# EMBEDDINGS (FREE, LOCAL)
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def ingest_resumes():
    if not os.path.exists(DATA_FOLDER):
        raise FileNotFoundError(f"'resumes' folder not found at {DATA_FOLDER}")

    print(f"Loading PDFs from: {DATA_FOLDER}")

    loader = PyPDFDirectoryLoader(DATA_FOLDER)
    raw_docs = loader.load()

    if not raw_docs:
        raise ValueError("No PDFs found in resumes folder")

    print(f"Loaded {len(raw_docs)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    documents = splitter.split_documents(raw_docs)

    print(f"Created {len(documents)} chunks")

    print("Building FAISS index (local, free)...")
    vector_store = FAISS.from_documents(documents, embeddings)

    vector_store.save_local(INDEX_PATH)
    print(f"Index saved at: {INDEX_PATH}")

if __name__ == "__main__":
    ingest_resumes()