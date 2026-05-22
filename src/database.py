import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Use langchain-chroma instead of deprecated langchain_community Chroma
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

# Define persistent storage directories relative to the project root (this file's parent dir)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(_PROJECT_ROOT, "chroma_db")
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")

_embeddings_instance = None

def get_embeddings_model() -> HuggingFaceEmbeddings:
    """Initializes and returns a free, local open-source Hugging Face embedding model."""
    global _embeddings_instance
    if _embeddings_instance is None:
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"local_files_only": True}
        )
    return _embeddings_instance

def ingest_pdf_directory(data_path: str = DATA_DIR) -> int:
    """
    Scans the data directory, parses all PDFs, chunks them semantically,
    and indexes them into a persistent local ChromaDB instance using Hugging Face embeddings.
    """
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print(f" Created empty data folder at: {data_path}. Drop your PDFs there!")
        return 0

    # 1. Gather and parse all PDF documents
    raw_documents: List[Document] = []
    for file in os.listdir(data_path):
        if file.lower().endswith(".pdf"):
            file_path = os.path.join(data_path, file)
            print(f" Parsing document: {file}")
            try:
                loader = PyPDFLoader(file_path)
                raw_documents.extend(loader.load())
            except Exception as e:
                print(f"❌ Error parsing {file}: {e}")

    if not raw_documents:
        print("⚠️ No valid PDF content extracted. Add PDFs to the data folder.")
        return 0

    print(f" Successfully extracted {len(raw_documents)} raw pages.")

    # 2. Semantic Chunking Implementation
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1600,
        chunk_overlap=300,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
        is_separator_regex=False
    )
    
    chunks = text_splitter.split_documents(raw_documents)
    
    # Inject a metadata tracking signature to preserve structural document context
    for i, chunk in enumerate(chunks):
        source_path = chunk.metadata.get('source', 'Unknown')
        source = os.path.basename(source_path)
        chunk.metadata['source'] = source
        page = chunk.metadata.get('page', 'Unknown')
        chunk.metadata['tracking_signature'] = f"Source: {source} | Page: {page} | Chunk ID: {i}"
        
    print(f" Split raw pages into {len(chunks)} optimized semantic chunks.")

    # 3. Persistent Local Ingestion
    print(f" Saving vectors to disk at: {DB_DIR}...")
    embeddings = get_embeddings_model()
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    
    print(" Vector database ingestion completed successfully!")
    return len(chunks)

def get_local_retriever(k_neighbors: int = 5):
    """Connects to the existing local database on disk using Hugging Face embeddings."""
    if not os.path.exists(DB_DIR):
        raise FileNotFoundError(
            f"Database directory '{DB_DIR}' not found. Run `ingest_pdf_directory()` first."
        )
        
    embeddings = get_embeddings_model()
    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": k_neighbors})

if __name__ == "__main__":
    print("--- STARTING VECTOR DB INGESTION RUNTIME ---")
    ingest_pdf_directory()
