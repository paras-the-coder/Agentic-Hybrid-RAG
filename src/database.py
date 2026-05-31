import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()

# Define local data directory for uploaded PDFs (relative to project root)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

# If you are downloading this Hugging Face model for the first time, 
# temporarily comment out the three offline lines below using (Ctrl + /)
# Once downloaded, uncomment them to resume offline mode.
        # os.environ["HF_HUB_OFFLINE"] = "1"
        # os.environ["TRANSFORMERS_OFFLINE"] = "1"
        #         model_kwargs={"local_files_only": True} click (Ctrl + /)


def get_pinecone_index_name() -> str:
    """Returns the Pinecone index name from the environment."""
    index_name = os.getenv("PINECONE_INDEX_NAME")
    if not index_name:
        raise EnvironmentError(
            "PINECONE_INDEX_NAME is not set in .env. "
            "Create a free index at https://app.pinecone.io and add it to your .env file."
        )
    return index_name


def get_vectorstore() -> PineconeVectorStore:
    """Returns a connected PineconeVectorStore instance using HuggingFace embeddings."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "PINECONE_API_KEY is not set in .env. "
            "Get your free API key at https://app.pinecone.io and add it to your .env file."
        )
    
    embeddings = get_embeddings_model()
    index_name = get_pinecone_index_name()
    
    return PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=api_key,
    )


def ingest_pdf_directory(data_path: str = DATA_DIR) -> int:
    """
    Scans the data directory, parses all PDFs, chunks them semantically,
    and indexes them into the Pinecone cloud vector database using Hugging Face embeddings.
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
                print(f" Error parsing {file}: {e}")

    if not raw_documents:
        print(" No valid PDF content extracted. Add PDFs to the data folder.")
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
        # Pinecone metadata values must be strings, numbers, booleans, or lists of strings
        chunk.metadata['page'] = int(page) if isinstance(page, (int, float)) else 0
        chunk.metadata['tracking_signature'] = f"Source: {source} | Page: {page} | Chunk ID: {i}"
        
    print(f" Split raw pages into {len(chunks)} optimized semantic chunks.")

    # 3. Cloud Ingestion to Pinecone
    print(f" Uploading vectors to Pinecone index: {get_pinecone_index_name()}...")
    embeddings = get_embeddings_model()
    
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=get_pinecone_index_name(),
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    )
    
    print(" Vector database ingestion to Pinecone completed successfully!")
    return len(chunks)

def get_local_retriever(k_neighbors: int = 5):
    """Connects to the Pinecone cloud index using Hugging Face embeddings."""
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k_neighbors})

if __name__ == "__main__":
    print("--- STARTING VECTOR DB INGESTION RUNTIME ---")
    ingest_pdf_directory()
