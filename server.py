import os
import sys
import json
import shutil
import asyncio
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.graph import app as graph_app
from src.database import DB_DIR, DATA_DIR, get_embeddings_model

app = FastAPI(title="Agentic RAG API")

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def get_index():
    """Serves the main dashboard page."""
    return FileResponse("index.html")

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Accepts a PDF document upload, splits it into semantic chunks, 
    injects tracking signatures, and appends it to the persistent ChromaDB.
    """
    # 1. Create storage folders if missing
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DB_DIR, exist_ok=True)
    
    file_path = os.path.join(DATA_DIR, file.filename)
    
    # 2. Save the uploaded file to disk
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 3. Parse and chunk the document
        loader = PyPDFLoader(file_path)
        raw_docs = loader.load()
        
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1600,
            chunk_overlap=300,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
            is_separator_regex=False
        )
        chunks = text_splitter.split_documents(raw_docs)
        
        # 4. Inject metadata signatures
        for i, chunk in enumerate(chunks):
            source_path = chunk.metadata.get('source', file.filename)
            source = os.path.basename(source_path)
            chunk.metadata['source'] = source
            page = chunk.metadata.get('page', 'Unknown')
            chunk.metadata['tracking_signature'] = f"Source: {source} | Page: {page} | Chunk ID: {i}"
            
        # 5. Connect and append to Chroma
        try:
            from langchain_chroma import Chroma
        except ImportError:
            from langchain_community.vectorstores import Chroma
            
        embeddings = get_embeddings_model()
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        vectorstore.add_documents(chunks)
        
        return {
            "status": "success",
            "message": f"Successfully ingested {len(chunks)} semantic chunks from {file.filename}."
        }
        
    except Exception as e:
        # Clean up failed file upload to prevent garbage files
        if os.path.exists(file_path):
            os.remove(file_path)
        return {
            "status": "error",
            "message": f"Failed to ingest document: {str(e)}"
        }

@app.get("/api/status")
async def get_status():
    """
    Queries the Chroma vector database to see exactly what files are currently ingested.
    Returns status, total number of chunks, and a list of distinct documents.
    """
    if not os.path.exists(DB_DIR):
        return {
            "status": "success",
            "total_documents": 0,
            "total_chunks": 0,
            "documents": []
        }
    
    try:
        try:
            from langchain_chroma import Chroma
        except ImportError:
            from langchain_community.vectorstores import Chroma
            
        embeddings = get_embeddings_model()
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        
        collection = vectorstore._collection
        res = collection.get(include=["metadatas"])
        metadatas = res.get("metadatas", [])
        
        sources = set()
        for meta in metadatas:
            if meta and "source" in meta:
                basename = os.path.basename(meta["source"])
                sources.add(basename)
                
        doc_list = sorted(list(sources))
        return {
            "status": "success",
            "total_documents": len(doc_list),
            "total_chunks": len(metadatas),
            "documents": doc_list
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to query database status: {str(e)}"
        }

@app.get("/api/chat")
async def chat_stream(
    question: str = Query(..., description="User query to RAG system"),
    source: str = Query(None, description="Optional document source filter")
):
    """
    Executes the LangGraph RAG agent workflow and streams intermediate 
    states and documents node-by-node via Server-Sent Events (SSE).
    """
    async def event_generator():
        inputs = {
            "question": question.strip(),
            "source": source
        }
        
        try:
            async for event in graph_app.astream(inputs):
                for node_name, node_state in event.items():
                    docs_payload = []
                    if "documents" in node_state and node_state["documents"]:
                        for doc in node_state["documents"]:
                            docs_payload.append({
                                "content": doc.page_content,
                                "metadata": doc.metadata
                            })
                    
                    payload = {
                           "node": node_name,
                           "search_fallback": node_state.get("search_fallback"),
                           "documents": docs_payload,
                           "generation": node_state.get("generation"),
                           "confidence": node_state.get("confidence")
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                    
            yield f"data: {json.dumps({'status': 'done'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    # Make sure DB_DIR exists so retriever doesn't crash on import
    from src.database import ingest_pdf_directory
    if not os.path.exists(DB_DIR):
        print(" Database directory missing. Ingesting 'data/' directory first...")
        ingest_pdf_directory()
        
    print(" Starting Agentic RAG Web Server on http://localhost:8000")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
