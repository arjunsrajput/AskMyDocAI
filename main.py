import os
import shutil
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import UPLOAD_DIRECTORY
from src.document_loader import MultiFormatDocumentLoader
from src.text_splitter import DocumentSplitter
from src.vector_store import VectorStoreManager
from src.rag_engine import RAGEngine

# Ensure upload directory exists
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# Instantiate FastAPI App
app = FastAPI(
    title="📚 Multi-Document RAG REST API",
    description="Production-ready FastAPI backend for document ingestion, vector search, and citation-backed Q&A.",
    version="1.0.0"
)

# CORS Middleware (Allows frontend on different port/domain to communicate)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core Services
vector_manager = VectorStoreManager()
splitter = DocumentSplitter()
rag_engine = RAGEngine(vector_manager)

# --- PYDANTIC SCHEMAS (Data Validation) ---

class QueryRequest(BaseModel):
    query: str = Field(..., example="What are the key findings of the report?")
    top_k: int = Field(default=4, ge=1, le=10, example=4)

class SourceCitation(BaseModel):
    file: str
    page: int
    snippet: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceCitation]

class UploadResponse(BaseModel):
    status: str
    message: str
    files_processed: List[str]
    total_chunks: int

# --- API ENDPOINTS ---

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint to verify backend service status."""
    return {"status": "healthy", "service": "RAG Backend Microservice"}

@app.post("/api/v1/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Accepts PDF, DOCX, TXT files, parses them, splits into semantic chunks,
    and indexes embeddings into ChromaDB.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    processed_file_names = []
    all_chunks = []

    try:
        for file in files:
            file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
            
            # Read file bytes asynchronously
            content = await file.read()
            with open(file_path, "wb") as buffer:
                buffer.write(content)

            # Load & Split into chunks
            docs = MultiFormatDocumentLoader.load_file(file_path)
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)
            processed_file_names.append(file.filename)

        # Ensure we actually extracted chunks
        if not all_chunks:
            raise HTTPException(
                status_code=400, 
                detail="No readable text found in the uploaded document(s). If it's a scanned PDF without text, please try a text-based PDF or TXT file."
            )

        # Store in ChromaDB
        vector_manager.add_documents(all_chunks)

        return UploadResponse(
            status="success",
            message=f"Successfully indexed {len(all_chunks)} chunks from {len(files)} file(s).",
            files_processed=processed_file_names,
            total_chunks=len(all_chunks)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

@app.post("/api/v1/query", response_model=QueryResponse, tags=["RAG"])
async def query_documents(request: QueryRequest):
    """
    Queries the RAG engine: retrieves relevant context from ChromaDB
    and synthesizes an answer with citations via Google Gemini.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        result = rag_engine.answer_query(request.query, top_k=request.top_k)
        return QueryResponse(
            query=request.query,
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/api/v1/documents", tags=["Documents"])
async def list_indexed_documents():
    """Lists all document filenames currently indexed in the vector store."""
    documents = vector_manager.list_sources()
    return {"total_documents": len(documents), "documents": documents}

@app.delete("/api/v1/reset", tags=["Documents"])
async def reset_database():
    """Wipes the vector database and clears indexed files."""
    try:
        vector_manager.clear_database()
        return {"status": "success", "message": "Vector database has been reset."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting database: {str(e)}")