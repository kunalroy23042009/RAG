"""
FastAPI-based REST API for the Academic Notice RAG System.

Provides OpenAI/Groq-style endpoints with streaming responses.
No authentication initially (to be added later).
"""
import os
import shutil
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import (
    INPUT_PATH, PROCESSED_PATH, GOOGLE_API_KEY,
    CATEGORIES
)
from src.extraction import extract_structured_metadata
from src.classifier.predict import predict_category
from src.notice_store import save_notice_metadata
from src.chunking import create_intelligent_context_chunks
from src.retrieval import (
    configure_hybrid_retrieval_system,
    generate_augmented_response,
    generate_augmented_response_stream,
)

# Initialize FastAPI app
app = FastAPI(
    title="Academic Notice RAG API",
    description="REST API for querying academic notices with hybrid retrieval",
    version="1.0.0"
)

# Add CORS middleware for open access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (open access)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global embedding model (cached for performance)
_embedding_model = None
_hybrid_retriever = None
_model_loading = False


def get_embedding_model():
    """Get or initialize the embedding model (cached)."""
    global _embedding_model
    if _embedding_model is None:
        from src.embeddings import initialize_multilingual_embedding_model
        _embedding_model = initialize_multilingual_embedding_model()
    return _embedding_model


def get_hybrid_retriever():
    """Get or initialize the hybrid retriever (cached)."""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        embeddings = get_embedding_model()
        _hybrid_retriever = configure_hybrid_retrieval_system([], embeddings)
    return _hybrid_retriever


@app.on_event("startup")
async def startup_event():
    """Pre-load model in background to avoid cold-start latency on first query."""
    import threading
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    def preload():
        try:
            logger.info("Pre-loading embedding model...")
            get_embedding_model()
            logger.info("Pre-loading hybrid retriever...")
            get_hybrid_retriever()
            logger.info("Model pre-load complete")
        except MemoryError:
            logger.error("Out of memory during model pre-load - will retry on first query")
        except Exception as e:
            logger.error(f"Model pre-load failed: {e}")
    
    thread = threading.Thread(target=preload, daemon=True)
    thread.start()


# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    stream: bool = False


class QueryResponse(BaseModel):
    response: str
    sources: List[dict]


class HealthResponse(BaseModel):
    status: str
    model: str


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint - fast, no model loading."""
    return HealthResponse(status="healthy", model="gemini-2.5-flash")


@app.post("/v1/query", response_model=QueryResponse)
async def query_notices(request: QueryRequest):
    """
    Query the RAG system with a question.
    
    Returns a complete response (non-streaming).
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        retriever = get_hybrid_retriever()
        response = generate_augmented_response(request.query, retriever)
        
        # Get sources for context
        retrieved_docs = retriever.invoke(request.query)
        sources = [
            {
                "content": doc.page_content[:200] + "...",
                "metadata": doc.metadata
            }
            for doc in retrieved_docs
        ]
        
        return QueryResponse(response=response, sources=sources)
    except MemoryError:
        raise HTTPException(status_code=503, detail="Service temporarily unavailable - model loading. Please retry in a few seconds.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/v1/query/stream")
async def query_notices_stream(request: QueryRequest):
    """
    Query the RAG system with streaming response.
    
    Yields response word-by-word as it's generated.
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        retriever = get_hybrid_retriever()
        
        async def generate():
            for word in generate_augmented_response_stream(request.query, retriever):
                yield word
        
        return StreamingResponse(generate(), media_type="text/plain")
    except MemoryError:
        raise HTTPException(status_code=503, detail="Service temporarily unavailable - model loading. Please retry in a few seconds.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/v1/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (image or PDF) for processing.
    
    Supports: .jpg, .jpeg, .png, .webp, .pdf
    """
    SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
    SUPPORTED_PDF_EXTENSIONS = (".pdf",)
    
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    file_lower = filename.lower()
    if not (file_lower.endswith(SUPPORTED_IMAGE_EXTENSIONS) or file_lower.endswith(SUPPORTED_PDF_EXTENSIONS)):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: {SUPPORTED_IMAGE_EXTENSIONS + SUPPORTED_PDF_EXTENSIONS}"
        )
    
    # Save file to input directory
    os.makedirs(INPUT_PATH, exist_ok=True)
    file_path = os.path.join(INPUT_PATH, filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the document
        from main import process_image_notice, process_pdf_notice
        
        if file_lower.endswith(SUPPORTED_IMAGE_EXTENSIONS):
            chunks = process_image_notice(file_path, filename)
        elif file_lower.endswith(SUPPORTED_PDF_EXTENSIONS):
            chunks = process_pdf_notice(file_path, filename)
        
        # Update retriever with new documents
        global _hybrid_retriever
        embeddings = get_embedding_model()
        _hybrid_retriever = configure_hybrid_retrieval_system(chunks, embeddings)
        
        # Move to processed directory
        os.makedirs(PROCESSED_PATH, exist_ok=True)
        shutil.move(file_path, os.path.join(PROCESSED_PATH, filename))
        
        return {
            "status": "success",
            "filename": filename,
            "chunks_created": len(chunks),
            "message": "Document processed and added to RAG system"
        }
    except Exception as e:
        # Clean up on error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@app.get("/v1/categories")
async def get_categories():
    """Get available notice categories."""
    return {"categories": CATEGORIES}


if __name__ == "__main__":
    import uvicorn
    import os
    from config import API_HOST
    
    port = int(os.environ.get("PORT", os.environ.get("API_PORT", "8000")))
    
    uvicorn.run(
        "api_server:app",
        host=API_HOST,
        port=port,
        reload=False
    )
