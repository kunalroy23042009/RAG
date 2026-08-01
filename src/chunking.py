"""
Context-aware chunking of a structured notice into embeddable Documents.

Converted from notebook cell 7. One change from the original: the function
now accepts an optional `category` (the ML classifier's prediction) and
injects it into the chunk metadata, so retrieval can later be filtered by
category. If you don't pass a category, behaviour is identical to before.
"""
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.extraction import AcademicNoticeSchema
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict

def chunk_document_with_metadata(document_pages: List[Dict], chunk_size: int = 800, chunk_overlap: int = 150) -> List[Dict]:
    """
    Splits page text into semantic chunks while inheriting and expanding 
    upon the original page metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunked_data = []
    
    for page in document_pages:
        chunks = splitter.split_text(page["text"])
        
        for i, chunk_text in enumerate(chunks):
            # Duplicate the parent metadata and append chunk-specific identifiers
            meta = page["metadata"].copy()
            meta["chunk_index"] = i
            
            chunked_data.append({
                "text": chunk_text,
                "metadata": meta
            })
            
    return chunked_data

def create_intelligent_context_chunks(
    structured_notice: AcademicNoticeSchema,
    category: Optional[str] = None,
) -> List[Document]:
    """
    Synthesizes context-aware document chunks from a structured Pydantic object.
    Prepends global identifiers to every chunk to avoid orphaned context during
    vector retrieval.
    """
    global_metadata = {
        "issuing_authority": structured_notice.issuing_authority,
        "reference_number": structured_notice.reference_number or "UNKNOWN_REF",
        "date_issued": structured_notice.date_issued,
        "subject": structured_notice.subject_line or "General Administrative Notice",
        "category": category or "Uncategorized",  # NEW
    }

    context_header = (
        f"Notice Reference: {global_metadata['reference_number']}, "
        f"Date: {global_metadata['date_issued']}, "
        f"Subject: {global_metadata['subject']}.\nContent: "
    )

    body_text = (structured_notice.main_body_content or "").strip()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=250,
        separators=["\n\n", "\n", "।", ".", " ", ""],
    )

    raw_text_chunks = text_splitter.split_text(body_text)

    processed_documents = []
    for chunk in raw_text_chunks:
        enriched_text = context_header + chunk
        doc = Document(page_content=enriched_text, metadata=global_metadata)
        processed_documents.append(doc)

    return processed_documents
