"""
Persistent store for every notice's structured metadata.

Why this file exists: in the original notebook, the structured data Gemini
extracted only ever lived inside the chunking step in memory — it was never
saved on its own. Both the classifier (needs subject/body text to train on)
and the dashboard (needs date/authority/category to chart) need that data
to persist across runs. This module is the one place both read from and
write to, stored as JSON Lines (one notice per line) for easy appending.
"""
import json
import os
from datetime import datetime, timezone
from typing import List, Dict, Optional

from config import NOTICES_STORE_PATH
from src.extraction import AcademicNoticeSchema
import chromadb
from typing import List, Dict

class NoticeStore:
    def __init__(self, db_path: str = "data/chroma_db"):
        # Initialize persistent local storage
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Compatible with standard embedding pipelines (e.g., BGE-M3)
        self.collection = self.client.get_or_create_collection(
            name="multipage_notices",
            metadata={"hnsw:space": "cosine"}
        )

    def upsert_chunks(self, chunks: List[Dict]):
        """
        Pushes document chunks and their associated metadata into ChromaDB.
        """
        if not chunks:
            return
            
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        # Generate deterministic IDs: e.g., "exam_schedule.pdf_p2_c0"
        ids = [
            f"{meta['source_document']}_p{meta['page_number']}_c{meta['chunk_index']}" 
            for meta in metadatas
        ]
        
        self.collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Successfully upserted {len(documents)} chunks to the vector store.")

def save_notice_metadata(
    filename: str,
    structured_notice: AcademicNoticeSchema,
    category: Optional[str] = None,
) -> None:
    """Appends one processed notice's metadata as a JSON line."""
    record = {
        "filename": filename,
        "issuing_authority": structured_notice.issuing_authority,
        "reference_number": structured_notice.reference_number,
        "date_issued": structured_notice.date_issued,
        "subject_line": structured_notice.subject_line,
        "main_body_content": structured_notice.main_body_content,
        "document_type": structured_notice.document_type,
        "category": category or "Uncategorized",
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    os.makedirs(os.path.dirname(NOTICES_STORE_PATH), exist_ok=True)
    with open(NOTICES_STORE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_all_notices() -> List[Dict]:
    """Loads every stored notice as a list of dicts. Empty list if none yet."""
    if not os.path.exists(NOTICES_STORE_PATH):
        return []
    records = []
    with open(NOTICES_STORE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
