"""
Main entry point for the Academic Notice RAG System.

Processes new notice images and multi-page PDFs in `data/input_docs`,
runs them through:
Extraction -> Classification (Module 1) -> Intelligent Chunking -> Vector Embedding -> Hybrid Retrieval Storage,
then starts the augmented Q&A query loop.

Usage:
    python main.py
"""

import os
import shutil
import fitz  # PyMuPDF for handling multi-page PDFs
from typing import List

from config import INPUT_PATH, PROCESSED_PATH
from src.extraction import extract_structured_metadata
from src.classifier.predict import predict_category
from src.notice_store import save_notice_metadata
from src.chunking import create_intelligent_context_chunks
from src.embeddings import initialize_multilingual_embedding_model
from src.retrieval import (
    load_processed_log,
    save_processed_log,
    configure_hybrid_retrieval_system,
    generate_augmented_response,
)

SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
SUPPORTED_PDF_EXTENSIONS = (".pdf",)


def process_image_notice(file_path: str, filename: str) -> List:
    """Processes a single image-based notice file."""
    structured_notice = extract_structured_metadata(file_path)

    # Predict notice category using Module 1 classifier
    subject = getattr(structured_notice, "subject_line", "") or ""
    body = getattr(structured_notice, "main_body_content", "") or ""
    category = predict_category(subject, body)

    # Persist metadata and generate contextual chunks
    save_notice_metadata(filename, structured_notice, category=category)
    chunks = create_intelligent_context_chunks(structured_notice, category=category)

    # Tag page-level metadata on chunks
    for chunk in chunks:
        if isinstance(chunk, dict):
            chunk.setdefault("metadata", {})
            chunk["metadata"]["source_file"] = filename
            chunk["metadata"]["page_number"] = 1
            chunk["metadata"]["total_pages"] = 1
        elif hasattr(chunk, "metadata"):
            chunk.metadata["source_file"] = filename
            chunk.metadata["page_number"] = 1
            chunk.metadata["total_pages"] = 1

    return chunks


def process_pdf_notice(pdf_path: str, filename: str) -> List:
    """
    Processes multi-page PDFs page by page, supporting both scanned
    documents and text-based PDFs while preserving page-level metadata.
    """
    pdf_chunks = []
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    print(f"  --> Processing PDF '{filename}' with {total_pages} page(s)...")

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        page = doc[page_idx]
        temp_img_path = f"_temp_{filename}_p{page_num}.png"

        try:
            # Render page to high-res image to guarantee accurate extraction for scanned PDFs
            pix = page.get_pixmap(dpi=150)
            pix.save(temp_img_path)

            # Extract structured notice schema using the vision/OCR extractor
            structured_notice = extract_structured_metadata(temp_img_path)

            # Retrieve text or fallback to page text string
            subject = getattr(structured_notice, "subject_line", "") or ""
            body = getattr(structured_notice, "main_body_content", "") or page.get_text("text")

            # Predict category
            category = predict_category(subject, body)

            # Save notice metadata using a unique page key
            page_identifier = f"{filename}_page_{page_num}"
            save_notice_metadata(page_identifier, structured_notice, category=category)

            # Generate context chunks
            chunks = create_intelligent_context_chunks(structured_notice, category=category)

            # Inject precise multi-page metadata into chunk models
            for chunk in chunks:
                if isinstance(chunk, dict):
                    chunk.setdefault("metadata", {})
                    chunk["metadata"]["source_file"] = filename
                    chunk["metadata"]["page_number"] = page_num
                    chunk["metadata"]["total_pages"] = total_pages
                elif hasattr(chunk, "metadata"):
                    chunk.metadata["source_file"] = filename
                    chunk.metadata["page_number"] = page_num
                    chunk.metadata["total_pages"] = total_pages

            pdf_chunks.extend(chunks)
            print(f"      [✓] Processed Page {page_num}/{total_pages} ({len(chunks)} chunks created)")

        except Exception as err:
            print(f"      [!] Error processing page {page_num} of '{filename}': {err}")

        finally:
            if os.path.exists(temp_img_path):
                try:
                    os.remove(temp_img_path)
                except OSError:
                    pass

    doc.close()
    return pdf_chunks


def process_new_notices() -> List:
    """Scans INPUT_PATH for unprocessed notices/PDFs and pipelines them to vector index."""
    documents_to_add = []

    os.makedirs(INPUT_PATH, exist_ok=True)
    os.makedirs(PROCESSED_PATH, exist_ok=True)

    all_files = os.listdir(INPUT_PATH)
    processed_files = load_processed_log()
    
    # Filter for new files with supported formats
    new_files = [
        f for f in all_files 
        if f not in processed_files and f.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS + SUPPORTED_PDF_EXTENSIONS)
    ]

    print("New files detected:", new_files)

    for file in new_files:
        file_path = os.path.join(INPUT_PATH, file)
        file_lower = file.lower()

        try:
            if file_lower.endswith(SUPPORTED_IMAGE_EXTENSIONS):
                print(f"Processing Image Notice: {file}")
                chunks = process_image_notice(file_path, file)
                documents_to_add.extend(chunks)

            elif file_lower.endswith(SUPPORTED_PDF_EXTENSIONS):
                chunks = process_pdf_notice(file_path, file)
                documents_to_add.extend(chunks)

        except Exception as e:
            print(f"[!] Critical error processing file '{file}': {e}")
            continue

    # Update processed log
    processed_files.extend(new_files)
    save_processed_log(processed_files)

    # Move processed input files into data/processed_docs
    for file in new_files:
        src_path = os.path.join(INPUT_PATH, file)
        if os.path.exists(src_path):
            shutil.move(src_path, os.path.join(PROCESSED_PATH, file))

    return documents_to_add


def main():
    print("--- Starting Notice RAG Ingestion Pipeline ---")
    documents_to_add = process_new_notices()

    print("\nInitializing Embedding Engine & Vector Store...")
    embedding_infrastructure = initialize_multilingual_embedding_model()
    hybrid_retriever = configure_hybrid_retrieval_system(documents_to_add, embedding_infrastructure)

    print("\n========================================================")
    print("RAG System Ready. Type your question (or 'exit'/'quit' to exit).")
    print("========================================================\n")

    while True:
        try:
            query = input("Query: ").strip()
            if query.lower() in ("exit", "quit"):
                print("Exiting system...")
                break
            if not query:
                continue

            response = generate_augmented_response(query, hybrid_retriever)
            print("\n--- System Response ---")
            print(response)
            print("-" * 25)

        except KeyboardInterrupt:
            print("\nExiting system...")
            break
        except Exception as e:
            print(f"An error occurred while answering query: {e}")


if __name__ == "__main__":
    main()