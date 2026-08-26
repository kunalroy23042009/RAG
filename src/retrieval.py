"""
Processed-file log, hybrid (semantic + BM25) retrieval, and answer
generation. Converted from notebook cells 8, 14 and 15 — logic unchanged.
"""
import json
import os

from langchain_community.vectorstores import Chroma
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from config import LOG_PATH, BM25_DOCS_PATH, CHROMA_PATH, GOOGLE_API_KEY
from google.genai import types
from src.extraction import client

def load_processed_log():
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r") as f:
            return json.load(f)
    return []


def save_processed_log(log):
    with open(LOG_PATH, "w") as f:
        json.dump(log, f)


def load_bm25_docs():
    if os.path.exists(BM25_DOCS_PATH):
        with open(BM25_DOCS_PATH, "r") as f:
            return json.load(f)
    return []


def save_bm25_docs(docs):
    with open(BM25_DOCS_PATH, "w") as f:
        json.dump(docs, f, indent=2)


def configure_bm25_only_retriever(documents):
    """Create a BM25-only retriever (no embedding model needed)."""
    existing_docs = load_bm25_docs()
    new_docs = [{"content": doc.page_content, "metadata": doc.metadata} for doc in documents]
    all_texts = existing_docs + new_docs
    save_bm25_docs(all_texts)

    if not all_texts:
        # Return a dummy retriever that returns empty results
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.documents import Document
        from typing import List
        from pydantic import Field
        
        class EmptyRetriever(BaseRetriever):
            """Retriever that returns empty results."""
            def _get_relevant_documents(self, query: str) -> List[Document]:
                return []
            async def _aget_relevant_documents(self, query: str) -> List[Document]:
                return []
        
        return EmptyRetriever()

    bm25_retriever = BM25Retriever.from_texts([doc["content"] for doc in all_texts])
    bm25_retriever.k = 4
    return bm25_retriever


def configure_hybrid_retrieval_system(documents, embeddings):
    try:
        vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

        if documents:
            vector_store.add_documents(documents)

        semantic_retriever = vector_store.as_retriever(search_kwargs={"k": 4})

        existing_docs = load_bm25_docs()
        new_docs = [{"content": doc.page_content, "metadata": doc.metadata} for doc in documents]
        all_texts = existing_docs + new_docs
        save_bm25_docs(all_texts)

        retrievers_to_ensemble = [semantic_retriever]
        weights = [1.0]

        if all_texts:
            bm25_retriever = BM25Retriever.from_texts([doc["content"] for doc in all_texts])
            bm25_retriever.k = 4
            retrievers_to_ensemble.append(bm25_retriever)
            weights = [0.6, 0.4]

        return EnsembleRetriever(retrievers=retrievers_to_ensemble, weights=weights)
    except MemoryError:
        # Fallback to BM25-only if memory is low
        return configure_bm25_only_retriever(documents)
    except Exception:
        # Fallback to BM25-only on any error
        return configure_bm25_only_retriever(documents)


def generate_augmented_response(user_query: str, hybrid_retriever: EnsembleRetriever) -> str:
    retrieved_documents = hybrid_retriever.invoke(user_query)
    aggregated_context = "\n\n---\n\n".join(doc.page_content for doc in retrieved_documents)

    generation_prompt = f"""
    You are an administrative intelligence assistant for academic notices.

    Use the provided context to answer questions about notices, exams, holidays, events, circulars, and admissions.
    If the question is about information in the context, answer from the context.
    If the question is a general conversational query (greetings, thanks, how are you, etc.) or unrelated to notices, respond naturally and helpfully.
    If the question is about notices but the information is not in the context, say: "Information regarding this query is not available in the processed notices."

    Context:
    {aggregated_context}

    Query: {user_query}
    """

    import concurrent.futures

    def call_llm_with_timeout(prompt, timeout=30):
        """Call LLM with timeout."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                lambda: client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.2),
                )
            )
            return future.result(timeout=timeout)

    response = call_llm_with_timeout(generation_prompt, timeout=30)

    return response.text
def generate_augmented_response_stream(user_query: str, hybrid_retriever: EnsembleRetriever):
    """
    Same as generate_augmented_response, but yields the answer word-by-word
    as it's generated, instead of returning the full text at once. Built
    for st.write_stream() in the chat UI.
    """
    retrieved_documents = hybrid_retriever.invoke(user_query)
    aggregated_context = "\n\n---\n\n".join(doc.page_content for doc in retrieved_documents)

    generation_prompt = f"""
    You are an administrative intelligence assistant for academic notices.

    Use the provided context to answer questions about notices, exams, holidays, events, circulars, and admissions.
    If the question is about information in the context, answer from the context.
    If the question is a general conversational query (greetings, thanks, how are you, etc.) or unrelated to notices, respond naturally and helpfully.
    If the question is about notices but the information is not in the context, say: "Information regarding this query is not available in the processed notices."

    Context:
    {aggregated_context}

    Query: {user_query}
"""

    def generate_augmented_response_stream(user_query: str, hybrid_retriever: EnsembleRetriever):
        """
        Same as generate_augmented_response, but yields the answer word-by-word
        as it's generated, instead of returning the full text at once. Built
        for st.write_stream() in the chat UI.
        """
        retrieved_documents = hybrid_retriever.invoke(user_query)
    aggregated_context = "\n\n---\n\n".join(doc.page_content for doc in retrieved_documents)

    generation_prompt = f"""
    You are an administrative intelligence assistant for academic notices.

    Use the provided context to answer questions about notices, exams, holidays, events, circulars, and admissions.
    If the question is about information in the context, answer from the context.
    If the question is a general conversational query (greetings, thanks, how are you, etc.) or unrelated to notices, respond naturally and helpfully.
    If the question is about notices but the information is not in the context, say: "Information regarding this query is not available in the processed notices."

    Context:
    {aggregated_context}

    Query: {user_query}
    """

    def stream_with_timeout():
        response_stream = client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=generation_prompt,
            config=types.GenerateContentConfig(temperature=0.2),
        )
        for chunk in response_stream:
            if chunk.text:
                for word in chunk.text.split(" "):
                    if word:
                        yield word + " "

    # Run with timeout
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(lambda: list(stream_with_timeout()))
        try:
            for word in future.result(timeout=30):
                yield word
        except concurrent.futures.TimeoutError:
            yield " [Response timed out]"

    # Fallback to simple streaming if timeout doesn't work
    for chunk in client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=generation_prompt,
        config=types.GenerateContentConfig(temperature=0.2),
    ):
        if chunk.text:
            for word in chunk.text.split(" "):
                if word:
                    yield word + " "
