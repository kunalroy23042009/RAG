"""
Processed-file log, hybrid (semantic + BM25) retrieval, and answer
generation. Converted from notebook cells 8, 14 and 15 — logic unchanged.
"""
import json
import os

from langchain_community.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from config import LOG_PATH, BM25_DOCS_PATH, CHROMA_PATH, GOOGLE_API_KEY
from google.genai import types
from src.extraction import client
def retrieve_relevant_chunks(self, query: str, target_document: str = None, top_k: int = 5):
    """
    Retrieves chunks based on semantic similarity, with optional metadata filtering.
    """
    where_filter = {}
    if target_document:
        where_filter = {"source_document": target_document}

    results = self.collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where_filter if where_filter else None
    )
    
    return results

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


def configure_hybrid_retrieval_system(documents, embeddings):
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


def generate_augmented_response(user_query: str, hybrid_retriever: EnsembleRetriever) -> str:
    retrieved_documents = hybrid_retriever.invoke(user_query)
    aggregated_context = "\n\n---\n\n".join(doc.page_content for doc in retrieved_documents)

    generation_prompt = f"""
    You are an administrative intelligence assistant.

    Answer ONLY from the provided context.
    If not found, say: "Information regarding this query is not available."

    Context:
    {aggregated_context}

    Query: {user_query}
    """

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=generation_prompt,
        config=types.GenerateContentConfig(temperature=0.2),
    )

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
    You are an administrative intelligence assistant.

    Answer ONLY from the provided context.
    If not found, say: "Information regarding this query is not available."

    Context:
    {aggregated_context}

    Query: {user_query}
    """

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
