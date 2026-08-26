"""
Multilingual embedding model setup. Converted from notebook cell 11 — logic
is unchanged.
"""
from langchain_huggingface import HuggingFaceEmbeddings


def initialize_multilingual_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initializes a lightweight multilingual embedding model via HuggingFace.
    Uses sentence-transformers/all-MiniLM-L6-v2 (~80MB) for fast downloads
    and low memory usage on free tier hosting.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
