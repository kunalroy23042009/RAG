"""
Multilingual embedding model setup. Converted from notebook cell 11 — logic
is unchanged.
"""
from langchain_huggingface import HuggingFaceEmbeddings


def initialize_multilingual_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initializes the BAAI BGE-M3 multilingual embedding model via HuggingFace.
    Selected for Hindi-English code-mixed text handling and 8192 token context.
    """
    model_name = "BAAI/bge-m3"
    model_kwargs = {"device": "cpu"}  # switch to "cuda" if a GPU is available
    encode_kwargs = {"normalize_embeddings": True}

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
