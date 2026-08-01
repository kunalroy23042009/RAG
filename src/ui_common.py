"""
Shared, cached resources for the Streamlit UI (chat page + admin portal).

The embedding model is slow to load, so it's cached process-wide with
st.cache_resource — this cache is shared across every page of the app,
not just the one that first calls it.
"""
import streamlit as st

from src.embeddings import initialize_multilingual_embedding_model
from src.retrieval import configure_hybrid_retrieval_system


@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def get_embedding_model():
    return initialize_multilingual_embedding_model()


def get_retriever(new_documents=None):
    """
    Builds a hybrid retriever against whatever is currently persisted on
    disk, optionally adding new_documents (e.g. right after the admin
    portal processes a fresh upload). Not cached on purpose — it needs to
    reflect newly uploaded notices immediately.
    """
    embeddings = get_embedding_model()
    return configure_hybrid_retrieval_system(new_documents or [], embeddings)