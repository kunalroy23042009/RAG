"""
Cloud-based embedding model using Google's Generative AI API.
This avoids loading any local model into memory - uses Google's cloud embedding API.
"""
from langchain_core.embeddings import Embeddings
from google import genai
from config import GOOGLE_API_KEY
from typing import List


class GoogleGenAIEmbeddings(Embeddings):
    """Embeddings using Google's Generative AI API (cloud-based, no local model)."""
    
    def __init__(self, model_name: str = "models/text-embedding-004"):
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.model_name = model_name
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = []
        # Process in batches to avoid API limits
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=batch,
            )
            embeddings.extend([e.values for e in response.embeddings])
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=[text],
        )
        return response.embeddings[0].values


def initialize_multilingual_embedding_model():
    """
    Initializes cloud-based embeddings using Google's Generative AI API.
    No local model loaded - uses Google's cloud embedding API.
    """
    return GoogleGenAIEmbeddings()