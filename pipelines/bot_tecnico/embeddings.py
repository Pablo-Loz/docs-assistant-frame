"""
Embedding functions for vector storage.

Single Responsibility: Generate embeddings only.
"""

from chromadb import EmbeddingFunction, Documents, Embeddings
from google import genai


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Custom embedding function using Google Gemini embedding-001."""

    MODEL = "models/gemini-embedding-001"

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for a list of documents."""
        embeddings = []
        for text in input:
            result = self.client.models.embed_content(
                model=self.MODEL,
                contents=text,
            )
            embeddings.append(result.embeddings[0].values)
        return embeddings
