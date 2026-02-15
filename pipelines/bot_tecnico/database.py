"""
ChromaDB operations for the Bot Tecnico pipeline.

Single Responsibility: Vector database operations only.
"""

from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
import logfire

from .config import CHROMA_DIR, CHROMA_HOST
from .models import ProductInfo


class VectorStore:
    """Manages ChromaDB operations for technical manuals."""

    COLLECTION_NAME = "technical_manuals"

    def __init__(self, api_key: str = None):
        """Initialize the vector store with default embedding function."""
        self._embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        if CHROMA_HOST:
            # Server mode: connect to ChromaDB server
            logfire.info(f"Connecting to ChromaDB server at {CHROMA_HOST}")
            self._client = chromadb.HttpClient(host=CHROMA_HOST.replace("http://", "").split(":")[0],
                                                port=int(CHROMA_HOST.split(":")[-1]))
        else:
            # Local mode: use persistent client (legacy)
            if not CHROMA_DIR.exists():
                raise FileNotFoundError(
                    f"ChromaDB directory not found at {CHROMA_DIR}. "
                    "Please run 'python ingest.py' first to populate the vector database."
                )
            self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))

        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self._embedding_fn,
        )

        logfire.info(f"Connected to ChromaDB with {self.document_count} documents")

    @property
    def document_count(self) -> int:
        """Return the number of documents in the collection."""
        return self._collection.count()

    def discover_products(self) -> tuple[list[str], dict[str, ProductInfo]]:
        """
        Discover available documents from ChromaDB metadata.

        Supports two metadata formats:
        - Structured: product_key, product_code, year, standard (from filename pattern)
        - Simple: source filename only (fallback for non-standard filenames)

        Returns:
            Tuple of (sorted document keys, dict of ProductInfo by key)
        """
        results = self._collection.get(
            limit=10000,
            include=["metadatas"],
        )

        products: dict[str, ProductInfo] = {}

        for metadata in results.get("metadatas", []):
            if not metadata:
                continue

            if "product_key" in metadata:
                key = metadata["product_key"]
                if key not in products:
                    products[key] = ProductInfo(
                        key=key,
                        code=metadata.get("product_code", ""),
                        year=metadata.get("year", ""),
                        standard=metadata.get("standard", ""),
                    )
            elif "source" in metadata:
                source = metadata["source"]
                stem = source.rsplit(".", 1)[0] if "." in source else source
                if stem not in products:
                    readable = stem.replace("_", " ")
                    products[stem] = ProductInfo(
                        key=stem,
                        code=stem.split("_")[0] if "_" in stem else stem,
                        year=next((p for p in stem.split("_") if p.isdigit() and len(p) == 4), ""),
                        standard=readable,
                    )

        sorted_keys = sorted(products.keys())
        logfire.info(f"Discovered documents: {sorted_keys}")

        return sorted_keys, products

    def search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.3,
        product_filter: Optional[str] = None,
    ) -> str:
        """
        Search for relevant context with optional product filtering.

        Args:
            query: The search query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            product_filter: Optional product_key to filter by

        Returns:
            Formatted context string
        """
        query_kwargs = {
            "query_texts": [query],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }

        if product_filter:
            # Support both structured (product_key) and simple (source filename) filters
            query_kwargs["where"] = {
                "$or": [
                    {"product_key": product_filter},
                    {"source": f"{product_filter}.md"},
                    {"source": product_filter},
                ]
            }

        results = self._collection.query(**query_kwargs)

        if not results["documents"] or not results["documents"][0]:
            return ""

        context_parts = []
        for doc, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = 1 - distance
            if similarity >= similarity_threshold:
                product = metadata.get("product_code", "Unknown")
                source = metadata.get("source", "Unknown")
                context_parts.append(
                    f"[{product} - {source} | Relevance: {similarity:.2f}]\n{doc}"
                )
                logfire.info(f"Found relevant doc from {source} (score: {similarity:.2f})")

        return "\n\n---\n\n".join(context_parts)
