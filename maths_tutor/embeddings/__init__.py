"""
Embeddings module - Handles embedding generation and vector storage.

This module is responsible for:
1. Converting text chunks to embeddings
2. Storing and retrieving embeddings from ChromaDB
"""

from .embedder import Embedder, get_embeddings
from .vector_store import VectorStore

__all__ = ["Embedder", "get_embeddings", "VectorStore"]
