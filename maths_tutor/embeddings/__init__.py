"""
Embeddings module - Handles embedding generation and vector storage.

This module is responsible for:
1. Converting text chunks to embeddings
2. Storing and retrieving embeddings from ChromaDB

Import directly from submodules to avoid pulling in heavy dependencies
(sentence-transformers) when only the vector store is needed:

    from maths_tutor.embeddings.vector_store import VectorStore
    from maths_tutor.embeddings.embedder import Embedder
"""
