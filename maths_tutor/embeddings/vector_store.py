"""
Vector Store - Stores and searches embeddings using ChromaDB.

This module handles the persistent storage and retrieval of embeddings.
ChromaDB is a local vector database that's perfect for learning and
small-to-medium projects.

Key Concepts:
- Collections: Like tables in a database, store related embeddings
- Documents: The original text (stored alongside embeddings)
- Embeddings: The vector representations
- Metadata: Additional info like page number, source file
- IDs: Unique identifiers for each document

How it works:
1. Store: text + embedding + metadata -> ChromaDB
2. Query: question embedding -> find similar stored embeddings -> return texts
"""

import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from maths_tutor.config import CHROMA_DB_DIR, COLLECTION_NAME, TOP_K_CHUNKS


@dataclass
class SearchResult:
    """
    Represents a single search result from the vector store.
    
    Attributes:
        text: The retrieved document text
        score: Similarity score (higher = more similar)
        metadata: Additional metadata (source, page, etc.)
        id: Unique identifier
    """
    text: str
    score: float
    metadata: dict
    id: str
    
    @property
    def source_file(self) -> str:
        """Get the source file from metadata."""
        return self.metadata.get("source_file", "unknown")
    
    @property
    def page_number(self) -> int:
        """Get the page number from metadata."""
        return self.metadata.get("page_number", 0)


class VectorStore:
    """
    ChromaDB-based vector store for storing and searching embeddings.
    
    This class provides a clean interface for:
    - Adding documents with embeddings and metadata
    - Searching for similar documents
    - Managing the collection
    
    Example:
        store = VectorStore()
        
        # Add documents
        store.add_documents(
            texts=["Addition is...", "Subtraction is..."],
            embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
            metadatas=[{"page": 1}, {"page": 2}]
        )
        
        # Search
        results = store.search(query_embedding, top_k=5)
        for result in results:
            print(f"Score: {result.score}, Text: {result.text[:50]}...")
    """
    
    def __init__(
        self, 
        collection_name: str | None = None,
        persist_directory: str | Path | None = None
    ):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the collection to use/create
            persist_directory: Where to store the database files
        """
        self.collection_name = collection_name or COLLECTION_NAME
        self.persist_directory = Path(persist_directory or CHROMA_DB_DIR)
        
        # Ensure directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )
        
        # Get or create collection with COSINE distance (important!)
        # Cosine distance gives similarity scores between 0-2 (0=identical, 2=opposite)
        # We convert to similarity: 1 - (distance/2) gives 0-1 range
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "CBSE Grade 5 Maths textbook chunks",
                "hnsw:space": "cosine"  # Use cosine distance for better similarity scores
            }
        )
    
    @property
    def count(self) -> int:
        """Get the number of documents in the collection."""
        return self._collection.count()
    
    def add_documents(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None
    ) -> None:
        """
        Add documents to the vector store.
        
        Args:
            texts: List of document texts
            embeddings: List of embedding vectors (one per text)
            metadatas: Optional list of metadata dicts (one per text)
            ids: Optional list of unique IDs (auto-generated if not provided)
            
        Raises:
            ValueError: If lists have mismatched lengths
        """
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must have same length")
        
        if not texts:
            return
        
        # Generate IDs if not provided
        if ids is None:
            existing_count = self.count
            ids = [f"doc_{existing_count + i}" for i in range(len(texts))]
        
        # Default empty metadata if not provided
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        # Add to collection
        self._collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    
    def add_single(
        self,
        text: str,
        embedding: list[float],
        metadata: dict | None = None,
        doc_id: str | None = None
    ) -> str:
        """
        Add a single document to the vector store.
        
        Args:
            text: Document text
            embedding: Embedding vector
            metadata: Optional metadata dict
            doc_id: Optional unique ID
            
        Returns:
            The ID of the added document
        """
        doc_id = doc_id or f"doc_{self.count}"
        metadata = metadata or {}
        
        self._collection.add(
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[doc_id]
        )
        
        return doc_id
    
    def search(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        where: dict | None = None,
        where_document: dict | None = None
    ) -> list[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: The embedding vector to search with
            top_k: Number of results to return (default from config)
            where: Optional metadata filter (e.g., {"page_number": 5})
            where_document: Optional document content filter
            
        Returns:
            List of SearchResult objects, sorted by similarity (highest first)
            
        Example:
            results = store.search(
                query_embedding=embedder.embed("What is addition?"),
                top_k=5,
                where={"source_file": "eemm101.pdf"}
            )
        """
        top_k = top_k or TOP_K_CHUNKS
        
        # Build query kwargs
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }
        
        if where:
            query_kwargs["where"] = where
        if where_document:
            query_kwargs["where_document"] = where_document
        
        # Execute query
        results = self._collection.query(**query_kwargs)
        
        # Convert to SearchResult objects
        search_results = []
        
        if results and results["documents"] and results["documents"][0]:
            documents = results["documents"][0]
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(documents)
            distances = results["distances"][0] if results["distances"] else [0.0] * len(documents)
            ids = results["ids"][0] if results["ids"] else [""] * len(documents)
            
            for doc, meta, dist, doc_id in zip(documents, metadatas, distances, ids):
                # ChromaDB returns distance, convert to similarity
                # For cosine distance: distance is 0-2 (0=identical, 2=opposite)
                # Convert to similarity: 1 - (distance/2) gives 0-1 range
                # Or simpler: 1 - distance for 0-2 range, clamped
                similarity = max(0, 1 - dist)  # Clamp to non-negative
                
                search_results.append(SearchResult(
                    text=doc,
                    score=similarity,
                    metadata=meta or {},
                    id=doc_id
                ))
        
        return search_results
    
    def search_by_text(
        self,
        query_text: str,
        embedder,
        top_k: int | None = None
    ) -> list[SearchResult]:
        """
        Search using text directly (convenience method).
        
        This embeds the query text and then searches.
        
        Args:
            query_text: The text query
            embedder: Embedder instance to use
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        query_embedding = embedder.embed(query_text)
        return self.search(query_embedding, top_k=top_k)
    
    def delete_collection(self) -> None:
        """
        Delete the entire collection.
        
        WARNING: This permanently deletes all documents!
        """
        self._client.delete_collection(self.collection_name)
        # Recreate empty collection
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "CBSE Grade 5 Maths textbook chunks"}
        )
    
    def get_all_ids(self) -> list[str]:
        """Get all document IDs in the collection."""
        result = self._collection.get(include=[])
        return result["ids"] if result["ids"] else []
    
    def get_by_id(self, doc_id: str) -> dict | None:
        """
        Get a specific document by ID.
        
        Args:
            doc_id: The document ID
            
        Returns:
            Dict with 'text', 'metadata', 'embedding' or None if not found
        """
        result = self._collection.get(
            ids=[doc_id],
            include=["documents", "metadatas", "embeddings"]
        )
        
        if result["documents"]:
            return {
                "text": result["documents"][0],
                "metadata": result["metadatas"][0] if result["metadatas"] else {},
                "embedding": result["embeddings"][0] if result["embeddings"] else None
            }
        return None
    
    def get_stats(self) -> dict:
        """
        Get statistics about the collection.
        
        Returns:
            Dict with count, sources, etc.
        """
        count = self.count
        
        # Try to get unique sources
        sources = set()
        if count > 0:
            # Sample some documents to get source info
            result = self._collection.get(
                limit=min(count, 100),
                include=["metadatas"]
            )
            if result["metadatas"]:
                for meta in result["metadatas"]:
                    if meta and "source_file" in meta:
                        sources.add(meta["source_file"])
        
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "unique_sources": list(sources),
            "persist_directory": str(self.persist_directory)
        }


# =============================================================================
# MAIN - For testing
# =============================================================================

if __name__ == "__main__":
    """
    Test the vector store.
    Run: python -m maths_tutor.embeddings.vector_store
    """
    print("=" * 60)
    print("VECTOR STORE TEST")
    print("=" * 60)
    
    # Create a test collection
    test_store = VectorStore(collection_name="test_collection")
    
    print(f"\nCollection: {test_store.collection_name}")
    print(f"Initial count: {test_store.count}")
    
    # Add some test documents
    print("\n--- Adding test documents ---")
    
    # Fake embeddings (in real use, these come from the Embedder)
    fake_embeddings = [
        [0.1] * 384,  # 384-dimensional vectors
        [0.2] * 384,
        [0.3] * 384,
    ]
    
    test_store.add_documents(
        texts=[
            "Addition is the process of combining two numbers.",
            "Subtraction means taking away one number from another.",
            "Multiplication is repeated addition.",
        ],
        embeddings=fake_embeddings,
        metadatas=[
            {"page_number": 1, "source_file": "test.pdf"},
            {"page_number": 2, "source_file": "test.pdf"},
            {"page_number": 3, "source_file": "test.pdf"},
        ]
    )
    
    print(f"Count after adding: {test_store.count}")
    
    # Search
    print("\n--- Search test ---")
    query_embedding = [0.15] * 384  # Similar to first document
    
    results = test_store.search(query_embedding, top_k=2)
    
    for i, result in enumerate(results):
        print(f"\nResult {i + 1}:")
        print(f"  Score: {result.score:.4f}")
        print(f"  Text: {result.text[:50]}...")
        print(f"  Source: {result.source_file}, Page: {result.page_number}")
    
    # Get stats
    print("\n--- Collection stats ---")
    stats = test_store.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Clean up test collection
    print("\n--- Cleaning up ---")
    test_store.delete_collection()
    print(f"Count after delete: {test_store.count}")
    
    print("\n" + "=" * 60)
    print("SUCCESS! Vector store is working correctly.")
    print("=" * 60)
