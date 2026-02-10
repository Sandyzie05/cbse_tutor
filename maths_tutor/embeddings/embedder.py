"""
Embedder - Converts text to vector embeddings.

This module handles the conversion of text into numerical vectors (embeddings)
using sentence-transformers. This is EXACTLY what you learned in Week 1!

Key Concepts:
- Embeddings are lists of numbers that represent meaning
- Similar text = Similar embeddings
- We use the same model for storing and querying (important!)
- The model creates 384-dimensional vectors

Example:
    embedder = Embedder()
    
    # Single text
    vector = embedder.embed("What is addition?")
    print(len(vector))  # 384
    
    # Multiple texts (more efficient)
    vectors = embedder.embed_batch(["text1", "text2", "text3"])
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Union
from pathlib import Path

from maths_tutor.config import EMBEDDING_MODEL, EMBEDDING_DIMENSION


class Embedder:
    """
    Converts text to vector embeddings using sentence-transformers.
    
    This class wraps the SentenceTransformer model to provide a clean
    interface for embedding generation. It handles both single texts
    and batches efficiently.
    
    IMPORTANT: Always use the same model for indexing and querying!
    If you index with 'all-MiniLM-L6-v2', you must query with the same model.
    
    Example:
        embedder = Embedder()
        
        # Embed a question
        question_embedding = embedder.embed("What is 5 + 3?")
        
        # Embed multiple chunks efficiently
        chunk_embeddings = embedder.embed_batch(chunks)
    """
    
    def __init__(self, model_name: str | None = None):
        """
        Initialize the embedder with a model.
        
        Args:
            model_name: Name of the sentence-transformer model to use.
                       Defaults to the model specified in config.
                       
        Note:
            First run will download the model (~90MB for MiniLM).
            Subsequent runs use the cached version.
        """
        self.model_name = model_name or EMBEDDING_MODEL
        self._model = None  # Lazy loading
        
    @property
    def model(self) -> SentenceTransformer:
        """
        Lazy-load the model on first use.
        
        This saves memory if you create an Embedder but don't use it,
        and allows for faster initialization.
        """
        if self._model is None:
            print(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            print(f"Model loaded! Embedding dimension: {self.dimension}")
        return self._model
    
    @property
    def dimension(self) -> int:
        """
        Get the embedding dimension.
        
        For 'all-MiniLM-L6-v2', this is 384.
        """
        return self.model.get_sentence_embedding_dimension()
    
    def embed(self, text: str) -> list[float]:
        """
        Convert a single text to an embedding vector.
        
        Args:
            text: The text to embed
            
        Returns:
            List of floats (the embedding vector)
            
        Example:
            vector = embedder.embed("What is addition?")
            print(f"Vector has {len(vector)} dimensions")
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return [0.0] * EMBEDDING_DIMENSION
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_batch(
        self, 
        texts: list[str], 
        show_progress: bool = True,
        batch_size: int = 32
    ) -> list[list[float]]:
        """
        Convert multiple texts to embeddings efficiently.
        
        Batch processing is MUCH faster than embedding one at a time!
        
        Args:
            texts: List of texts to embed
            show_progress: Whether to show a progress bar
            batch_size: Number of texts to process at once
            
        Returns:
            List of embedding vectors (one per text)
            
        Example:
            chunks = ["chunk1", "chunk2", "chunk3"]
            vectors = embedder.embed_batch(chunks)
            # vectors[i] is the embedding for chunks[i]
        """
        if not texts:
            return []
        
        # Filter out empty texts but remember their positions
        non_empty_indices = []
        non_empty_texts = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                non_empty_indices.append(i)
                non_empty_texts.append(text)
        
        if not non_empty_texts:
            return [[0.0] * EMBEDDING_DIMENSION for _ in texts]
        
        # Batch encode non-empty texts
        embeddings = self.model.encode(
            non_empty_texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            batch_size=batch_size
        )
        
        # Build result with zero vectors for empty texts
        result = [[0.0] * EMBEDDING_DIMENSION for _ in texts]
        for idx, embedding in zip(non_empty_indices, embeddings):
            result[idx] = embedding.tolist()
        
        return result
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.
        
        This is the SAME as what you learned in Week 1!
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
            (1 = identical meaning, 0 = completely different)
            
        Example:
            score = embedder.similarity("cat", "kitten")
            print(f"Similarity: {score:.2f}")  # ~0.8
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        emb1 = np.array(self.embed(text1)).reshape(1, -1)
        emb2 = np.array(self.embed(text2)).reshape(1, -1)
        
        return float(cosine_similarity(emb1, emb2)[0][0])


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global embedder instance (singleton pattern)
_global_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """
    Get or create the global embedder instance.
    
    This ensures we only load the model once, saving memory.
    
    Returns:
        The global Embedder instance
    """
    global _global_embedder
    if _global_embedder is None:
        _global_embedder = Embedder()
    return _global_embedder


def get_embeddings(texts: Union[str, list[str]]) -> Union[list[float], list[list[float]]]:
    """
    Simple function to get embeddings for text(s).
    
    Args:
        texts: Single text string or list of texts
        
    Returns:
        Single embedding vector or list of vectors
        
    Example:
        # Single text
        vector = get_embeddings("What is addition?")
        
        # Multiple texts
        vectors = get_embeddings(["text1", "text2"])
    """
    embedder = get_embedder()
    
    if isinstance(texts, str):
        return embedder.embed(texts)
    else:
        return embedder.embed_batch(texts)


# =============================================================================
# MAIN - For testing
# =============================================================================

if __name__ == "__main__":
    """
    Test the embedder with sample texts.
    Run: python -m maths_tutor.embeddings.embedder
    """
    print("=" * 60)
    print("EMBEDDER TEST")
    print("=" * 60)
    
    # Create embedder
    embedder = Embedder()
    
    # Test single embedding
    print("\n--- Single Text Embedding ---")
    text = "What is addition?"
    embedding = embedder.embed(text)
    print(f"Text: '{text}'")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    
    # Test batch embedding
    print("\n--- Batch Embedding ---")
    texts = [
        "What is addition?",
        "How do you add numbers?",
        "What is the capital of France?",
        "Explain multiplication",
    ]
    embeddings = embedder.embed_batch(texts, show_progress=False)
    print(f"Embedded {len(texts)} texts")
    
    # Test similarity (just like Week 1!)
    print("\n--- Similarity Test ---")
    print("(Remember: similar text = higher score)")
    
    test_pairs = [
        ("What is addition?", "How do you add numbers?"),
        ("What is addition?", "What is the capital of France?"),
        ("multiplication", "times tables"),
        ("fraction", "decimal"),
    ]
    
    for text1, text2 in test_pairs:
        score = embedder.similarity(text1, text2)
        print(f"  '{text1}' <-> '{text2}'")
        print(f"  Similarity: {score:.4f}")
        print()
    
    print("=" * 60)
    print("SUCCESS! Embedder is working correctly.")
    print("=" * 60)
