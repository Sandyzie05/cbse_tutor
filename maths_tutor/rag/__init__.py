"""
RAG module - Retrieval-Augmented Generation pipeline.

This module is responsible for:
1. Retrieving relevant chunks for a query
2. Generating responses using Ollama
"""

from .retriever import Retriever
from .generator import Generator

__all__ = ["Retriever", "Generator"]
