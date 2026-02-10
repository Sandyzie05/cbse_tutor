"""
Ingestion module - Handles PDF parsing and text chunking.

This module is responsible for:
1. Extracting text from PDF files
2. Splitting text into manageable chunks for embedding
"""

from .pdf_parser import PDFParser, extract_text_from_pdf
from .chunker import TextChunker, chunk_text

__all__ = ["PDFParser", "extract_text_from_pdf", "TextChunker", "chunk_text"]
