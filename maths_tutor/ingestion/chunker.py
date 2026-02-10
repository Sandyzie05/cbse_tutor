"""
Text Chunker - Splits text into smaller pieces for embedding.

This module handles the splitting of long text into manageable chunks
that can be effectively embedded and retrieved.

Key Concepts:
- Chunk Size: How many characters per chunk (default: 800)
- Overlap: How many characters overlap between chunks (default: 100)
- Why Overlap? To avoid losing context at chunk boundaries

Example:
    Text: "ABCDEFGHIJ" (10 chars)
    Chunk size: 5, Overlap: 2
    
    Chunk 1: "ABCDE"
    Chunk 2: "DEFGH"  <- 'DE' overlaps with chunk 1
    Chunk 3: "GHIJ"   <- 'GH' overlaps with chunk 2
"""

from dataclasses import dataclass, field
from typing import Generator
import re


@dataclass
class TextChunk:
    """
    Represents a single chunk of text with metadata.
    
    Attributes:
        text: The chunk content
        chunk_index: Position of this chunk (0-indexed)
        start_char: Character position where chunk starts in original text
        end_char: Character position where chunk ends in original text
        metadata: Additional metadata (source file, page, etc.)
    """
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)
    
    @property
    def char_count(self) -> int:
        """Return the number of characters in this chunk."""
        return len(self.text)
    
    @property
    def word_count(self) -> int:
        """Return approximate word count."""
        return len(self.text.split())


class TextChunker:
    """
    Splits text into overlapping chunks for embedding.
    
    The chunker uses a simple but effective strategy:
    1. Try to split at sentence boundaries when possible
    2. Fall back to word boundaries
    3. Maintain overlap between chunks
    
    Example:
        chunker = TextChunker(chunk_size=800, chunk_overlap=100)
        chunks = chunker.chunk_text("Your long text here...")
        for chunk in chunks:
            print(f"Chunk {chunk.chunk_index}: {chunk.char_count} chars")
    """
    
    def __init__(
        self, 
        chunk_size: int = 800, 
        chunk_overlap: int = 100,
        min_chunk_size: int = 100
    ):
        """
        Initialize the chunker with size parameters.
        
        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            min_chunk_size: Minimum chunk size (smaller chunks are merged)
            
        Raises:
            ValueError: If overlap >= chunk_size or sizes are invalid
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("Overlap must be less than chunk size")
        if chunk_size < 50:
            raise ValueError("Chunk size must be at least 50 characters")
        if min_chunk_size < 10:
            raise ValueError("Minimum chunk size must be at least 10")
            
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # Sentence-ending patterns (for smart splitting)
        self.sentence_endings = re.compile(r'[.!?]\s+')
    
    def _find_split_point(self, text: str, target_pos: int) -> int:
        """
        Find a good split point near the target position.
        
        Tries to split at:
        1. Sentence boundary (. ! ?)
        2. Paragraph boundary (newline)
        3. Word boundary (space)
        4. Exact position (fallback)
        
        Args:
            text: Text to find split point in
            target_pos: Target position to split near
            
        Returns:
            Best split position
        """
        # Search window: look back up to 100 chars from target
        search_start = max(0, target_pos - 100)
        search_text = text[search_start:target_pos]
        
        # Try to find sentence boundary
        sentence_matches = list(self.sentence_endings.finditer(search_text))
        if sentence_matches:
            # Use the last sentence boundary found
            last_match = sentence_matches[-1]
            return search_start + last_match.end()
        
        # Try to find paragraph boundary
        newline_pos = search_text.rfind('\n')
        if newline_pos != -1 and newline_pos > len(search_text) // 2:
            return search_start + newline_pos + 1
        
        # Try to find word boundary
        space_pos = search_text.rfind(' ')
        if space_pos != -1:
            return search_start + space_pos + 1
        
        # Fallback: use target position
        return target_pos
    
    def chunk_text(
        self, 
        text: str, 
        metadata: dict | None = None
    ) -> list[TextChunk]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: The text to chunk
            metadata: Optional metadata to attach to all chunks
            
        Returns:
            List of TextChunk objects
        """
        if not text or not text.strip():
            return []
        
        # Clean the text
        text = text.strip()
        metadata = metadata or {}
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Determine end position
            end = start + self.chunk_size
            
            if end >= len(text):
                # Last chunk - take everything remaining
                end = len(text)
            else:
                # Find a good split point
                end = self._find_split_point(text, end)
            
            # Extract chunk text
            chunk_text = text[start:end].strip()
            
            # Only add if chunk meets minimum size
            if len(chunk_text) >= self.min_chunk_size:
                chunk = TextChunk(
                    text=chunk_text,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                    metadata=metadata.copy()
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position (accounting for overlap)
            # If this is the last chunk, we're done
            if end >= len(text):
                break
            
            start = end - self.chunk_overlap
            
            # Safety: ensure we're making progress
            if start >= end:
                start = end
        
        return chunks
    
    def chunk_with_pages(
        self, 
        pages: list[tuple[int, str]], 
        source_file: str = ""
    ) -> list[TextChunk]:
        """
        Chunk text while preserving page information.
        
        This method handles page-by-page content and tracks which
        page(s) each chunk came from.
        
        Args:
            pages: List of (page_number, page_text) tuples
            source_file: Name of the source file
            
        Returns:
            List of TextChunk objects with page metadata
        """
        all_chunks = []
        
        for page_num, page_text in pages:
            if not page_text.strip():
                continue
                
            metadata = {
                "source_file": source_file,
                "page_number": page_num,
            }
            
            page_chunks = self.chunk_text(page_text, metadata)
            all_chunks.extend(page_chunks)
        
        # Re-index chunks sequentially
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_index = i
        
        return all_chunks


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def chunk_text(
    text: str, 
    chunk_size: int = 800, 
    chunk_overlap: int = 100
) -> list[str]:
    """
    Simple function to chunk text and return just the text strings.
    
    This is a convenience wrapper for simple use cases.
    
    Args:
        text: Text to chunk
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunk text strings
        
    Example:
        chunks = chunk_text("Your long text...", chunk_size=500)
        print(f"Created {len(chunks)} chunks")
    """
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_text(text)
    return [chunk.text for chunk in chunks]


def estimate_chunks(text_length: int, chunk_size: int = 800, overlap: int = 100) -> int:
    """
    Estimate how many chunks will be created from text of given length.
    
    Args:
        text_length: Length of text in characters
        chunk_size: Target chunk size
        overlap: Overlap between chunks
        
    Returns:
        Estimated number of chunks
    """
    if text_length <= chunk_size:
        return 1
    
    effective_step = chunk_size - overlap
    return (text_length - overlap) // effective_step + 1


# =============================================================================
# MAIN - For testing
# =============================================================================

if __name__ == "__main__":
    """
    Test the chunker with sample text.
    Run: python -m maths_tutor.ingestion.chunker
    """
    print("=" * 60)
    print("TEXT CHUNKER TEST")
    print("=" * 60)
    
    # Sample text (a simplified maths lesson)
    sample_text = """
    Chapter 1: Understanding Numbers
    
    Numbers are everywhere in our daily life. We use numbers to count things, 
    measure quantities, and solve problems. In this chapter, we will learn about 
    different types of numbers and how to work with them.
    
    Section 1.1: Whole Numbers
    
    Whole numbers are the numbers we use for counting: 0, 1, 2, 3, 4, 5, and so on.
    They do not include fractions or decimals. Whole numbers go on forever - there 
    is no largest whole number!
    
    Let's practice counting:
    - Count from 1 to 10: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    - Count by twos: 2, 4, 6, 8, 10, 12, 14, 16, 18, 20
    - Count by fives: 5, 10, 15, 20, 25, 30, 35, 40, 45, 50
    
    Section 1.2: Place Value
    
    In our number system, the position of a digit tells us its value. This is 
    called place value. For example, in the number 352:
    - The 3 is in the hundreds place, so it means 300
    - The 5 is in the tens place, so it means 50
    - The 2 is in the ones place, so it means 2
    
    So 352 = 300 + 50 + 2
    
    Understanding place value helps us read and write large numbers correctly.
    It also helps us add and subtract numbers more easily.
    
    Section 1.3: Comparing Numbers
    
    We can compare numbers using these symbols:
    - Greater than: >
    - Less than: <
    - Equal to: =
    
    For example:
    - 5 > 3 (5 is greater than 3)
    - 2 < 7 (2 is less than 7)
    - 4 = 4 (4 is equal to 4)
    
    When comparing larger numbers, we look at the digits from left to right.
    The number with the larger digit in the leftmost position is greater.
    """
    
    print(f"\nOriginal text length: {len(sample_text)} characters")
    print(f"Estimated chunks: {estimate_chunks(len(sample_text))}")
    print("-" * 60)
    
    # Create chunker with default settings
    chunker = TextChunker(chunk_size=400, chunk_overlap=50)  # Smaller for demo
    chunks = chunker.chunk_text(sample_text, metadata={"source": "demo"})
    
    print(f"\nCreated {len(chunks)} chunks:")
    print("-" * 60)
    
    for chunk in chunks:
        print(f"\n[Chunk {chunk.chunk_index}] ({chunk.char_count} chars, {chunk.word_count} words)")
        print(f"Position: chars {chunk.start_char}-{chunk.end_char}")
        print("-" * 40)
        # Show first 150 chars of each chunk
        preview = chunk.text[:150].replace('\n', ' ')
        print(f"{preview}...")
    
    print("\n" + "=" * 60)
    print("OVERLAP DEMONSTRATION")
    print("=" * 60)
    
    if len(chunks) >= 2:
        # Show overlap between first two chunks
        chunk1_end = chunks[0].text[-100:]
        chunk2_start = chunks[1].text[:100]
        
        print("\nEnd of Chunk 0 (last 100 chars):")
        print(f"  ...{chunk1_end}")
        print("\nStart of Chunk 1 (first 100 chars):")
        print(f"  {chunk2_start}...")
        print("\n(Notice the overlapping text!)")
