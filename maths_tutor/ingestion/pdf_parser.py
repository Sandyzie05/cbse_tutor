"""
PDF Parser - Extracts text from PDF files.

This module handles the extraction of text content from PDF files.
It uses pymupdf (fitz) which is excellent for handling complex PDFs
including those with mathematical content.

Key Concepts:
- PDFs store text in a structured way (pages, blocks, lines)
- Mathematical symbols may not always extract perfectly
- Images and diagrams are not extracted (text only)
- We preserve page boundaries for better metadata
"""

import fitz  # pymupdf - the library is called 'fitz' historically
from pathlib import Path
from typing import Generator
from dataclasses import dataclass


@dataclass
class PageContent:
    """
    Represents the content of a single PDF page.
    
    Attributes:
        page_number: 1-indexed page number
        text: Extracted text content
        source_file: Name of the source PDF file
    """
    page_number: int
    text: str
    source_file: str


@dataclass
class DocumentContent:
    """
    Represents the full content of a PDF document.
    
    Attributes:
        filename: Name of the PDF file
        total_pages: Total number of pages
        pages: List of PageContent objects
        full_text: All text concatenated
    """
    filename: str
    total_pages: int
    pages: list[PageContent]
    full_text: str


class PDFParser:
    """
    Parses PDF files and extracts text content.
    
    This class provides methods to:
    - Extract text from a single PDF
    - Extract text from multiple PDFs in a directory
    - Get page-by-page content with metadata
    
    Example:
        parser = PDFParser()
        content = parser.parse_pdf("chapter1.pdf")
        print(content.full_text)
    """
    
    def __init__(self, clean_text: bool = True):
        """
        Initialize the PDF parser.
        
        Args:
            clean_text: If True, apply text cleaning (remove extra whitespace, etc.)
        """
        self.clean_text = clean_text
    
    def _clean_extracted_text(self, text: str) -> str:
        """
        Clean extracted text by removing common artifacts.
        
        This handles issues like:
        - Multiple consecutive newlines
        - Extra whitespace
        - Page headers/footers (basic removal)
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not self.clean_text:
            return text
        
        # Replace multiple newlines with double newline (paragraph break)
        import re
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove lines that are just numbers (likely page numbers)
        lines = text.split('\n')
        cleaned_lines = [
            line for line in lines 
            if not (line.strip().isdigit() and len(line.strip()) < 4)
        ]
        text = '\n'.join(cleaned_lines)
        
        return text.strip()
    
    def parse_pdf(self, pdf_path: str | Path) -> DocumentContent:
        """
        Parse a single PDF file and extract all text.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            DocumentContent with all extracted text and metadata
            
        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            RuntimeError: If the PDF cannot be parsed
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF {pdf_path}: {e}")
        
        pages = []
        all_text = []
        total_pages = len(doc)  # Save before closing!
        
        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text()
            cleaned = self._clean_extracted_text(text)
            
            if cleaned:  # Only add non-empty pages
                page_content = PageContent(
                    page_number=page_num + 1,  # 1-indexed
                    text=cleaned,
                    source_file=pdf_path.name
                )
                pages.append(page_content)
                all_text.append(cleaned)
        
        doc.close()
        
        return DocumentContent(
            filename=pdf_path.name,
            total_pages=total_pages,
            pages=pages,
            full_text='\n\n'.join(all_text)
        )
    
    def parse_directory(
        self, 
        directory: str | Path, 
        pattern: str = "*.pdf"
    ) -> Generator[DocumentContent, None, None]:
        """
        Parse all PDF files in a directory.
        
        Args:
            directory: Path to the directory containing PDFs
            pattern: Glob pattern for PDF files (default: "*.pdf")
            
        Yields:
            DocumentContent for each PDF file found
            
        Example:
            parser = PDFParser()
            for doc in parser.parse_directory("./pdfs"):
                print(f"Parsed {doc.filename}: {doc.total_pages} pages")
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        pdf_files = sorted(directory.glob(pattern))
        
        for pdf_path in pdf_files:
            try:
                yield self.parse_pdf(pdf_path)
            except Exception as e:
                print(f"Warning: Failed to parse {pdf_path}: {e}")
                continue
    
    def get_text_with_page_markers(self, pdf_path: str | Path) -> str:
        """
        Extract text with page markers for easier reference.
        
        This is useful when you want to know which page content came from.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Text with [Page X] markers
        """
        content = self.parse_pdf(pdf_path)
        marked_text = []
        
        for page in content.pages:
            marked_text.append(f"[Page {page.page_number}]")
            marked_text.append(page.text)
            marked_text.append("")  # Empty line after each page
        
        return '\n'.join(marked_text)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def extract_text_from_pdf(pdf_path: str | Path, clean: bool = True) -> str:
    """
    Simple function to extract all text from a PDF.
    
    This is a convenience wrapper around PDFParser for simple use cases.
    
    Args:
        pdf_path: Path to the PDF file
        clean: Whether to clean the extracted text
        
    Returns:
        Extracted text as a string
        
    Example:
        text = extract_text_from_pdf("chapter1.pdf")
        print(text[:500])  # First 500 characters
    """
    parser = PDFParser(clean_text=clean)
    content = parser.parse_pdf(pdf_path)
    return content.full_text


def get_pdf_info(pdf_path: str | Path) -> dict:
    """
    Get basic information about a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary with PDF metadata
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    
    info = {
        "filename": pdf_path.name,
        "pages": len(doc),
        "metadata": doc.metadata,
    }
    
    doc.close()
    return info


# =============================================================================
# MAIN - For testing
# =============================================================================

if __name__ == "__main__":
    """
    Test the PDF parser with a sample file.
    Run: python -m maths_tutor.ingestion.pdf_parser
    """
    from maths_tutor.config import BOOKS_DIR
    
    print("=" * 60)
    print("PDF PARSER TEST")
    print("=" * 60)
    
    # Find first PDF across all subject directories
    pdf_files = list(BOOKS_DIR.rglob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {BOOKS_DIR}")
        exit(1)
    
    test_pdf = pdf_files[0]
    print(f"\nTesting with: {test_pdf.name}")
    print("-" * 60)
    
    # Get PDF info
    info = get_pdf_info(test_pdf)
    print(f"Pages: {info['pages']}")
    print(f"Title: {info['metadata'].get('title', 'N/A')}")
    
    # Extract text
    parser = PDFParser()
    content = parser.parse_pdf(test_pdf)
    
    print(f"\nExtracted {len(content.pages)} pages with content")
    print(f"Total characters: {len(content.full_text)}")
    
    # Show sample
    print("\n--- Sample text (first 500 chars) ---")
    print(content.full_text[:500])
    print("...")
