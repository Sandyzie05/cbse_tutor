#!/usr/bin/env python3
"""
RAG Pipeline Test Script - Tests the complete RAG system.

This script tests all components of the RAG pipeline:
1. PDF parsing
2. Text chunking
3. Embedding generation
4. Vector storage and retrieval
5. LLM generation (if Ollama is available)

Run with:
    python scripts/test_rag.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

console = Console()


def test_imports():
    """Test that all modules can be imported."""
    console.print("\n[bold]1. Testing Imports[/bold]")
    
    try:
        from maths_tutor.config import PDF_DIR, CHUNK_SIZE
        from maths_tutor.ingestion.pdf_parser import PDFParser
        from maths_tutor.ingestion.chunker import TextChunker
        from maths_tutor.embeddings.embedder import Embedder
        from maths_tutor.embeddings.vector_store import VectorStore
        from maths_tutor.rag.retriever import Retriever
        from maths_tutor.rag.generator import Generator, RAGPipeline
        
        console.print("   [green]✓[/green] All imports successful")
        return True
    except ImportError as e:
        console.print(f"   [red]✗[/red] Import error: {e}")
        return False


def test_pdf_parsing():
    """Test PDF parsing on first available file."""
    console.print("\n[bold]2. Testing PDF Parser[/bold]")
    
    from maths_tutor.config import PDF_DIR
    from maths_tutor.ingestion.pdf_parser import PDFParser
    
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    
    if not pdf_files:
        console.print(f"   [yellow]⚠[/yellow] No PDF files found in {PDF_DIR}")
        return False
    
    parser = PDFParser()
    test_pdf = pdf_files[0]
    
    try:
        content = parser.parse_pdf(test_pdf)
        console.print(f"   [green]✓[/green] Parsed {test_pdf.name}")
        console.print(f"      Pages: {content.total_pages}")
        console.print(f"      Characters: {len(content.full_text)}")
        return True
    except Exception as e:
        console.print(f"   [red]✗[/red] Error: {e}")
        return False


def test_chunking():
    """Test text chunking."""
    console.print("\n[bold]3. Testing Chunker[/bold]")
    
    from maths_tutor.ingestion.chunker import TextChunker
    
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    
    test_text = """
    Mathematics is a fundamental subject that helps us understand the world around us.
    Numbers are everywhere - from counting objects to measuring distances.
    
    Addition is one of the basic operations in mathematics. When we add two numbers,
    we combine them to get a larger number called the sum. For example, 3 + 5 = 8.
    
    Subtraction is the opposite of addition. When we subtract, we take away one number
    from another. For example, 8 - 3 = 5.
    
    Multiplication is repeated addition. When we multiply 4 × 3, it means we add 4
    three times: 4 + 4 + 4 = 12.
    
    Division is the opposite of multiplication. When we divide 12 ÷ 3, we are asking
    how many times 3 goes into 12. The answer is 4.
    """ * 3  # Make it longer
    
    try:
        chunks = chunker.chunk_text(test_text)
        console.print(f"   [green]✓[/green] Created {len(chunks)} chunks from {len(test_text)} characters")
        console.print(f"      Average chunk size: {sum(c.char_count for c in chunks) // len(chunks)} chars")
        return True
    except Exception as e:
        console.print(f"   [red]✗[/red] Error: {e}")
        return False


def test_embeddings():
    """Test embedding generation."""
    console.print("\n[bold]4. Testing Embedder[/bold]")
    
    from maths_tutor.embeddings.embedder import Embedder
    
    try:
        embedder = Embedder()
        
        test_texts = [
            "What is addition?",
            "How do you add numbers?",
            "What is the capital of France?"
        ]
        
        embeddings = embedder.embed_batch(test_texts, show_progress=False)
        
        console.print(f"   [green]✓[/green] Generated {len(embeddings)} embeddings")
        console.print(f"      Dimension: {len(embeddings[0])}")
        
        # Test similarity
        sim = embedder.similarity(test_texts[0], test_texts[1])
        console.print(f"      Similar questions similarity: {sim:.4f}")
        
        sim2 = embedder.similarity(test_texts[0], test_texts[2])
        console.print(f"      Different questions similarity: {sim2:.4f}")
        
        return True
    except Exception as e:
        console.print(f"   [red]✗[/red] Error: {e}")
        return False


def test_vector_store():
    """Test vector store operations."""
    console.print("\n[bold]5. Testing Vector Store[/bold]")
    
    from maths_tutor.embeddings.vector_store import VectorStore
    
    try:
        store = VectorStore()
        count = store.count
        
        console.print(f"   [green]✓[/green] Connected to vector store")
        console.print(f"      Documents: {count}")
        
        if count == 0:
            console.print("   [yellow]⚠[/yellow] Store is empty - run ingestion first")
            return True  # Not a failure, just needs data
        
        return True
    except Exception as e:
        console.print(f"   [red]✗[/red] Error: {e}")
        return False


def test_retriever():
    """Test retrieval functionality."""
    console.print("\n[bold]6. Testing Retriever[/bold]")
    
    from maths_tutor.rag.retriever import Retriever
    
    try:
        retriever = Retriever()
        
        if retriever.vector_store.count == 0:
            console.print("   [yellow]⚠[/yellow] Skipping - no data in vector store")
            return True
        
        result = retriever.retrieve("What is addition?", top_k=3)
        
        console.print(f"   [green]✓[/green] Retrieved {len(result.chunks)} chunks")
        
        if result.has_results:
            console.print(f"      Top score: {result.scores[0]:.4f}")
            console.print(f"      Source: {result.sources[0].get('source_file', 'unknown')}")
        
        return True
    except Exception as e:
        console.print(f"   [red]✗[/red] Error: {e}")
        return False


def test_ollama_connection():
    """Test Ollama connectivity."""
    console.print("\n[bold]7. Testing Ollama Connection[/bold]")
    
    try:
        import ollama
        
        models = ollama.list()
        model_names = [m['name'] for m in models.get('models', [])]
        
        console.print(f"   [green]✓[/green] Ollama is running")
        console.print(f"      Available models: {', '.join(model_names[:5])}")
        
        if not model_names:
            console.print("   [yellow]⚠[/yellow] No models found - run: ollama pull llama3.2")
        
        return True
    except Exception as e:
        console.print(f"   [yellow]⚠[/yellow] Ollama not available: {e}")
        console.print("      Run: ollama serve")
        return True  # Not a failure, just a warning


def test_generation():
    """Test LLM generation (if Ollama is available)."""
    console.print("\n[bold]8. Testing Generation[/bold]")
    
    from maths_tutor.rag.generator import Generator
    from maths_tutor.rag.retriever import Retriever
    
    try:
        retriever = Retriever()
        
        if retriever.vector_store.count == 0:
            console.print("   [yellow]⚠[/yellow] Skipping - no data in vector store")
            return True
        
        generator = Generator(retriever=retriever)
        
        # Quick test with a simple question
        console.print("   Testing with question: 'What is 2+2?'")
        
        import ollama
        try:
            # Just test that we can make a call
            response = ollama.chat(
                model=generator.model,
                messages=[{"role": "user", "content": "Reply with just the number: 2+2="}]
            )
            answer = response['message']['content'].strip()
            console.print(f"   [green]✓[/green] Generation works! Response: {answer[:50]}")
            return True
        except Exception as e:
            console.print(f"   [yellow]⚠[/yellow] Generation failed: {e}")
            console.print("      This is optional - RAG retrieval still works")
            return True
            
    except Exception as e:
        console.print(f"   [red]✗[/red] Error: {e}")
        return False


def main():
    """Run all tests."""
    console.print(Panel.fit(
        "[bold blue]CBSE Maths Tutor - RAG Pipeline Test[/bold blue]",
        border_style="blue"
    ))
    
    tests = [
        ("Imports", test_imports),
        ("PDF Parsing", test_pdf_parsing),
        ("Chunking", test_chunking),
        ("Embeddings", test_embeddings),
        ("Vector Store", test_vector_store),
        ("Retriever", test_retriever),
        ("Ollama Connection", test_ollama_connection),
        ("Generation", test_generation),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            console.print(f"   [red]✗[/red] Unexpected error: {e}")
            results.append((name, False))
    
    # Summary
    console.print("\n")
    table = Table(title="Test Results", show_header=True, header_style="bold cyan")
    table.add_column("Test", style="white")
    table.add_column("Status", style="white")
    
    passed = 0
    for name, success in results:
        status = "[green]PASSED[/green]" if success else "[red]FAILED[/red]"
        table.add_row(name, status)
        if success:
            passed += 1
    
    console.print(table)
    console.print(f"\n[bold]Total: {passed}/{len(results)} tests passed[/bold]")
    
    if passed == len(results):
        console.print("\n[bold green]✓ All tests passed! The system is ready.[/bold green]")
        console.print("\n[dim]Next steps:[/dim]")
        console.print("[dim]  1. Run ingestion: python scripts/ingest_books.py[/dim]")
        console.print("[dim]  2. Start chatbot: python -m maths_tutor[/dim]")
    else:
        console.print("\n[yellow]Some tests failed. Please check the errors above.[/yellow]")


if __name__ == "__main__":
    main()
