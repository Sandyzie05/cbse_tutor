#!/usr/bin/env python3
"""
Ingestion Script - Process all PDF files and store in vector database.

This script orchestrates the full ingestion pipeline:
1. Discover all subject subdirectories under cbse-books/
2. Parse all PDF files from every subject
3. Split text into chunks
4. Generate embeddings for each chunk
5. Store in ChromaDB with subject metadata

Run this script ONCE to populate the vector database:
    python scripts/ingest_books.py

The script is idempotent - running it again will clear and rebuild the database.
"""

import re
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.table import Table

from maths_tutor.config import BOOKS_DIR, CHUNK_SIZE, CHUNK_OVERLAP, COLLECTION_NAME
from maths_tutor.ingestion.pdf_parser import PDFParser
from maths_tutor.ingestion.chunker import TextChunker
from maths_tutor.embeddings.embedder import Embedder
from maths_tutor.embeddings.vector_store import VectorStore

# Rich console for beautiful output
console = Console()


def _extract_subject_from_dirname(dirname: str) -> str:
    """
    Extract a human-readable subject name from a directory name.

    Examples:
        'cbse-grade-5-maths' -> 'maths'
        'cbse-grade-5-theWorldAroundUs' -> 'the world around us'
        'cbse-grade-5-physicalEducationAndWellbeing' -> 'physical education and wellbeing'
    """
    # Strip the 'cbse-grade-<N>-' prefix
    match = re.match(r"cbse-grade-\d+-(.+)", dirname)
    if not match:
        return dirname
    raw = match.group(1)
    # Split camelCase into words
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", raw).lower()
    return words


def discover_subjects(books_dir: Path) -> list[dict]:
    """
    Discover all subject directories and their PDF files.

    Returns:
        List of dicts with keys: subject, directory, pdf_files
    """
    subjects = []
    if not books_dir.exists():
        return subjects

    for subdir in sorted(books_dir.iterdir()):
        if not subdir.is_dir():
            continue
        pdf_files = sorted(subdir.glob("*.pdf"))
        if pdf_files:
            subjects.append({
                "subject": _extract_subject_from_dirname(subdir.name),
                "directory": subdir,
                "pdf_files": pdf_files,
            })
    return subjects


def ingest_books(clear_existing: bool = True) -> dict:
    """
    Main ingestion function that processes all PDFs across all subjects.
    
    Args:
        clear_existing: If True, clear existing data before ingesting
        
    Returns:
        Dictionary with ingestion statistics
    """
    stats = {
        "pdfs_processed": 0,
        "total_pages": 0,
        "total_chunks": 0,
        "subjects_processed": 0,
        "subjects": [],
        "errors": []
    }
    
    # Print header
    console.print(Panel.fit(
        "[bold blue]CBSE Grade 5 - All Subjects Ingestion[/bold blue]\n"
        "Processing PDF files from all subjects and storing in vector database",
        border_style="blue"
    ))
    
    # Check books directory
    if not BOOKS_DIR.exists():
        console.print(f"[red]Error: Books directory not found: {BOOKS_DIR}[/red]")
        return stats
    
    # Discover subjects
    subjects = discover_subjects(BOOKS_DIR)
    if not subjects:
        console.print(f"[red]Error: No subject directories with PDFs found in {BOOKS_DIR}[/red]")
        return stats
    
    # Show discovered subjects
    subject_table = Table(title="Discovered Subjects", show_header=True, header_style="bold cyan")
    subject_table.add_column("Subject", style="white")
    subject_table.add_column("PDFs", style="green", justify="right")
    subject_table.add_column("Directory", style="dim")
    
    total_pdfs = 0
    for subj in subjects:
        subject_table.add_row(
            subj["subject"].title(),
            str(len(subj["pdf_files"])),
            subj["directory"].name
        )
        total_pdfs += len(subj["pdf_files"])
    
    console.print(subject_table)
    console.print(f"\n[cyan]Total: {len(subjects)} subjects, {total_pdfs} PDF files[/cyan]")
    console.print(f"[cyan]Chunk size: {CHUNK_SIZE} chars, Overlap: {CHUNK_OVERLAP} chars[/cyan]\n")
    
    # Initialize components
    console.print("[yellow]Initializing components...[/yellow]")
    
    parser = PDFParser()
    chunker = TextChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    embedder = Embedder()
    vector_store = VectorStore()
    
    # Clear existing data if requested
    if clear_existing and vector_store.count > 0:
        console.print(f"[yellow]Clearing existing data ({vector_store.count} documents)...[/yellow]")
        vector_store.delete_collection()
        vector_store = VectorStore()  # Recreate
    
    # Process each subject
    all_chunks = []
    all_metadatas = []
    
    for subj in subjects:
        subject_name = subj["subject"]
        pdf_files = subj["pdf_files"]
        console.print(f"\n[bold magenta]--- {subject_name.title()} ({len(pdf_files)} PDFs) ---[/bold magenta]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            pdf_task = progress.add_task(f"[cyan]Processing {subject_name}...", total=len(pdf_files))
            
            for pdf_path in pdf_files:
                progress.update(pdf_task, description=f"[cyan]Parsing {pdf_path.name}...")
                
                try:
                    # Parse PDF
                    doc_content = parser.parse_pdf(pdf_path)
                    stats["pdfs_processed"] += 1
                    stats["total_pages"] += doc_content.total_pages
                    
                    # Create chunks with page and subject information
                    for page in doc_content.pages:
                        if not page.text.strip():
                            continue
                        
                        page_chunks = chunker.chunk_text(
                            page.text,
                            metadata={
                                "source_file": pdf_path.name,
                                "page_number": page.page_number,
                            }
                        )
                        
                        for chunk in page_chunks:
                            all_chunks.append(chunk.text)
                            all_metadatas.append({
                                "source_file": pdf_path.name,
                                "page_number": page.page_number,
                                "chunk_index": chunk.chunk_index,
                                "char_count": chunk.char_count,
                                "subject": subject_name,
                            })
                    
                except Exception as e:
                    stats["errors"].append(f"{pdf_path.name}: {str(e)}")
                    console.print(f"[red]Error processing {pdf_path.name}: {e}[/red]")
                
                progress.advance(pdf_task)
        
        stats["subjects_processed"] += 1
        stats["subjects"].append(subject_name)
    
    stats["total_chunks"] = len(all_chunks)
    
    if not all_chunks:
        console.print("[red]No text chunks extracted![/red]")
        return stats
    
    console.print(f"\n[green]Extracted {stats['total_chunks']} chunks from {stats['pdfs_processed']} PDFs[/green]")
    
    # Generate embeddings
    console.print("\n[yellow]Generating embeddings (this may take a few minutes)...[/yellow]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        
        embed_task = progress.add_task(
            "[cyan]Embedding chunks...", 
            total=len(all_chunks)
        )
        
        # Process in batches to show progress
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            batch_embeddings = embedder.embed_batch(batch, show_progress=False)
            all_embeddings.extend(batch_embeddings)
            progress.advance(embed_task, len(batch))
    
    # Store in vector database
    console.print("\n[yellow]Storing in vector database...[/yellow]")
    
    # Generate unique IDs
    ids = [f"chunk_{i:06d}" for i in range(len(all_chunks))]
    
    # Add to vector store
    vector_store.add_documents(
        texts=all_chunks,
        embeddings=all_embeddings,
        metadatas=all_metadatas,
        ids=ids
    )
    
    # Print summary
    console.print("\n")
    table = Table(title="Ingestion Summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="white")
    table.add_column("Value", style="green")
    
    table.add_row("Subjects Processed", str(stats["subjects_processed"]))
    table.add_row("Subjects", ", ".join(s.title() for s in stats["subjects"]))
    table.add_row("PDFs Processed", str(stats["pdfs_processed"]))
    table.add_row("Total Pages", str(stats["total_pages"]))
    table.add_row("Total Chunks", str(stats["total_chunks"]))
    table.add_row("Chunks in Database", str(vector_store.count))
    table.add_row("Errors", str(len(stats["errors"])))
    
    console.print(table)
    
    if stats["errors"]:
        console.print("\n[red]Errors encountered:[/red]")
        for error in stats["errors"]:
            console.print(f"  [red]• {error}[/red]")
    
    console.print("\n[bold green]✓ Ingestion complete![/bold green]")
    console.print("[dim]You can now use the chatbot to ask questions about the textbook.[/dim]")
    
    return stats


def verify_ingestion() -> None:
    """
    Verify that ingestion was successful by running a test query.
    """
    console.print("\n[yellow]Verifying ingestion with test query...[/yellow]")
    
    vector_store = VectorStore()
    embedder = Embedder()
    
    if vector_store.count == 0:
        console.print("[red]Vector store is empty! Run ingestion first.[/red]")
        return
    
    # Test query
    test_query = "What topics are covered in this textbook?"
    query_embedding = embedder.embed(test_query)
    
    results = vector_store.search(query_embedding, top_k=3)
    
    console.print(f"\n[cyan]Test query: '{test_query}'[/cyan]")
    console.print(f"[cyan]Found {len(results)} results:[/cyan]\n")
    
    for i, result in enumerate(results, 1):
        subject = result.metadata.get("subject", "unknown")
        console.print(f"[bold]Result {i}[/bold] (score: {result.score:.4f})")
        console.print(f"  Subject: {subject.title()}, Source: {result.source_file}, Page {result.page_number}")
        preview = result.text[:200].replace('\n', ' ')
        console.print(f"  Preview: {preview}...")
        console.print()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ingest CBSE Grade 5 book PDFs (all subjects) into vector database"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing ingestion, don't process"
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Don't clear existing data before ingesting"
    )
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_ingestion()
    else:
        stats = ingest_books(clear_existing=not args.no_clear)
        verify_ingestion()


if __name__ == "__main__":
    main()
