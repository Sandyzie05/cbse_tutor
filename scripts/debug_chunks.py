#!/usr/bin/env python3
"""
Debug script to inspect chunks stored in the vector database.

This helps diagnose issues with:
- What text was actually extracted from PDFs
- What chunks are stored for each source file
- Why certain queries might not find expected content
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt

from maths_tutor.embeddings.vector_store import VectorStore
from maths_tutor.embeddings.embedder import Embedder

console = Console()


def show_stats():
    """Show overall database statistics."""
    store = VectorStore()
    stats = store.get_stats()
    
    table = Table(title="Database Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Chunks", str(stats["document_count"]))
    table.add_row("Unique Sources", str(len(stats["unique_sources"])))
    
    console.print(table)
    
    console.print("\n[bold]Source Files:[/bold]")
    for source in sorted(stats["unique_sources"]):
        console.print(f"  • {source}")
    
    return stats["unique_sources"]


def show_chunks_by_source(source_file: str, limit: int = 10):
    """Show chunks from a specific source file."""
    store = VectorStore()
    
    # Get all chunks and filter by source
    # ChromaDB doesn't have a direct "get by metadata" without query
    # So we'll use a workaround
    
    console.print(f"\n[bold cyan]Chunks from {source_file}:[/bold cyan]")
    console.print("-" * 60)
    
    # Get all documents (up to a limit)
    result = store._collection.get(
        where={"source_file": source_file},
        include=["documents", "metadatas"],
        limit=limit
    )
    
    if not result["documents"]:
        console.print(f"[yellow]No chunks found for {source_file}[/yellow]")
        return
    
    console.print(f"[green]Found {len(result['documents'])} chunks (showing up to {limit})[/green]\n")
    
    for i, (doc, meta) in enumerate(zip(result["documents"], result["metadatas"])):
        page = meta.get("page_number", "?")
        console.print(f"[bold]Chunk {i+1}[/bold] (Page {page}):")
        console.print(f"[dim]{doc[:500]}...[/dim]" if len(doc) > 500 else f"[dim]{doc}[/dim]")
        console.print()


def search_in_chunks(query: str, source_filter: str = None):
    """Search for specific text in chunks."""
    store = VectorStore()
    embedder = Embedder()
    
    console.print(f"\n[bold cyan]Searching for: '{query}'[/bold cyan]")
    
    query_embedding = embedder.embed(query)
    
    where_filter = {"source_file": source_filter} if source_filter else None
    results = store.search(query_embedding, top_k=10, where=where_filter)
    
    if not results:
        console.print("[yellow]No results found[/yellow]")
        return
    
    console.print(f"[green]Found {len(results)} results:[/green]\n")
    
    for i, result in enumerate(results, 1):
        console.print(f"[bold]Result {i}[/bold] (score: {result.score:.4f})")
        console.print(f"  Source: {result.source_file}, Page {result.page_number}")
        console.print(f"  [dim]{result.text[:300]}...[/dim]")
        console.print()


def search_text_directly(search_term: str):
    """Search for exact text in chunks (not semantic)."""
    store = VectorStore()
    
    console.print(f"\n[bold cyan]Text search for: '{search_term}'[/bold cyan]")
    
    # Get all documents
    all_docs = store._collection.get(
        include=["documents", "metadatas"],
        limit=500  # Get more to search through
    )
    
    matches = []
    search_lower = search_term.lower()
    
    for doc, meta in zip(all_docs["documents"], all_docs["metadatas"]):
        if search_lower in doc.lower():
            matches.append((doc, meta))
    
    if not matches:
        console.print(f"[yellow]No chunks contain '{search_term}'[/yellow]")
        return
    
    console.print(f"[green]Found {len(matches)} chunks containing '{search_term}':[/green]\n")
    
    for i, (doc, meta) in enumerate(matches[:10], 1):
        source = meta.get("source_file", "unknown")
        page = meta.get("page_number", "?")
        console.print(f"[bold]Match {i}[/bold] ({source}, Page {page}):")
        
        # Highlight the search term in context
        idx = doc.lower().find(search_lower)
        start = max(0, idx - 100)
        end = min(len(doc), idx + len(search_term) + 100)
        snippet = doc[start:end]
        console.print(f"  ...{snippet}...")
        console.print()


def main():
    console.print(Panel.fit(
        "[bold blue]ChromaDB Debug Tool[/bold blue]\n"
        "Inspect chunks stored in the vector database",
        border_style="blue"
    ))
    
    sources = show_stats()
    
    console.print("\n[bold]Options:[/bold]")
    console.print("  1. View chunks from a specific PDF")
    console.print("  2. Semantic search (like the chatbot)")
    console.print("  3. Text search (exact match)")
    console.print("  4. Exit")
    
    while True:
        choice = Prompt.ask("\n[cyan]Choose option[/cyan]", choices=["1", "2", "3", "4"])
        
        if choice == "1":
            console.print(f"\nAvailable sources: {', '.join(sorted(sources))}")
            source = Prompt.ask("[cyan]Enter source file name[/cyan]")
            limit = int(Prompt.ask("[cyan]How many chunks to show?[/cyan]", default="10"))
            show_chunks_by_source(source, limit)
            
        elif choice == "2":
            query = Prompt.ask("[cyan]Enter search query[/cyan]")
            source = Prompt.ask("[cyan]Filter by source? (leave empty for all)[/cyan]", default="")
            search_in_chunks(query, source if source else None)
            
        elif choice == "3":
            term = Prompt.ask("[cyan]Enter text to search for[/cyan]")
            search_text_directly(term)
            
        elif choice == "4":
            break


if __name__ == "__main__":
    main()
