"""
Retriever - Finds relevant chunks for a given query.

This module handles the retrieval part of RAG:
1. Takes a user question
2. Converts it to an embedding
3. Searches the vector store for similar chunks
4. Filters and ranks results
5. Returns the most relevant context

Key Concept:
This is the "R" in RAG - Retrieval!
The quality of retrieval directly affects the quality of answers.
"""

from dataclasses import dataclass

from maths_tutor.config import MIN_SIMILARITY_SCORE, TOP_K_CHUNKS
from maths_tutor.embeddings.embedder import Embedder, get_embedder
from maths_tutor.embeddings.vector_store import VectorStore


@dataclass
class RetrievalResult:
    """
    Contains the retrieved context and metadata.

    Attributes:
        chunks: List of relevant text chunks
        scores: Similarity scores for each chunk
        sources: Source information (file, page) for each chunk
        query: The original query
    """

    chunks: list[str]
    scores: list[float]
    sources: list[dict]
    query: str

    @property
    def context(self) -> str:
        """
        Get the combined context as a single string.

        This is what gets sent to the LLM as context.
        """
        return "\n\n---\n\n".join(self.chunks)

    @property
    def has_results(self) -> bool:
        """Check if any results were found."""
        return len(self.chunks) > 0

    def format_for_prompt(self) -> str:
        """
        Format the context for inclusion in a prompt.

        Uses human-readable labels (subject + page) so the LLM can
        refer to sources naturally (e.g. "In the English textbook,
        page 15 ...") instead of opaque filenames or "Excerpt N".
        """
        if not self.has_results:
            return "No relevant information found in the textbook."

        formatted_parts = []
        for chunk, source in zip(self.chunks, self.sources, strict=False):
            subject = source.get("subject", "").title()
            book_title = source.get("book_title", "")
            page = source.get("page_number", "?")
            content_type = source.get("content_type", "")

            label_parts = [subject]
            if book_title:
                label_parts[0] = f"{subject} ({book_title})"
            label_parts.append(f"page {page}")
            if content_type and content_type != "chapter":
                label_parts.append(content_type.replace("_", " "))

            label = ", ".join(label_parts)
            formatted_parts.append(f"--- {label} ---\n{chunk}")

        return "\n\n".join(formatted_parts)


class Retriever:
    """
    Retrieves relevant context from the vector store.

    This class handles:
    - Embedding queries
    - Searching the vector store
    - Filtering results by score
    - Formatting context for the LLM

    Example:
        retriever = Retriever()
        result = retriever.retrieve("What is addition?")
        print(f"Found {len(result.chunks)} relevant chunks")
        print(result.context)
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        top_k: int | None = None,
        min_score: float | None = None,
    ):
        """
        Initialize the retriever.

        Args:
            embedder: Embedder instance (uses global if not provided)
            vector_store: VectorStore instance (creates new if not provided)
            top_k: Number of chunks to retrieve
            min_score: Minimum similarity score threshold
        """
        self.embedder = embedder or get_embedder()
        self.vector_store = vector_store or VectorStore()
        self.top_k = top_k or TOP_K_CHUNKS
        self.min_score = min_score or MIN_SIMILARITY_SCORE

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
        source_filter: str | None = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: The user's question
            top_k: Override default number of results
            min_score: Override minimum score threshold
            source_filter: Filter by source file name (optional)

        Returns:
            RetrievalResult with chunks, scores, and sources

        Example:
            result = retriever.retrieve(
                "How do you multiply fractions?",
                top_k=3,
                min_score=0.4
            )
        """
        top_k = top_k or self.top_k
        min_score = min_score or self.min_score

        # Check if database has documents
        if self.vector_store.count == 0:
            return RetrievalResult(chunks=[], scores=[], sources=[], query=query)

        # Embed the query
        query_embedding = self.embedder.embed(query)

        # Build filter if source_filter provided
        where_filter = None
        if source_filter:
            where_filter = {"source_file": source_filter}

        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=top_k, where=where_filter)

        # Filter by minimum score
        filtered_results = [r for r in results if r.score >= min_score]

        # Extract chunks, scores, and sources
        chunks = [r.text for r in filtered_results]
        scores = [r.score for r in filtered_results]
        sources = [r.metadata for r in filtered_results]

        return RetrievalResult(chunks=chunks, scores=scores, sources=sources, query=query)

    def retrieve_for_topic(self, topic: str, top_k: int | None = None) -> RetrievalResult:
        """
        Retrieve context for a specific topic (for quizzes/practice).

        This uses a broader search to get more context about a topic.

        Args:
            topic: The topic to retrieve context for
            top_k: Number of chunks to retrieve

        Returns:
            RetrievalResult with topic-related chunks
        """
        # Use a higher top_k for topic retrieval
        top_k = top_k or (self.top_k * 2)

        # Create a topic-focused query
        topic_query = f"Explain {topic}. What are the key concepts about {topic}?"

        return self.retrieve(
            query=topic_query,
            top_k=top_k,
            min_score=0.2,  # Lower threshold for broader coverage
        )

    def get_available_sources(self) -> list[str]:
        """
        Get list of available source files in the database.

        Returns:
            List of source file names
        """
        stats = self.vector_store.get_stats()
        return stats.get("unique_sources", [])

    def search_multiple(self, queries: list[str], top_k_per_query: int = 3) -> RetrievalResult:
        """
        Search with multiple queries and combine results.

        Useful when you want to get context from different angles.

        Args:
            queries: List of query strings
            top_k_per_query: Results per query

        Returns:
            Combined RetrievalResult (deduplicated)
        """
        seen_chunks = set()
        all_chunks = []
        all_scores = []
        all_sources = []

        for query in queries:
            result = self.retrieve(query, top_k=top_k_per_query)

            for chunk, score, source in zip(
                result.chunks, result.scores, result.sources, strict=False
            ):
                # Deduplicate by first 100 chars
                chunk_key = chunk[:100]
                if chunk_key not in seen_chunks:
                    seen_chunks.add(chunk_key)
                    all_chunks.append(chunk)
                    all_scores.append(score)
                    all_sources.append(source)

        # Sort by score and take top results
        combined = list(zip(all_chunks, all_scores, all_sources, strict=False))
        combined.sort(key=lambda x: x[1], reverse=True)
        combined = combined[: self.top_k]

        if combined:
            chunks, scores, sources = zip(*combined, strict=False)
            return RetrievalResult(
                chunks=list(chunks),
                scores=list(scores),
                sources=list(sources),
                query=" | ".join(queries),
            )

        return RetrievalResult(chunks=[], scores=[], sources=[], query=" | ".join(queries))


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def retrieve_context(query: str, top_k: int = 5) -> str:
    """
    Simple function to retrieve context for a query.

    Args:
        query: The user's question
        top_k: Number of chunks to retrieve

    Returns:
        Combined context string

    Example:
        context = retrieve_context("What is addition?")
        print(context)
    """
    retriever = Retriever()
    result = retriever.retrieve(query, top_k=top_k)
    return result.context


# =============================================================================
# MAIN - For testing
# =============================================================================

if __name__ == "__main__":
    """
    Test the retriever.
    Run: python -m maths_tutor.rag.retriever
    """
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    console.print(Panel.fit("[bold blue]RETRIEVER TEST[/bold blue]", border_style="blue"))

    retriever = Retriever()

    # Check if database has data
    if retriever.vector_store.count == 0:
        console.print("[red]Vector store is empty![/red]")
        console.print("[yellow]Run 'python scripts/ingest_books.py' first.[/yellow]")
        exit(1)

    console.print(f"[green]Database has {retriever.vector_store.count} chunks[/green]\n")

    # Test queries
    test_queries = [
        "What is addition?",
        "How do you multiply numbers?",
        "What are fractions?",
    ]

    for query in test_queries:
        console.print(f"\n[bold cyan]Query: {query}[/bold cyan]")
        console.print("-" * 50)

        result = retriever.retrieve(query, top_k=3)

        if result.has_results:
            for i, (chunk, score, source) in enumerate(
                zip(result.chunks, result.scores, result.sources, strict=False), 1
            ):
                console.print(f"\n[bold]Result {i}[/bold] (score: {score:.4f})")
                console.print(
                    f"Source: {source.get('source_file', '?')}, Page {source.get('page_number', '?')}"
                )
                preview = chunk[:200].replace("\n", " ")
                console.print(f"Preview: {preview}...")
        else:
            console.print("[yellow]No results found[/yellow]")

    console.print("\n[bold green]✓ Retriever test complete![/bold green]")
