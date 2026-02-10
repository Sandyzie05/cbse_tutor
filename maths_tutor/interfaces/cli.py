#!/usr/bin/env python3
"""
CLI Interface - Interactive command-line chatbot.

This module provides a beautiful terminal interface for interacting
with the CBSE Tutor. It supports:
- Free-form questions across all Grade 5 subjects
- Quiz generation (/quiz)
- Practice problems (/practice)
- Concept explanations (/explain)
- Help and commands (/help)

Run with:
    python -m maths_tutor.interfaces.cli
"""

import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table

from maths_tutor.rag.generator import RAGPipeline
from maths_tutor.embeddings.vector_store import VectorStore


# Rich console for beautiful output
console = Console()


def print_welcome():
    """Print welcome message and instructions."""
    welcome_text = """
[bold blue]Welcome to CBSE Tutor - Grade 5![/bold blue]

I'm here to help you learn all Class 5 subjects. You can ask about:

• [cyan]Maths[/cyan] - Numbers, fractions, geometry, and more
• [cyan]English[/cyan] - Grammar, comprehension, and vocabulary
• [cyan]Arts[/cyan] - Creative expression, culture, and art forms
• [cyan]The World Around Us[/cyan] - Science, environment, and society
• [cyan]Physical Education & Wellbeing[/cyan] - Health, fitness, and wellness

[dim]Type /help for all commands[/dim]
"""
    console.print(Panel(welcome_text, border_style="blue"))


def print_help():
    """Print help message with available commands."""
    table = Table(title="Available Commands", show_header=True, header_style="bold cyan")
    table.add_column("Command", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Example", style="dim")
    
    commands = [
        ("(any question)", "Ask about the textbook", "What is addition?"),
        ("/quiz <topic> [n]", "Generate n quiz questions", "/quiz fractions 5"),
        ("/practice <topic> [n]", "Generate n practice problems", "/practice multiplication 3"),
        ("/explain <concept>", "Explain a concept simply", "/explain place value"),
        ("/sources", "Show available sources", "/sources"),
        ("/stats", "Show database statistics", "/stats"),
        ("/clear", "Clear the screen", "/clear"),
        ("/help", "Show this help message", "/help"),
        ("/exit", "Exit the chatbot", "/exit"),
    ]
    
    for cmd, desc, example in commands:
        table.add_row(cmd, desc, example)
    
    console.print(table)


def parse_command(user_input: str) -> tuple[str, list[str]]:
    """
    Parse user input into command and arguments.
    
    Returns:
        Tuple of (command, arguments)
        For regular questions, command is 'ask'
    """
    user_input = user_input.strip()
    
    if not user_input:
        return ("empty", [])
    
    if user_input.startswith("/"):
        parts = user_input[1:].split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1].split() if len(parts) > 1 else []
        return (command, args)
    
    return ("ask", [user_input])


def stream_response(rag: RAGPipeline, question: str):
    """Stream a response with a nice display."""
    console.print("\n[bold green]🎓 CBSE Tutor:[/bold green]")
    
    try:
        # Get streaming response
        response_gen = rag.query(question, stream=True)
        
        # Collect the full response for markdown rendering
        full_response = ""
        
        with Live(console=console, refresh_per_second=10) as live:
            for chunk in response_gen:
                full_response += chunk
                # Show response as it streams
                live.update(Markdown(full_response))
        
        console.print()  # Add newline after response
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[yellow]Make sure Ollama is running: ollama serve[/yellow]")


def handle_quiz(rag: RAGPipeline, args: list[str]):
    """Handle /quiz command."""
    if not args:
        console.print("[yellow]Usage: /quiz <topic> [number_of_questions][/yellow]")
        console.print("[dim]Example: /quiz fractions 5[/dim]")
        return
    
    topic = args[0]
    num_questions = int(args[1]) if len(args) > 1 else 5
    
    console.print(f"\n[cyan]Generating {num_questions} quiz questions about {topic}...[/cyan]")
    
    with console.status("[bold green]Thinking...", spinner="dots"):
        quiz = rag.quiz(topic, num_questions)
    
    console.print("\n[bold green]📝 Quiz:[/bold green]")
    console.print(Markdown(quiz))


def handle_practice(rag: RAGPipeline, args: list[str]):
    """Handle /practice command."""
    if not args:
        console.print("[yellow]Usage: /practice <topic> [number_of_problems][/yellow]")
        console.print("[dim]Example: /practice addition 3[/dim]")
        return
    
    topic = args[0]
    num_problems = int(args[1]) if len(args) > 1 else 3
    
    console.print(f"\n[cyan]Generating {num_problems} practice problems about {topic}...[/cyan]")
    
    with console.status("[bold green]Thinking...", spinner="dots"):
        practice = rag.practice(topic, num_problems)
    
    console.print("\n[bold green]✏️ Practice Problems:[/bold green]")
    console.print(Markdown(practice))


def handle_explain(rag: RAGPipeline, args: list[str]):
    """Handle /explain command."""
    if not args:
        console.print("[yellow]Usage: /explain <concept>[/yellow]")
        console.print("[dim]Example: /explain fractions[/dim]")
        return
    
    concept = " ".join(args)
    
    console.print(f"\n[cyan]Explaining {concept}...[/cyan]")
    
    with console.status("[bold green]Thinking...", spinner="dots"):
        explanation = rag.explain(concept)
    
    console.print("\n[bold green]💡 Explanation:[/bold green]")
    console.print(Markdown(explanation))


def handle_sources(rag: RAGPipeline):
    """Handle /sources command."""
    sources = rag.retriever.get_available_sources()
    
    if sources:
        console.print("\n[bold]Available Sources:[/bold]")
        for source in sorted(sources):
            console.print(f"  • {source}")
    else:
        console.print("[yellow]No sources found. Run ingestion first.[/yellow]")


def handle_stats(rag: RAGPipeline):
    """Handle /stats command."""
    stats = rag.retriever.vector_store.get_stats()
    
    table = Table(title="Database Statistics", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="white")
    table.add_column("Value", style="green")
    
    table.add_row("Collection Name", stats["collection_name"])
    table.add_row("Total Chunks", str(stats["document_count"]))
    table.add_row("Unique Sources", str(len(stats["unique_sources"])))
    table.add_row("Storage Location", stats["persist_directory"])
    
    console.print(table)


def main():
    """Main CLI loop."""
    print_welcome()
    
    # Check if database has data
    vector_store = VectorStore()
    if vector_store.count == 0:
        console.print("[red]⚠️  Vector store is empty![/red]")
        console.print("[yellow]Please run the ingestion script first:[/yellow]")
        console.print("[cyan]  python scripts/ingest_books.py[/cyan]")
        console.print()
    else:
        console.print(f"[dim]📚 Loaded {vector_store.count} chunks from textbook[/dim]")
    
    # Initialize RAG pipeline
    try:
        rag = RAGPipeline()
    except Exception as e:
        console.print(f"[red]Error initializing: {e}[/red]")
        console.print("[yellow]Make sure Ollama is running.[/yellow]")
        return
    
    console.print()
    
    # Main loop
    while True:
        try:
            # Get user input
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
            
            # Parse command
            command, args = parse_command(user_input)
            
            # Handle commands
            if command == "empty":
                continue
            
            elif command == "exit" or command == "quit":
                console.print("\n[bold blue]Goodbye! Keep learning! 📚[/bold blue]")
                break
            
            elif command == "help":
                print_help()
            
            elif command == "clear":
                console.clear()
                print_welcome()
            
            elif command == "quiz":
                handle_quiz(rag, args)
            
            elif command == "practice":
                handle_practice(rag, args)
            
            elif command == "explain":
                handle_explain(rag, args)
            
            elif command == "sources":
                handle_sources(rag)
            
            elif command == "stats":
                handle_stats(rag)
            
            elif command == "ask":
                question = args[0] if args else ""
                if question:
                    stream_response(rag, question)
            
            else:
                # Unknown command - treat as question
                full_input = f"/{command} {' '.join(args)}".strip()
                console.print(f"[yellow]Unknown command. Treating as question...[/yellow]")
                stream_response(rag, full_input)
            
            console.print()  # Add spacing
            
        except KeyboardInterrupt:
            console.print("\n\n[bold blue]Goodbye! Keep learning! 📚[/bold blue]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            console.print("[dim]Type /help for available commands[/dim]")


if __name__ == "__main__":
    main()
