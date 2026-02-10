"""
Generator - Generates responses using Ollama LLM.

This module handles the generation part of RAG:
1. Takes retrieved context and user question
2. Builds a prompt with the context
3. Sends to Ollama for generation
4. Returns the response

Key Concept:
This is the "AG" in RAG - Augmented Generation!
We "augment" the LLM's knowledge with our retrieved context.
"""

import ollama
from typing import Optional, Generator as TypeGenerator

from maths_tutor.config import (
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    SYSTEM_PROMPT,
    RAG_PROMPT_TEMPLATE,
    QUIZ_PROMPT_TEMPLATE,
    PRACTICE_PROMPT_TEMPLATE
)
from maths_tutor.rag.retriever import Retriever, RetrievalResult


class Generator:
    """
    Generates responses using Ollama LLM with RAG context.
    
    This class handles:
    - Building prompts with context
    - Calling Ollama API
    - Streaming responses
    - Different generation modes (Q&A, quiz, practice)
    
    Example:
        generator = Generator()
        
        # Simple question answering
        answer = generator.generate(
            question="What is addition?",
            context="Addition is combining two numbers..."
        )
        print(answer)
    """
    
    def __init__(
        self,
        model: str | None = None,
        retriever: Retriever | None = None
    ):
        """
        Initialize the generator.
        
        Args:
            model: Ollama model name (uses config default if not provided)
            retriever: Retriever instance for getting context
        """
        self.model = model or OLLAMA_MODEL
        self.retriever = retriever or Retriever()
        self._check_ollama()
    
    def _check_ollama(self) -> bool:
        """
        Check if Ollama is running and the model is available.
        
        Returns:
            True if Ollama is ready, False otherwise
        """
        try:
            # List available models - handle different API versions
            response = ollama.list()
            
            # Handle different response formats (dict or object)
            if hasattr(response, 'models'):
                models_list = response.models
            elif isinstance(response, dict):
                models_list = response.get('models', [])
            else:
                models_list = []
            
            # Extract model names safely
            available_models = []
            for m in models_list:
                # Handle both dict and object responses
                if hasattr(m, 'model'):
                    name = m.model
                elif hasattr(m, 'name'):
                    name = m.name
                elif isinstance(m, dict):
                    name = m.get('name') or m.get('model', '')
                else:
                    continue
                if name:
                    available_models.append(name.split(':')[0])
            
            if available_models and self.model not in available_models:
                print(f"Warning: Model '{self.model}' not found. Available: {available_models}")
                print(f"Run: ollama pull {self.model}")
                return False
            
            return True
        except Exception as e:
            print(f"Warning: Cannot connect to Ollama: {e}")
            print("Make sure Ollama is running: ollama serve")
            return False
    
    def generate(
        self,
        question: str,
        context: str | None = None,
        system_prompt: str | None = None,
        stream: bool = False
    ) -> str | TypeGenerator[str, None, None]:
        """
        Generate a response using the LLM.
        
        Args:
            question: The user's question
            context: Retrieved context (if None, retrieves automatically)
            system_prompt: Override default system prompt
            stream: If True, yields response chunks for streaming
            
        Returns:
            Generated response string (or generator if streaming)
            
        Example:
            answer = generator.generate("What is 5 + 3?")
            print(answer)
        """
        # Get context if not provided
        if context is None:
            retrieval = self.retriever.retrieve(question)
            context = retrieval.format_for_prompt()
        
        # Build the prompt
        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )
        
        # Use default system prompt if not provided
        system = system_prompt or SYSTEM_PROMPT
        
        try:
            if stream:
                return self._stream_response(prompt, system)
            else:
                return self._generate_response(prompt, system)
        except Exception as e:
            return f"Error generating response: {e}\n\nMake sure Ollama is running and the model is available."
    
    def _generate_response(self, prompt: str, system: str) -> str:
        """Generate a complete response (non-streaming)."""
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ]
        )
        return response['message']['content']
    
    def _stream_response(self, prompt: str, system: str) -> TypeGenerator[str, None, None]:
        """Generate response with streaming."""
        stream = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
        
        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                yield chunk['message']['content']
    
    def ask(self, question: str, stream: bool = True) -> str | TypeGenerator[str, None, None]:
        """
        Ask a question about the textbook.
        
        This is the main method for Q&A.
        
        Args:
            question: The user's question
            stream: Whether to stream the response
            
        Returns:
            Answer string or generator for streaming
        """
        return self.generate(question, stream=stream)
    
    def generate_quiz(
        self,
        topic: str,
        num_questions: int = 5
    ) -> str:
        """
        Generate a quiz about a topic.
        
        Args:
            topic: The topic to quiz on
            num_questions: Number of questions to generate
            
        Returns:
            Quiz with multiple choice questions
        """
        # Get context for the topic
        retrieval = self.retriever.retrieve_for_topic(topic)
        context = retrieval.format_for_prompt()
        
        # Build quiz prompt
        prompt = QUIZ_PROMPT_TEMPLATE.format(
            context=context,
            num_questions=num_questions
        )
        
        return self._generate_response(prompt, SYSTEM_PROMPT)
    
    def generate_practice(
        self,
        topic: str,
        num_problems: int = 3
    ) -> str:
        """
        Generate practice problems for a topic.
        
        Args:
            topic: The topic to practice
            num_problems: Number of problems to generate
            
        Returns:
            Practice problems with solutions
        """
        # Get context for the topic
        retrieval = self.retriever.retrieve_for_topic(topic)
        context = retrieval.format_for_prompt()
        
        # Build practice prompt
        prompt = PRACTICE_PROMPT_TEMPLATE.format(
            context=context,
            num_problems=num_problems
        )
        
        return self._generate_response(prompt, SYSTEM_PROMPT)
    
    def explain(self, concept: str, simple: bool = True) -> str:
        """
        Explain a concept from the CBSE curriculum.
        
        Args:
            concept: The concept to explain
            simple: If True, use simpler language
            
        Returns:
            Explanation of the concept
        """
        # Get context
        retrieval = self.retriever.retrieve(f"Explain {concept}")
        context = retrieval.format_for_prompt()
        
        # Build explanation prompt
        complexity = "very simple terms suitable for a 10-year-old" if simple else "clear terms"
        prompt = f"""Context from CBSE Grade 5 textbooks:
---
{context}
---

Please explain {concept} in {complexity}.
Include:
1. A simple definition
2. A real-world example
3. Why it's useful

Keep your explanation friendly and encouraging!"""
        
        return self._generate_response(prompt, SYSTEM_PROMPT)


class RAGPipeline:
    """
    Complete RAG pipeline combining retrieval and generation.
    
    This is a high-level class that orchestrates the full RAG process.
    
    Example:
        rag = RAGPipeline()
        answer = rag.query("What is addition?")
        print(answer)
    """
    
    def __init__(
        self,
        model: str | None = None,
        top_k: int | None = None
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            model: Ollama model to use
            top_k: Number of chunks to retrieve
        """
        self.retriever = Retriever(top_k=top_k)
        self.generator = Generator(model=model, retriever=self.retriever)
    
    def query(
        self,
        question: str,
        stream: bool = False,
        include_sources: bool = False
    ) -> dict | str:
        """
        Process a question through the full RAG pipeline.
        
        Args:
            question: User's question
            stream: Whether to stream the response
            include_sources: Include source information in result
            
        Returns:
            Answer string or dict with answer and sources
        """
        # Retrieve context
        retrieval = self.retriever.retrieve(question)
        
        # Generate response
        if stream:
            response = self.generator.generate(
                question=question,
                context=retrieval.format_for_prompt(),
                stream=True
            )
        else:
            response = self.generator.generate(
                question=question,
                context=retrieval.format_for_prompt(),
                stream=False
            )
        
        if include_sources:
            return {
                "answer": response,
                "sources": retrieval.sources,
                "scores": retrieval.scores,
                "num_chunks": len(retrieval.chunks)
            }
        
        return response
    
    def quiz(self, topic: str, num_questions: int = 5) -> str:
        """Generate a quiz on a topic."""
        return self.generator.generate_quiz(topic, num_questions)
    
    def practice(self, topic: str, num_problems: int = 3) -> str:
        """Generate practice problems on a topic."""
        return self.generator.generate_practice(topic, num_problems)
    
    def explain(self, concept: str) -> str:
        """Explain a concept."""
        return self.generator.explain(concept)


# =============================================================================
# MAIN - For testing
# =============================================================================

if __name__ == "__main__":
    """
    Test the generator.
    Run: python -m maths_tutor.rag.generator
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    
    console = Console()
    
    console.print(Panel.fit(
        "[bold blue]GENERATOR TEST[/bold blue]",
        border_style="blue"
    ))
    
    # Initialize
    generator = Generator()
    
    # Check database
    if generator.retriever.vector_store.count == 0:
        console.print("[red]Vector store is empty![/red]")
        console.print("[yellow]Run 'python scripts/ingest_books.py' first.[/yellow]")
        exit(1)
    
    console.print(f"[green]Using model: {generator.model}[/green]")
    console.print(f"[green]Database has {generator.retriever.vector_store.count} chunks[/green]\n")
    
    # Test question
    console.print("[bold cyan]Test Question: What is addition?[/bold cyan]")
    console.print("-" * 50)
    
    try:
        # Non-streaming test
        answer = generator.ask("What is addition?", stream=False)
        console.print(Markdown(answer))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[yellow]Make sure Ollama is running: ollama serve[/yellow]")
        console.print(f"[yellow]And pull the model: ollama pull {generator.model}[/yellow]")
    
    console.print("\n[bold green]✓ Generator test complete![/bold green]")
