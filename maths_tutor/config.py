"""
Configuration settings for the Maths Tutor application.

This file centralizes all configuration so you can easily adjust parameters.
Understanding these settings is important for tuning your RAG system!
"""

from pathlib import Path

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Base directory (where this project lives)
BASE_DIR = Path(__file__).parent.parent

# PDF source directory
PDF_DIR = BASE_DIR / "cbse-grade-5-maths"

# Data storage directory
DATA_DIR = BASE_DIR / "data"

# ChromaDB storage location
CHROMA_DB_DIR = DATA_DIR / "chroma_db"

# =============================================================================
# CHUNKING CONFIGURATION
# =============================================================================

# Chunk size in characters
# WHY 800? It's a balance between:
#   - Too small (200-300): Loses context, retrieves incomplete information
#   - Too large (1500+): Less precise retrieval, may exceed LLM context limits
CHUNK_SIZE = 800

# Overlap between chunks in characters
# WHY 100? Ensures we don't lose information at chunk boundaries
# Example: If a sentence spans two chunks, the overlap captures it fully
CHUNK_OVERLAP = 100

# =============================================================================
# EMBEDDING CONFIGURATION
# =============================================================================

# Embedding model name (same as Week 1!)
# This model creates 384-dimensional vectors
# Good balance between speed and quality
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Embedding dimension (for reference)
EMBEDDING_DIMENSION = 384

# =============================================================================
# CHROMADB CONFIGURATION
# =============================================================================

# Collection name for storing book embeddings
COLLECTION_NAME = "cbse_grade5_maths"

# =============================================================================
# OLLAMA CONFIGURATION
# =============================================================================

# Default Ollama model
# llama3.2 is recommended - good balance of speed and intelligence
# Other options: mistral, phi3, gemma2
OLLAMA_MODEL = "llama3.2"

# Ollama API base URL (default local installation)
OLLAMA_BASE_URL = "http://localhost:11434"

# =============================================================================
# RAG CONFIGURATION
# =============================================================================

# Number of chunks to retrieve for context
# WHY 8? More chunks = more context but slower and may confuse the LLM
# Fewer chunks = faster but might miss relevant information
# We use 8 to ensure we capture structural info (like table of contents)
TOP_K_CHUNKS = 8

# Minimum similarity score to include a chunk (0.0 to 1.0)
# Chunks below this threshold are considered irrelevant
MIN_SIMILARITY_SCORE = 0.3

# =============================================================================
# LLM PROMPT TEMPLATES
# =============================================================================

# System prompt for the maths tutor
SYSTEM_PROMPT = """You are a friendly and patient maths tutor for Class 5 students (ages 10-11).
Your job is to help students understand mathematical concepts from their CBSE textbook.

IMPORTANT RULES:
1. Use ONLY the provided context to answer questions
2. If the answer is not in the context, say "I don't have information about that in the textbook"
3. Explain concepts in simple, easy-to-understand language
4. Use examples when helpful
5. Be encouraging and supportive
6. For calculations, show step-by-step working

Remember: You're teaching young students, so be patient and clear!"""

# Template for RAG queries
RAG_PROMPT_TEMPLATE = """Context from the textbook:
---
{context}
---

Student's question: {question}

Please answer the question based on the context above. If the answer isn't in the context, let the student know."""

# Template for quiz generation
QUIZ_PROMPT_TEMPLATE = """Context from the textbook:
---
{context}
---

Based on the context above, create a quiz with {num_questions} multiple choice questions.
Each question should have 4 options (A, B, C, D) with one correct answer.
Format each question like this:

Q1: [Question text]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Correct Answer: [Letter]

Make the questions appropriate for Class 5 students (ages 10-11)."""

# Template for practice problems
PRACTICE_PROMPT_TEMPLATE = """Context from the textbook:
---
{context}
---

Based on the context above, create {num_problems} practice problems for a Class 5 student.
For each problem:
1. Write the problem clearly
2. Provide the solution with step-by-step working

Format:
Problem 1: [Problem text]
Solution:
Step 1: [First step]
Step 2: [Second step]
...
Answer: [Final answer]"""
