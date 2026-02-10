# CBSE AI Tutor - Architecture Guide

This document explains the architecture of the RAG-based chatbot system.
It's designed to help you understand how all the pieces fit together.

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [Data Flow](#data-flow)
3. [Component Details](#component-details)
4. [Web Interface](#web-interface)
5. [Key Concepts Explained](#key-concepts-explained)
6. [File-by-File Breakdown](#file-by-file-breakdown)

---

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CBSE AI TUTOR                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     USER INTERFACES                               │   │
│  │  ┌─────────────────┐              ┌─────────────────┐            │   │
│  │  │   CLI (cli.py)  │              │ Web (web_app.py)│            │   │
│  │  │   - Questions   │              │   - Chat UI     │            │   │
│  │  │   - Quizzes     │              │   - Quiz Mode   │            │   │
│  │  │   - Practice    │              │   - Progress    │            │   │
│  │  └────────┬────────┘              └────────┬────────┘            │   │
│  └───────────┼────────────────────────────────┼─────────────────────┘   │
│              │                                │                          │
│              └────────────────┬───────────────┘                          │
│                               │                                          │
│  ┌────────────────────────────▼─────────────────────────────────────┐   │
│  │                      RAG PIPELINE                                 │   │
│  │  ┌─────────────────┐              ┌─────────────────┐            │   │
│  │  │    Retriever    │──────────────│    Generator    │            │   │
│  │  │ (retriever.py)  │   context    │  (generator.py) │            │   │
│  │  │                 │──────────────│                 │            │   │
│  │  └────────┬────────┘              └────────┬────────┘            │   │
│  └───────────┼────────────────────────────────┼─────────────────────┘   │
│              │ query                          │ prompt                   │
│              │                                │                          │
│  ┌───────────▼──────────────┐    ┌───────────▼──────────────┐          │
│  │      VECTOR STORE        │    │         OLLAMA           │          │
│  │  ┌─────────────────┐     │    │  ┌─────────────────┐     │          │
│  │  │    ChromaDB     │     │    │  │    LLM Model    │     │          │
│  │  │ (vector_store.py)     │    │  │   (llama3.2)    │     │          │
│  │  └─────────────────┘     │    │  └─────────────────┘     │          │
│  └──────────────────────────┘    └──────────────────────────┘          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Phase 1: Ingestion (Run Once)

This happens when you run `python scripts/ingest_books.py`:

```
┌─────────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│                 │    │             │    │             │    │             │
│  PDF Files      │───▶│ PDF Parser  │───▶│  Chunker    │───▶│  Embedder   │
│  (5 subjects,   │    │  (pymupdf)  │    │ (800 chars) │    │(sentence-   │
│   ~60 PDFs)     │    │             │    │             │    │transformers)│
└─────────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘
                                                                    │
                                                                    │ embeddings
                                                                    │ (384-dim vectors)
                                                                    ▼
                                                             ┌─────────────┐
                                                             │             │
                                                             │  ChromaDB   │
                                                             │  (stored    │
                                                             │  w/ subject │
                                                             │  metadata)  │
                                                             └─────────────┘
```

**Step-by-step:**
1. **Subject Discovery** auto-discovers all subject folders under `cbse-books/`
2. **PDF Parser** reads each PDF and extracts text
3. **Chunker** splits text into ~800 character pieces (with 100 char overlap)
4. **Embedder** converts each chunk to a 384-dimensional vector
5. **ChromaDB** stores the vectors + original text + metadata (including subject)

### Phase 2: Query (Every Question)

This happens when you ask a question:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│             │    │             │    │             │    │             │
│  Question   │───▶│  Embedder   │───▶│ Vector      │───▶│  Top K      │
│  "What is   │    │(same model) │    │ Search      │    │  Chunks     │
│  a habitat?"│    │             │    │ (ChromaDB)  │    │  (context)  │
└─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘
                                                                │
                                                                │
                                                                ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│             │    │             │    │             │    │             │
│   Answer    │◀───│   Ollama    │◀───│   Prompt    │◀───│  Context +  │
│  "A habitat │    │   LLM       │    │  Template   │    │  Question   │
│   is..."    │    │             │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

**Step-by-step:**
1. **Embedder** converts your question to a vector
2. **Vector Search** finds the most similar chunks in ChromaDB (across all subjects)
3. **Prompt Builder** combines context + question into a prompt
4. **Ollama** generates a response using the context
5. **Answer** is displayed to the user

---

## Component Details

### 1. PDF Parser (`ingestion/pdf_parser.py`)

**Purpose:** Extract text from PDF files

**Input:** PDF file path  
**Output:** Raw text string

```python
# What it does (simplified):
import fitz  # pymupdf

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text
```

**Challenges:**
- Math symbols may not extract correctly
- Diagrams/images are lost (text-only)
- Headers/footers add noise

---

### 2. Chunker (`ingestion/chunker.py`)

**Purpose:** Split text into smaller pieces for embedding

**Input:** Long text string  
**Output:** List of text chunks

```
Original: "Chapter 1: Numbers. Numbers are everywhere in our daily life..."
                                         (2000 characters)

After chunking (size=800, overlap=100):

Chunk 1: "Chapter 1: Numbers. Numbers are everywhere..." (800 chars)
Chunk 2: "...everywhere in our daily life. We use numbers..." (800 chars)
Chunk 3: "...numbers to count things, measure..." (800 chars)
              ↑
              Overlap ensures continuity
```

**Why overlap?**
Without overlap, a sentence at the boundary might be cut in half:
```
Chunk 1: "The answer to 25 + 17"  ← Incomplete!
Chunk 2: "is 42. Next, we learn..."  ← Missing context!

With overlap:
Chunk 1: "The answer to 25 + 17 is 42."  ← Complete!
Chunk 2: "25 + 17 is 42. Next, we learn..."  ← Also has the answer!
```

---

### 3. Embedder (`embeddings/embedder.py`)

**Purpose:** Convert text to vectors (numbers)

**Input:** Text string  
**Output:** 384-dimensional vector

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode("What is a habitat?")
# Returns: [0.023, -0.156, 0.089, ...] (384 numbers)
```

**Why this works:**
- Similar text → Similar vectors
- "What is a habitat?" ≈ "Where do animals live?"
- This is the magic of semantic search!

---

### 4. Vector Store (`embeddings/vector_store.py`)

**Purpose:** Store and search embeddings

**Operations:**
- **Add:** Store chunk + embedding + metadata (including subject)
- **Query:** Find similar chunks for a question

```python
# Simplified example:
import chromadb

# Store a chunk
collection.add(
    documents=["Habitats are places where organisms live..."],
    embeddings=[[0.023, -0.156, ...]],
    metadatas=[{"page": 5, "subject": "the world around us"}],
    ids=["chunk_001"]
)

# Search for similar chunks
results = collection.query(
    query_embeddings=[[0.025, -0.150, ...]],  # Question embedding
    n_results=8  # Get top 8
)
```

---

### 5. Retriever (`rag/retriever.py`)

**Purpose:** Find relevant chunks for a question

**Process:**
1. Embed the question
2. Search ChromaDB for similar chunks
3. Filter by similarity score
4. Return top K chunks

```python
def retrieve(question: str) -> list[str]:
    # 1. Embed question
    query_embedding = embedder.encode(question)
    
    # 2. Search vector store
    results = vector_store.query(query_embedding, top_k=8)
    
    # 3. Filter low-quality matches
    relevant = [r for r in results if r.score > 0.3]
    
    # 4. Return chunks
    return [r.text for r in relevant]
```

---

### 6. Generator (`rag/generator.py`)

**Purpose:** Generate responses using Ollama

**Process:**
1. Build prompt with context + question
2. Send to Ollama
3. Return response

```python
def generate(question: str, context: list[str]) -> str:
    # Build prompt
    prompt = f"""
    Context from CBSE Grade 5 textbooks:
    {' '.join(context)}
    
    Question: {question}
    
    Answer based on the context above:
    """
    
    # Call Ollama
    response = ollama.generate(model="llama3.2", prompt=prompt)
    return response
```

---

## Web Interface

### Architecture (`interfaces/web_app.py`)

The web interface is built with **FastAPI** and serves both a browser-based chat UI
and a set of JSON REST endpoints that can be consumed by any client.

```
┌──────────────┐        ┌────────────────────────────────────────────┐
│   Browser    │──GET /─│  FastAPI (web_app.py)                      │
│   (HTML/JS)  │        │                                            │
│              │◀──HTML─│  Jinja2 renders templates/index.html       │
│              │        │                                            │
│              │──POST──│  /api/ask          → RAGPipeline.query()   │
│              │──POST──│  /api/ask/stream   → SSE token stream      │
│              │──POST──│  /api/quiz         → RAGPipeline.quiz()    │
│              │──POST──│  /api/practice     → RAGPipeline.practice()│
│              │──POST──│  /api/explain      → RAGPipeline.explain() │
│              │──GET───│  /api/stats        → VectorStore.get_stats │
│              │──GET───│  /health           → health check          │
└──────────────┘        └────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **App factory** (`create_app()`) | Allows tests to create isolated app instances with mocked RAG/VectorStore |
| **Lazy-loaded RAG singleton** | Heavy model loading (sentence-transformers, ChromaDB) happens on first request, not at import time |
| **SSE streaming** (`/api/ask/stream`) | Tokens appear in the browser as the LLM generates them, giving instant feedback |
| **Pydantic request models** | Input validation (min/max lengths, ranges) handled declaratively |
| **`asyncio.to_thread`** for non-streaming calls | Keeps the event loop free while Ollama blocks on generation |

### API Endpoints

| Method | Path | Body | Response |
|--------|------|------|----------|
| `GET` | `/` | – | HTML chat page |
| `GET` | `/health` | – | `{ status, chunks_loaded }` |
| `POST` | `/api/ask` | `{ question, stream? }` | `{ answer, sources, num_chunks }` |
| `POST` | `/api/ask/stream` | `{ question }` | SSE: `{ token }` … `{ done, sources }` |
| `POST` | `/api/quiz` | `{ topic, num_questions? }` | `{ quiz }` |
| `POST` | `/api/practice` | `{ topic, num_problems? }` | `{ practice }` |
| `POST` | `/api/explain` | `{ concept }` | `{ explanation }` |
| `GET` | `/api/stats` | – | `{ collection_name, document_count, unique_sources }` |

### Testing

Tests live in `tests/` and use **pytest** with FastAPI's `TestClient`. All external
dependencies (ChromaDB, sentence-transformers, Ollama) are replaced with lightweight
fakes defined in `tests/conftest.py`, so the full suite runs in ~5 seconds with zero
external services.

```bash
# Run tests
python -m pytest tests/ -v

# Lint (ruff)
ruff check maths_tutor/ tests/
```

---

## Key Concepts Explained

### What is a Vector?

A vector is just a list of numbers:
```
"cat" → [0.2, 0.8, -0.3, 0.5]
"dog" → [0.25, 0.75, -0.28, 0.48]  ← Similar to cat!
"car" → [-0.5, 0.1, 0.9, -0.2]     ← Different from cat
```

### What is Cosine Similarity?

Measures how similar two vectors are:
- 1.0 = Identical
- 0.5 = Somewhat similar
- 0.0 = Completely different

```
similarity(cat, dog) = 0.95  ← Very similar
similarity(cat, car) = 0.15  ← Not similar
```

### What is RAG?

**R**etrieval-**A**ugmented **G**eneration

1. **Retrieval:** Find relevant information
2. **Augmented:** Add it to the prompt
3. **Generation:** Let LLM use this context

**Why RAG?**
- LLMs don't know your textbook content
- RAG gives them the relevant context
- Results in accurate, grounded answers

### Token Limits

LLMs can only process a limited amount of text (tokens):
- llama3.2: ~8,000 tokens context
- This is why we chunk text into smaller pieces
- We only send the most relevant chunks

---

## File-by-File Breakdown

```
maths_tutor/                 # Python package (RAG engine)
│
├── __init__.py              # Package initialization
│
├── config.py                # All configuration in one place
│                            # - BOOKS_DIR (all subjects)
│                            # - Chunk size (800)
│                            # - Overlap (100)
│                            # - Model name
│                            # - Prompt templates
│
├── ingestion/
│   ├── pdf_parser.py        # Extracts text from PDFs
│   │                        # Uses: pymupdf (fitz)
│   │                        # Input: PDF file path
│   │                        # Output: Raw text
│   │
│   └── chunker.py           # Splits text into chunks
│                            # Input: Long text
│                            # Output: List of chunks
│
├── embeddings/
│   ├── embedder.py          # Converts text to vectors
│   │                        # Uses: sentence-transformers
│   │                        # Input: Text string
│   │                        # Output: 384-dim vector
│   │
│   └── vector_store.py      # Stores/searches vectors
│                            # Uses: ChromaDB
│                            # Operations: add, query
│
├── rag/
│   ├── retriever.py         # Finds relevant chunks
│   │                        # Input: Question
│   │                        # Output: Top 8 chunks
│   │
│   └── generator.py         # Generates responses
│                            # Uses: Ollama
│                            # Input: Question + Context
│                            # Output: Answer
│
└── interfaces/
    ├── cli.py               # Command-line interface
    │                        # Commands: ask, quiz, practice
    │
    ├── web_app.py           # FastAPI web interface
    │                        # REST API + SSE streaming
    │                        # Serves HTML chat UI
    │
    └── templates/
        └── index.html       # Chat UI (Jinja2 template)
                             # Vanilla JS, SSE client
```

---

## Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    THE BIG PICTURE                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. INGESTION (once):                                           │
│     Discover subjects → PDF → Text → Chunks → Embeddings        │
│     → ChromaDB (with subject metadata)                          │
│                                                                  │
│  2. QUERY (every question):                                     │
│     Question → Embedding → Search → Context → LLM → Answer      │
│                                                                  │
│  3. KEY INSIGHT:                                                │
│     Similar text = Similar embeddings                           │
│     This enables semantic search across ALL subjects!           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

This architecture is the foundation of most modern AI applications.
Understanding it will help you build chatbots, search engines, and more!
