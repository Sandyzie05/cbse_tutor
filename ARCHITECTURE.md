# CBSE AI Tutor - Architecture Guide

This document explains the architecture of the RAG-based chatbot system.
It's designed to help you understand how all the pieces fit together.

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [Data Flow](#data-flow)
3. [Per-Book Agent Design](#per-book-agent-design)
4. [Component Details](#component-details)
5. [Web Interface](#web-interface)
6. [Key Concepts Explained](#key-concepts-explained)
7. [File-by-File Breakdown](#file-by-file-breakdown)

---

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CBSE AI TUTOR                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     USER INTERFACES                               │   │
│  │  ┌─────────────────────────────────────────────────────────┐     │   │
│  │  │          Web (web_app.py) - Two Views                   │     │   │
│  │  │  View 1: Book Selection Grid                            │     │   │
│  │  │  View 2: Per-Book Chat (Ask/Quiz/Practice/Explain)      │     │   │
│  │  └─────────────────┬───────────────────────────────────────┘     │   │
│  │                    │                                              │   │
│  │  ┌─────────────────┐                                             │   │
│  │  │   CLI (cli.py)  │                                             │   │
│  │  └────────┬────────┘                                             │   │
│  └───────────┼────────┼─────────────────────────────────────────────┘   │
│              │        │                                                  │
│              └────┬───┘                                                  │
│                   │ (book_id selects which agent)                        │
│                   │                                                      │
│  ┌────────────────▼──────────────────────────────────────────────────┐  │
│  │              PER-BOOK RAG PIPELINES (cached)                      │  │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   ...        │  │
│  │  │ Maths Agent  │ │English Agent │ │  Arts Agent  │              │  │
│  │  │ (prompt +    │ │ (prompt +    │ │ (prompt +    │              │  │
│  │  │  retriever + │ │  retriever + │ │  retriever + │              │  │
│  │  │  generator)  │ │  generator)  │ │  generator)  │              │  │
│  │  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘              │  │
│  └─────────┼────────────────┼────────────────┼───────────────────────┘  │
│            │                │                │                           │
│  ┌─────────▼──────┐ ┌──────▼───────┐ ┌──────▼───────┐                  │
│  │  ChromaDB      │ │  ChromaDB    │ │  ChromaDB    │   ...            │
│  │  maths         │ │  english     │ │  arts        │                  │
│  └────────────────┘ └──────────────┘ └──────────────┘                  │
│                                                                          │
│  ┌──────────────────────────────────┐                                   │
│  │            OLLAMA                │                                   │
│  │    LLM Model (llama3.2)         │                                   │
│  └──────────────────────────────────┘                                   │
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
│  PDF Files      │───>│ PDF Parser  │───>│  Chunker    │───>│  Embedder   │
│  (per subject)  │    │  (pymupdf)  │    │ (800 chars) │    │(sentence-   │
│                 │    │             │    │             │    │transformers)│
└─────────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘
                                                                    │
                                                                    ▼
                                                   ┌─────────────────────────┐
                                                   │  Per-Subject ChromaDB   │
                                                   │  Collections            │
                                                   │                         │
                                                   │  cbse_grade5_maths      │
                                                   │  cbse_grade5_english    │
                                                   │  cbse_grade5_arts       │
                                                   │  cbse_grade5_the_...    │
                                                   │  cbse_grade5_phys...    │
                                                   └─────────────────────────┘
                                                                    +
                                                   ┌─────────────────────────┐
                                                   │  data/books.json        │
                                                   │  (manifest with book    │
                                                   │   metadata + chapters)  │
                                                   └─────────────────────────┘
```

**Step-by-step:**
1. **Subject Discovery** auto-discovers all subject folders under `cbse-books/`
2. **Chapter Extraction** parses the `*ps.pdf` (preface) to get the chapter/unit list
3. **PDF Parser** reads each PDF and extracts text
4. **Chunker** splits text into ~800 character pieces (with 100 char overlap)
5. **Embedder** converts each chunk to a 384-dimensional vector
6. **ChromaDB** stores vectors in a **per-subject collection**
7. **books.json** manifest is written with metadata + chapter lists

### Phase 2: Query (Every Question)

This happens when a student selects a book and asks a question:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Select Book │───>│ book_id     │───>│  Load       │───>│  Cached     │
│ (e.g.       │    │ "maths"     │    │  Pipeline   │    │  RAG Agent  │
│  Maths Mela)│    │             │    │  (if new)   │    │  (per-book) │
└─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘
                                                                │
┌─────────────┐    ┌─────────────┐    ┌──────────────┐          │
│  Question   │───>│  Embedder   │───>│ Vector Search│──────────┘
│  "What is   │    │(same model) │    │ (book-only   │
│  a fraction?"    │             │    │  collection) │
└─────────────┘    └─────────────┘    └──────┬───────┘
                                             │ Top K chunks
                                             ▼
┌─────────────┐    ┌─────────────┐    ┌──────────────┐
│   Answer    │<───│   Ollama    │<───│ Subject-     │
│  "A fraction│    │   LLM       │    │ specific     │
│   is..."    │    │             │    │ prompt +     │
└─────────────┘    └─────────────┘    │ context      │
                                      └──────────────┘
```

---

## Per-Book Agent Design

### Why Per-Book Agents?

The previous single-agent design had these problems:
- The AI retrieved chunks from **all subjects** at once (e.g., Maths chunks when asking about English)
- A single generic system prompt could not give subject-specific guidance
- No awareness of the book's chapter/unit structure

### How It Works Now

Each book gets:

1. **Its own ChromaDB collection** -- retrieval is scoped to that book only
2. **A tailored system prompt** -- subject-specific teaching guidance
3. **Chapter/unit awareness** -- the prompt includes the full list of chapters

The RAG pipeline for each book is created lazily on first use and cached:

```python
# Simplified pipeline creation (see web_app.py)
def _get_rag(book_id):
    collection = f"cbse_grade5_{book_id}"
    prompt = get_subject_system_prompt(book_id, ...)
    return RAGPipeline(collection_name=collection, system_prompt=prompt)
```

### Chapter vs Unit Terminology

Some books use "Chapter" while others use "Unit" to organize their content. During ingestion, the script:
1. Scans the preface PDF for `Unit N` and `Chapter N` patterns
2. Picks whichever term appears more frequently
3. Stores the term as `chapter_label` in `books.json`
4. The system prompt uses the correct terminology for that book

---

## Component Details

### 1. Config (`config.py`)

Centralizes all settings including:
- Path configuration (books, data, ChromaDB)
- Chunking parameters (800 chars, 100 overlap)
- Embedding model (`all-MiniLM-L6-v2`, 384 dimensions)
- `COLLECTION_PREFIX` for per-book collection naming
- `SUBJECT_SYSTEM_PROMPTS` dict with per-subject teaching prompts
- `get_subject_system_prompt()` function that builds the full prompt with chapter listings

### 2. Ingestion (`scripts/ingest_books.py`)

The ingestion script:
- Creates one ChromaDB collection per subject
- Extracts chapter/unit lists from `*ps.pdf` preface files
- Writes `data/books.json` manifest:

```json
{
  "books": [
    {
      "id": "maths",
      "title": "Maths Mela",
      "subject": "Mathematics",
      "collection_name": "cbse_grade5_maths",
      "chapter_label": "chapter",
      "chapters": [
        {"number": 1, "title": "Numbers and Number Names"},
        {"number": 2, "title": "Addition and Subtraction"}
      ],
      "chunk_count": 245,
      "pdf_count": 15
    }
  ]
}
```

### 3. Vector Store (`embeddings/vector_store.py`)

ChromaDB-based storage. Accepts a `collection_name` parameter to select which per-subject collection to use.

### 4. Retriever (`rag/retriever.py`)

Embeds query, searches the book-specific collection, filters by score. Accepts `collection_name` to scope retrieval.

### 5. Generator / RAGPipeline (`rag/generator.py`)

Builds prompts and calls Ollama. Each instance has its own `system_prompt` (per-subject). `RAGPipeline` wires Retriever + Generator together and accepts `collection_name` + `system_prompt`.

---

## Web Interface

### Architecture (`interfaces/web_app.py`)

```
┌──────────────┐        ┌────────────────────────────────────────────┐
│   Browser    │──GET /─│  FastAPI (web_app.py)                      │
│   (HTML/JS)  │        │                                            │
│              │<──HTML─│  Jinja2 renders templates/index.html       │
│              │        │  (book selection + chat in one page)       │
│              │        │                                            │
│              │──GET───│  /api/books      → books.json manifest     │
│              │──POST──│  /api/ask        → RAGPipeline.query()     │
│              │──POST──│  /api/ask/stream → SSE token stream        │
│              │──POST──│  /api/quiz       → RAGPipeline.quiz()      │
│              │──POST──│  /api/practice   → RAGPipeline.practice()  │
│              │──POST──│  /api/explain    → RAGPipeline.explain()   │
│              │──GET───│  /api/stats      → VectorStore.get_stats   │
│              │──GET───│  /health         → health check            │
└──────────────┘        └────────────────────────────────────────────┘
```

### UI Flow

1. **Book Selection** (View 1): Card grid loaded from `/api/books`. Each card shows book title, subject, chapter count, and first 3 chapters.
2. **Chat** (View 2): Scoped to selected book. Chapter sidebar, dynamic chips, streaming chat. All API calls include `book_id`.
3. **Switch Book**: Returns to book selection without page reload.

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Per-book RAG pipelines** | Each subject gets isolated retrieval + tailored prompts |
| **Lazy-cached pipelines** | Heavy initialization only on first use per book |
| **Book manifest (books.json)** | Single source of truth for UI + prompts, auto-generated |
| **Single-page two-view UI** | No page reloads; child-friendly flow |
| **Chapter sidebar** | Quick navigation; auto-populates questions |
| **SSE streaming** | Tokens appear in real-time as the LLM generates |

### API Endpoints

| Method | Path | Body / Params | Response |
|--------|------|---------------|----------|
| `GET` | `/` | -- | HTML (book selection + chat) |
| `GET` | `/health` | -- | `{ status, books_count, total_chunks }` |
| `GET` | `/api/books` | -- | `{ books: [...] }` manifest |
| `POST` | `/api/ask` | `{ question, book_id }` | `{ answer, sources, num_chunks }` |
| `POST` | `/api/ask/stream` | `{ question, book_id, history? }` | SSE: `{ token }` ... `{ done, sources }` |
| `POST` | `/api/quiz` | `{ topic, book_id, num_questions? }` | `{ quiz }` |
| `POST` | `/api/practice` | `{ topic, book_id, num_problems? }` | `{ practice }` |
| `POST` | `/api/explain` | `{ concept, book_id }` | `{ explanation }` |
| `GET` | `/api/stats` | `?book_id=...` | `{ collection_name, document_count, unique_sources }` |

### Testing

Tests live in `tests/` and use **pytest** with FastAPI's `TestClient`. All external
dependencies (ChromaDB, sentence-transformers, Ollama) are replaced with lightweight
fakes defined in `tests/conftest.py`, including a fake `books.json` manifest.

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
"cat" -> [0.2, 0.8, -0.3, 0.5]
"dog" -> [0.25, 0.75, -0.28, 0.48]  <- Similar to cat!
"car" -> [-0.5, 0.1, 0.9, -0.2]     <- Different from cat
```

### What is Cosine Similarity?

Measures how similar two vectors are:
- 1.0 = Identical
- 0.5 = Somewhat similar
- 0.0 = Completely different

### What is RAG?

**R**etrieval-**A**ugmented **G**eneration

1. **Retrieval:** Find relevant information from the book
2. **Augmented:** Add it to the prompt
3. **Generation:** Let LLM use this context

**Why RAG?**
- LLMs don't know your textbook content
- RAG gives them the relevant context
- Results in accurate, grounded answers

### Why Per-Subject Collections?

- Prevents cross-subject contamination in retrieval
- Each agent only sees its own book's content
- Tailored prompts provide better teaching for each subject

---

## File-by-File Breakdown

```
maths_tutor/                 # Python package (RAG engine)
│
├── __init__.py              # Package initialization
│
├── config.py                # All configuration in one place
│                            # - BOOKS_DIR, DATA_DIR
│                            # - Chunk size (800), Overlap (100)
│                            # - COLLECTION_PREFIX (per-book naming)
│                            # - SUBJECT_SYSTEM_PROMPTS (per-subject)
│                            # - get_subject_system_prompt() builder
│
├── ingestion/
│   ├── pdf_parser.py        # Extracts text from PDFs (pymupdf)
│   └── chunker.py           # Splits text into chunks
│
├── embeddings/
│   ├── embedder.py          # Text -> 384-dim vectors
│   └── vector_store.py      # ChromaDB storage (per-collection)
│
├── rag/
│   ├── retriever.py         # Embed query, search book collection
│   └── generator.py         # Build prompt, call Ollama
│                            # RAGPipeline: collection_name + system_prompt
│
└── interfaces/
    ├── cli.py               # Terminal chatbot
    ├── web_app.py           # FastAPI app
    │                        # - /api/books -> books.json manifest
    │                        # - Per-book pipeline caching
    │                        # - All endpoints require book_id
    └── templates/
        └── index.html       # Two-view UI
                             # View 1: Book selection grid
                             # View 2: Book-scoped chat
```

---

## Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    THE BIG PICTURE                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. INGESTION (once):                                           │
│     Discover subjects -> PDF -> Text -> Chunks -> Embeddings    │
│     -> Per-subject ChromaDB collections + books.json manifest   │
│                                                                  │
│  2. BOOK SELECTION:                                             │
│     Student picks a book -> loads per-book RAG pipeline         │
│     -> subject-specific prompt + chapter awareness              │
│                                                                  │
│  3. QUERY (every question):                                     │
│     Question -> Embedding -> Search (book only) -> Context      │
│     -> Subject-specific LLM prompt -> Answer                    │
│                                                                  │
│  4. KEY INSIGHT:                                                │
│     Isolating each subject prevents cross-contamination         │
│     and lets each AI agent be an expert in its book!            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

This architecture is the foundation of most modern AI applications.
Understanding it will help you build chatbots, search engines, and more!
