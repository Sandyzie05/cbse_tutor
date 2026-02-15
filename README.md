# CBSE AI Tutor

A RAG (Retrieval-Augmented Generation) chatbot that helps Class 5 students learn across all CBSE curriculum subjects using their official textbooks. Each subject gets its own AI agent with a dedicated knowledge base and tailored system prompt.

## What is This Project?

This project builds an AI tutor that covers the **full CBSE Grade 5 curriculum** with **per-subject agents**:

| Subject | Book | AI Agent Focus |
|---------|------|----------------|
| **Mathematics** | Maths Mela | Step-by-step calculations, visual thinking |
| **English** | Santoor | Grammar, vocabulary, comprehension, creative expression |
| **Arts** | (auto-detected) | Cultural context, creativity, observation |
| **The World Around Us** | (auto-detected) | Everyday life connections, scientific inquiry |
| **Physical Education & Wellbeing** | (auto-detected) | Health, movement, wellbeing |

The tutor can:
- Answer questions scoped to a specific textbook
- Generate quizzes to test understanding
- Create practice problems with solutions
- Explain concepts step-by-step
- Show the chapter/unit structure of each book

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          BOOK SELECTION UI                                  │
│   [Maths Mela]  [Santoor]  [Arts]  [World Around Us]  [PE & Wellbeing]   │
└────────┬───────────┬──────────┬───────────┬──────────────┬────────────────┘
         │           │          │           │              │
         v           v          v           v              v
   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
   │  Maths   │ │ English  │ │  Arts    │ │  EVS     │ │   PE     │
   │  Agent   │ │  Agent   │ │  Agent   │ │  Agent   │ │  Agent   │
   │(prompt+  │ │(prompt+  │ │(prompt+  │ │(prompt+  │ │(prompt+  │
   │ chapters)│ │  units)  │ │ chapters)│ │ chapters)│ │ chapters)│
   └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
        │             │            │             │             │
        v             v            v             v             v
   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
   │ ChromaDB │ │ ChromaDB │ │ ChromaDB │ │ ChromaDB │ │ ChromaDB │
   │  maths   │ │ english  │ │  arts    │ │   evs    │ │   pe     │
   └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

**Key design:**
- **One ChromaDB collection per book** -- each agent retrieves ONLY from its book's content
- **Per-subject system prompts** -- tailored guidance for each subject's teaching style
- **Chapter/unit awareness** -- the AI knows the book's structure (chapters or units)
- **Book manifest** (`data/books.json`) -- auto-generated during ingestion with metadata + chapter lists

## Books & Subjects

All textbooks live under `cbse-books/` organized by grade and subject:

```
cbse-books/
└── cbse-grade-5/
    ├── cbse-grade-5-maths/                          # 15 chapter PDFs
    ├── cbse-grade-5-english/                        # 11 chapter PDFs
    ├── cbse-grade-5-arts/                           # 20 chapter PDFs
    ├── cbse-grade-5-theWorldAroundUs/               # 11 chapter PDFs
    └── cbse-grade-5-physicalEducationAndWellbeing/  #  5 chapter PDFs
```

To add more grades or subjects in the future, create a grade folder (`cbse-grade-6/`) with subject subdirectories following the naming convention `cbse-grade-<N>-<subject>/` and place the PDF files inside.

## Prerequisites

1. **Python 3.9+**
2. **Ollama** - Install from https://ollama.ai
   ```bash
   # After installing Ollama, pull the model:
   ollama pull llama3.2
   ```

## Quick Start

### 1. Install Dependencies

```bash
cd cbse_tutor
pip install -r requirements.txt
```

### 2. Ingest All Textbooks (Run Once)

This processes all PDF files, creates per-subject ChromaDB collections, extracts chapter lists, and writes a `data/books.json` manifest:

```bash
python scripts/ingest_books.py
```

The script will:
- Auto-discover all subject subdirectories under `cbse-books/`
- Extract chapter/unit lists from each book's preface PDF
- Parse, chunk, and embed every PDF into per-subject collections
- Write `data/books.json` with book metadata and chapter listings

### 3. Start the Tutor

**Option A: Command-line interface**

```bash
python -m maths_tutor.interfaces.cli
```

**Option B: Web interface** (recommended)

```bash
uvicorn maths_tutor.interfaces.web_app:app --reload
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

The web UI provides:
- **Book selection screen** -- pick a subject to start a learning session
- **Per-book chat** with real-time streaming answers
- **Chapter sidebar** -- browse and ask about specific chapters/units
- Mode switcher: Ask / Quiz / Practice / Explain
- Source citations for every answer

## Project Structure

```
cbse_tutor/
├── cbse-books/            # PDF textbook files (by grade > subject)
│   └── cbse-grade-5/
│       ├── cbse-grade-5-maths/
│       ├── cbse-grade-5-english/
│       ├── cbse-grade-5-arts/
│       ├── cbse-grade-5-theWorldAroundUs/
│       └── cbse-grade-5-physicalEducationAndWellbeing/
├── maths_tutor/           # Main Python package (RAG engine)
│   ├── config.py          # Configuration + per-subject system prompts
│   ├── ingestion/         # PDF parsing & chunking
│   ├── embeddings/        # Vector operations (ChromaDB)
│   ├── rag/               # RAG pipeline (retriever + generator)
│   └── interfaces/        # CLI & Web interfaces
│       ├── cli.py         # Terminal chatbot
│       ├── web_app.py     # FastAPI app (REST + SSE streaming)
│       └── templates/     # Jinja2 HTML templates
├── tests/                 # Unit tests (pytest)
├── data/                  # Generated data
│   ├── chroma_db/         # ChromaDB per-subject collections
│   └── books.json         # Auto-generated book manifest
├── scripts/               # Utility scripts (ingestion, debug, test)
├── pyproject.toml         # Linting (ruff) & test config
└── ARCHITECTURE.md        # Detailed architecture docs
```

## Usage Examples

### Ask a Question (scoped to a book)

After selecting "Maths Mela" from the book selection screen:
```
What is the place value of 5 in 3572?

The place value of 5 in 3572 is 500 (five hundred).
In 3572, the 5 is in the hundreds place, so its value is 5 x 100 = 500.
```

After selecting "The World Around Us":
```
What are the different types of habitats?

There are mainly three types of habitats: terrestrial (land), aquatic (water),
and aerial (air). Each habitat has unique conditions...
```

### Generate a Quiz

Switch to "Quiz" mode and enter a topic:
```
fractions

Q1: What is 1/2 + 1/4?
A) 1/4  B) 2/4  C) 3/4  D) 1
Correct: C
```

### Get Practice Problems

Switch to "Practice" mode:
```
multiplication

Practice Problems:
1. Calculate 24 x 15
2. A box has 12 rows of 8 chocolates. How many chocolates in total?
3. If one book costs Rs 45, what is the cost of 7 books?
```

## Configuration

Edit `maths_tutor/config.py` to adjust:
- `BOOKS_DIR` - Root directory for all textbook PDFs
- `CHUNK_SIZE` - Chunk size (default: 800 characters)
- `TOP_K_CHUNKS` - Number of chunks to retrieve (default: 8)
- `OLLAMA_MODEL` - LLM model (default: llama3.2)
- `COLLECTION_PREFIX` - Prefix for per-subject ChromaDB collections
- `SUBJECT_SYSTEM_PROMPTS` - Per-subject AI agent prompts

## Troubleshooting

### "Ollama connection refused"
Make sure Ollama is running:
```bash
ollama serve
```

### "No books found" in the web UI
Run the ingestion script first:
```bash
python scripts/ingest_books.py
```

### "Model not found"
Pull the model:
```bash
ollama pull llama3.2
```

### Upgrading from single-collection version
If you previously ran the project with a single `cbse_grade5_all_subjects` collection, re-run ingestion to create the new per-subject collections:
```bash
python scripts/ingest_books.py
```
The old collection will remain but is no longer used. You can delete `data/chroma_db/` to clean up.

## Learning Resources

- [What are Embeddings?](https://vickiboykis.com/what_are_embeddings/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Ollama Documentation](https://ollama.ai/docs)
- [Sentence Transformers](https://www.sbert.net/)

## Development

### Running Tests

```bash
python -m pytest tests/ -v
```

### Linting

```bash
# Check
ruff check maths_tutor/ tests/

# Auto-fix
ruff check maths_tutor/ tests/ --fix
```

### Web API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Book selection + chat UI (HTML) |
| `GET` | `/health` | Health check |
| `GET` | `/api/books` | Book manifest with chapters |
| `POST` | `/api/ask` | Ask a question (JSON, requires `book_id`) |
| `POST` | `/api/ask/stream` | Ask with SSE streaming (requires `book_id`) |
| `POST` | `/api/quiz` | Generate quiz (requires `book_id`) |
| `POST` | `/api/practice` | Generate practice problems (requires `book_id`) |
| `POST` | `/api/explain` | Explain a concept (requires `book_id`) |
| `GET` | `/api/stats` | Vector store statistics (optional `book_id` query param) |

## Next Steps

After mastering this project, you can:
1. Add support for more grades (6, 7, 8, etc.)
2. Add image understanding for diagrams
3. Add a "Teacher mode" with deeper explanations and lesson plans
4. Add user progress tracking
5. Cross-book search for interdisciplinary questions
