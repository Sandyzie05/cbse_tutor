# CBSE AI Tutor

A RAG (Retrieval-Augmented Generation) chatbot that helps Class 5 students learn across all CBSE curriculum subjects using their official textbooks.

## What is This Project?

This project builds an AI tutor that covers the **full CBSE Grade 5 curriculum**:

| Subject | Description |
|---------|-------------|
| **Mathematics** | Numbers, fractions, geometry, data handling, and more |
| **English** | Grammar, comprehension, vocabulary, and literature |
| **Arts** | Creative expression, Indian art forms, culture |
| **The World Around Us** | Science, environment, social studies (EVS) |
| **Physical Education & Wellbeing** | Health, fitness, and wellness |

The tutor can:
- Answer questions from any of the above subjects
- Generate quizzes to test understanding
- Create practice problems with solutions
- Explain concepts step-by-step

## How It Works

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│  PDF Books      │ --> │  Extract &   │ --> │   Vector    │
│  (5 subjects,   │     │    Chunk     │     │   Store     │
│   ~60 PDFs)     │     │              │     │  (ChromaDB) │
└─────────────────┘     └──────────────┘     └─────────────┘
                                                   │
┌─────────────────┐     ┌──────────────┐           │
│    Student      │ --> │   Retrieve   │ <---------┘
│   Question      │     │   Context    │
└─────────────────┘     └──────────────┘
                              │
                              v
                       ┌──────────────┐     ┌─────────────┐
                       │   Generate   │ --> │   Answer    │
                       │   (Ollama)   │     │             │
                       └──────────────┘     └─────────────┘
```

## Books & Subjects

All textbooks live under `cbse-books/` organized by subject:

```
cbse-books/
├── cbse-grade-5-maths/                          # 15 chapter PDFs
├── cbse-grade-5-english/                        # 10 chapter PDFs
├── cbse-grade-5-arts/                           # 19 chapter PDFs
├── cbse-grade-5-theWorldAroundUs/               # 10 chapter PDFs
└── cbse-grade-5-physicalEducationAndWellbeing/  #  5 chapter PDFs
```

To add more grades or subjects in the future, simply create a new directory following the naming convention `cbse-grade-<N>-<subject>/` and place the PDF files inside.

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

This processes all PDF files across every subject and stores them in the vector database:

```bash
python scripts/ingest_books.py
```

The script will:
- Auto-discover all subject subdirectories under `cbse-books/`
- Parse, chunk, and embed every PDF
- Store everything in ChromaDB with subject metadata

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
- A chat interface with real-time streaming answers
- Mode switcher: Ask / Quiz / Practice / Explain
- Quick-action chips for common questions
- Source citations for every answer

## Project Structure

```
cbse_tutor/
├── cbse-books/            # PDF textbook files (organized by subject)
│   ├── cbse-grade-5-maths/
│   ├── cbse-grade-5-english/
│   ├── cbse-grade-5-arts/
│   ├── cbse-grade-5-theWorldAroundUs/
│   └── cbse-grade-5-physicalEducationAndWellbeing/
├── maths_tutor/           # Main Python package (RAG engine)
│   ├── config.py          # Configuration settings
│   ├── ingestion/         # PDF parsing & chunking
│   ├── embeddings/        # Vector operations (ChromaDB)
│   ├── rag/               # RAG pipeline (retriever + generator)
│   └── interfaces/        # CLI & Web interfaces
│       ├── cli.py         # Terminal chatbot
│       ├── web_app.py     # FastAPI app (REST + SSE streaming)
│       └── templates/     # Jinja2 HTML templates
├── tests/                 # Unit tests (pytest)
├── data/                  # Generated data (ChromaDB vector store)
├── scripts/               # Utility scripts (ingestion, debug, test)
├── pyproject.toml         # Linting (ruff) & test config
└── ARCHITECTURE.md        # Detailed architecture docs
```

## Usage Examples

### Ask a Question (any subject)

```
🎓 CBSE Tutor > What is the place value of 5 in 3572?

The place value of 5 in 3572 is 500 (five hundred).
In 3572, the 5 is in the hundreds place, so its value is 5 × 100 = 500.
```

```
🎓 CBSE Tutor > What are the different types of habitats?

There are mainly three types of habitats: terrestrial (land), aquatic (water),
and aerial (air). Each habitat has unique conditions...
```

### Generate a Quiz

```
🎓 CBSE Tutor > /quiz fractions 5

Creating 5 questions about fractions...

Q1: What is 1/2 + 1/4?
A) 1/4  B) 2/4  C) 3/4  D) 1
Correct: C
```

### Get Practice Problems

```
🎓 CBSE Tutor > /practice multiplication 3

Practice Problems:
1. Calculate 24 × 15
2. A box has 12 rows of 8 chocolates. How many chocolates in total?
3. If one book costs ₹45, what is the cost of 7 books?
```

## Key Concepts (For Learning)

### What is RAG?
RAG = Retrieval-Augmented Generation
1. **Retrieval**: Find relevant information from the textbooks
2. **Augmented**: Add this information to the prompt
3. **Generation**: Let the LLM generate a response using this context

### Why Chunking?
- LLMs have token limits
- Smaller chunks = more precise retrieval
- Overlap ensures we don't lose information at boundaries

### Why Embeddings?
- Convert text to numbers (vectors)
- Similar text = similar vectors
- Enables semantic search (search by meaning, not keywords)

## Configuration

Edit `maths_tutor/config.py` to adjust:
- `BOOKS_DIR` - Root directory for all textbook PDFs
- `CHUNK_SIZE` - Chunk size (default: 800 characters)
- `TOP_K_CHUNKS` - Number of chunks to retrieve (default: 8)
- `OLLAMA_MODEL` - LLM model (default: llama3.2)
- `COLLECTION_NAME` - ChromaDB collection name

## Troubleshooting

### "Ollama connection refused"
Make sure Ollama is running:
```bash
ollama serve
```

### "No chunks found"
Run the ingestion script first:
```bash
python scripts/ingest_books.py
```

### "Model not found"
Pull the model:
```bash
ollama pull llama3.2
```

### Old data from previous Maths-only ingestion
If you previously ran the project with only Maths books, re-run ingestion to rebuild the vector store with all subjects:
```bash
python scripts/ingest_books.py
```

### Re-ingestion after upgrades
The ingestion script now enriches every chunk with a subject/book preamble (e.g. `[English - Santoor (Grade 5)]`) and tags chunks with `content_type` metadata (`table_of_contents`, `preface`, `chapter`). If you are upgrading from an older version, you **must** re-run ingestion to rebuild the vector store:
```bash
python scripts/ingest_books.py
```
This ensures the LLM can retrieve table-of-contents listings and refer to sources by subject name instead of raw filenames.

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
| `GET` | `/` | Chat UI (HTML) |
| `GET` | `/health` | Health check |
| `POST` | `/api/ask` | Ask a question (JSON response) |
| `POST` | `/api/ask/stream` | Ask with Server-Sent Events streaming |
| `POST` | `/api/quiz` | Generate quiz on a topic |
| `POST` | `/api/practice` | Generate practice problems |
| `POST` | `/api/explain` | Explain a concept |
| `GET` | `/api/stats` | Vector store statistics |

## Next Steps

After mastering this project, you can:
1. Add support for more grades (6, 7, 8, etc.)
2. Add image understanding for diagrams
3. Implement conversation history
4. Add user progress tracking
5. Subject-specific quiz modes and practice sessions

Happy Learning! 📚
