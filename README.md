# CBSE Maths AI Tutor

A RAG (Retrieval-Augmented Generation) chatbot that helps Class 5 students learn mathematics using their CBSE textbook.

## What is This Project?

This project builds an AI tutor that:
- Answers questions about the Class 5 Maths textbook
- Generates quizzes to test understanding
- Creates practice problems with solutions
- Explains mathematical concepts step-by-step

## How It Works

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   PDF Book  │ --> │  Extract &   │ --> │   Vector    │
│  (16 files) │     │    Chunk     │     │   Store     │
└─────────────┘     └──────────────┘     └─────────────┘
                                               │
┌─────────────┐     ┌──────────────┐           │
│   Student   │ --> │   Retrieve   │ <---------┘
│  Question   │     │   Context    │
└─────────────┘     └──────────────┘
                           │
                           v
                    ┌──────────────┐     ┌─────────────┐
                    │   Generate   │ --> │   Answer    │
                    │   (Ollama)   │     │             │
                    └──────────────┘     └─────────────┘
```

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
cd cbse_books
pip install -r requirements.txt
```

### 2. Ingest the Textbook (Run Once)

This processes all PDF files and stores them in the vector database:

```bash
python scripts/ingest_books.py
```

### 3. Start the Tutor

```bash
python -m maths_tutor.interfaces.cli
```

## Project Structure

```
cbse_books/
├── cbse-grade-5-maths/     # PDF textbook files
├── maths_tutor/            # Main Python package
│   ├── config.py           # Configuration settings
│   ├── ingestion/          # PDF parsing & chunking
│   ├── embeddings/         # Vector operations
│   ├── rag/                # RAG pipeline
│   └── interfaces/         # CLI & Web interfaces
├── data/                   # Generated data (ChromaDB)
├── scripts/                # Utility scripts
└── ARCHITECTURE.md         # Detailed architecture docs
```

## Usage Examples

### Ask a Question
```
🎓 Maths Tutor > What is the place value of 5 in 3572?

The place value of 5 in 3572 is 500 (five hundred).
In 3572, the 5 is in the hundreds place, so its value is 5 × 100 = 500.
```

### Generate a Quiz
```
🎓 Maths Tutor > /quiz fractions 5

Creating 5 questions about fractions...

Q1: What is 1/2 + 1/4?
A) 1/4  B) 2/4  C) 3/4  D) 1
Correct: C
```

### Get Practice Problems
```
🎓 Maths Tutor > /practice multiplication 3

Practice Problems:
1. Calculate 24 × 15
2. A box has 12 rows of 8 chocolates. How many chocolates in total?
3. If one book costs ₹45, what is the cost of 7 books?
```

## Key Concepts (For Learning)

### What is RAG?
RAG = Retrieval-Augmented Generation
1. **Retrieval**: Find relevant information from the textbook
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
- Chunk size (default: 800 characters)
- Number of chunks to retrieve (default: 5)
- Ollama model (default: llama3.2)

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

## Learning Resources

- [What are Embeddings?](https://vickiboykis.com/what_are_embeddings/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Ollama Documentation](https://ollama.ai/docs)
- [Sentence Transformers](https://www.sbert.net/)

## Next Steps

After mastering this project, you can:
1. Add support for more textbooks
2. Implement a web interface with FastAPI
3. Add image understanding for diagrams
4. Implement conversation history
5. Add user progress tracking

Happy Learning! 📚
