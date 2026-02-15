"""Shared fixtures for the CBSE Tutor test suite."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fake books manifest used across all tests
# ---------------------------------------------------------------------------

FAKE_BOOKS_MANIFEST = {
    "books": [
        {
            "id": "maths",
            "title": "Maths Mela",
            "subject": "Mathematics",
            "collection_name": "cbse_grade5_maths",
            "chapter_label": "chapter",
            "chapters": [
                {"number": 1, "title": "Numbers and Number Names"},
                {"number": 2, "title": "Addition and Subtraction"},
            ],
            "chunk_count": 42,
            "pdf_count": 15,
        },
        {
            "id": "english",
            "title": "Santoor",
            "subject": "English",
            "collection_name": "cbse_grade5_english",
            "chapter_label": "chapter",
            "chapters": [
                {"number": 1, "title": "My Family", "unit_number": 1, "unit_title": "Let's Have Fun"},
                {"number": 2, "title": "The Kite", "unit_number": 1, "unit_title": "Let's Have Fun"},
            ],
            "units": [
                {"number": 1, "title": "Let's Have Fun"},
            ],
            "chunk_count": 38,
            "pdf_count": 11,
        },
    ]
}

# ---------------------------------------------------------------------------
# Fake RAG pipeline that never touches Ollama or ChromaDB
# ---------------------------------------------------------------------------


class FakeRetrievalResult:
    """Minimal stand-in for maths_tutor.rag.retriever.RetrievalResult."""

    def __init__(self):
        self.chunks = ["Chunk 1 about addition.", "Chunk 2 about subtraction."]
        self.scores = [0.92, 0.85]
        self.sources = [
            {
                "source_file": "eemm101.pdf",
                "page_number": 3,
                "subject": "maths",
                "book_title": "Maths Mela",
                "content_type": "chapter",
            },
            {
                "source_file": "eemm102.pdf",
                "page_number": 7,
                "subject": "maths",
                "book_title": "Maths Mela",
                "content_type": "chapter",
            },
        ]
        self.query = "test query"
        self.has_results = True

    def format_for_prompt(self) -> str:
        return "\n\n".join(self.chunks)


class FakeRetriever:
    """Retriever that returns canned results without vector DB access."""

    def __init__(self, **_):
        self.vector_store = MagicMock()
        self.vector_store.count = 42

    def retrieve(self, query: str, **kwargs) -> FakeRetrievalResult:
        return FakeRetrievalResult()

    def retrieve_for_topic(self, topic: str, **kwargs) -> FakeRetrievalResult:
        return FakeRetrievalResult()

    def get_available_sources(self):
        return ["eemm101.pdf", "eemm102.pdf"]


class FakeGenerator:
    """Generator that returns canned text without calling Ollama."""

    def __init__(self, **_):
        self.model = "fake-model"
        self.retriever = FakeRetriever()
        self.system_prompt = "You are a test tutor."

    def generate(self, question: str, context=None, system_prompt=None, stream=False):
        if stream:
            return iter(["Hello ", "world!"])
        return "This is a test answer about addition."

    def generate_quiz(self, topic: str, num_questions: int = 5) -> str:
        return "Q1: What is 2+2?\nA) 3 B) 4 C) 5 D) 6\nCorrect: B"

    def generate_practice(self, topic: str, num_problems: int = 3) -> str:
        return "Problem 1: Calculate 5 x 6.\nSolution: 30"

    def explain(self, concept: str, simple: bool = True) -> str:
        return f"{concept} is a very important concept."


class FakeRAGPipeline:
    """RAGPipeline that never touches external services."""

    def __init__(self, **_):
        self.retriever = FakeRetriever()
        self.generator = FakeGenerator()

    def query(self, question: str, stream=False, include_sources=False):
        if stream:
            return self.generator.generate(question, stream=True)
        if include_sources:
            return {
                "answer": "This is a test answer.",
                "sources": self.retriever.retrieve(question).sources,
                "scores": [0.92, 0.85],
                "num_chunks": 2,
            }
        return "This is a test answer."

    def quiz(self, topic: str, num_questions: int = 5) -> str:
        return self.generator.generate_quiz(topic, num_questions)

    def practice(self, topic: str, num_problems: int = 3) -> str:
        return self.generator.generate_practice(topic, num_problems)

    def explain(self, concept: str) -> str:
        return self.generator.explain(concept)


# ---------------------------------------------------------------------------
# Fake VectorStore
# ---------------------------------------------------------------------------


class FakeVectorStore:
    """VectorStore that returns canned stats without touching ChromaDB."""

    def __init__(self, **_):
        pass

    @property
    def count(self) -> int:
        return 42

    def get_stats(self) -> dict:
        return {
            "collection_name": "cbse_grade5_maths",
            "document_count": 42,
            "unique_sources": ["eemm101.pdf", "eemm102.pdf"],
            "persist_directory": "/tmp/fake_chroma",
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fake_ollama_chat(model, messages, stream=False, **_):
    """Stand-in for ollama.chat used by the streaming endpoint."""
    if stream:
        return iter([
            {"message": {"content": "Hello "}},
            {"message": {"content": "world!"}},
        ])
    return {"message": {"content": "This is a test answer."}}


@pytest.fixture()
def test_client(tmp_path):
    """
    TestClient with RAGPipeline and VectorStore fully mocked out.

    Heavy model loading (sentence-transformers, ChromaDB, Ollama) is
    completely bypassed so tests run in milliseconds without any external
    dependencies.

    A temporary books.json manifest is created so the /api/books endpoint
    and book_id validation work correctly.
    """
    # Write a fake books.json manifest
    manifest_path = tmp_path / "books.json"
    manifest_path.write_text(json.dumps(FAKE_BOOKS_MANIFEST))

    with (
        patch(
            "maths_tutor.rag.generator.RAGPipeline",
            FakeRAGPipeline,
        ),
        patch(
            "maths_tutor.embeddings.vector_store.VectorStore",
            FakeVectorStore,
        ),
        patch(
            "ollama.chat",
            _fake_ollama_chat,
        ),
        patch(
            "maths_tutor.interfaces.web_app.BOOKS_MANIFEST_PATH",
            manifest_path,
        ),
    ):
        from maths_tutor.interfaces.web_app import create_app

        app = create_app()
        yield TestClient(app)
