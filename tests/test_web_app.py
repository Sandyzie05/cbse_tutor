"""Tests for the FastAPI web interface."""

import json


# ── Page routes ──────────────────────────────────────────────────────────────


class TestIndexPage:
    """GET / serves the book selection + chat UI HTML."""

    def test_returns_html(self, test_client):
        resp = test_client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_contains_app_title(self, test_client):
        resp = test_client.get("/")
        assert "CBSE AI Tutor" in resp.text


# ── Health ───────────────────────────────────────────────────────────────────


class TestHealth:
    def test_health_ok(self, test_client):
        resp = test_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["books_count"] == 2
        assert body["total_chunks"] == 80  # 42 + 38


# ── GET /api/books ──────────────────────────────────────────────────────────


class TestBooks:
    def test_returns_books(self, test_client):
        resp = test_client.get("/api/books")
        assert resp.status_code == 200
        body = resp.json()
        assert "books" in body
        assert len(body["books"]) == 2

    def test_books_have_required_fields(self, test_client):
        resp = test_client.get("/api/books")
        body = resp.json()
        for book in body["books"]:
            assert "id" in book
            assert "title" in book
            assert "subject" in book
            assert "collection_name" in book
            assert "chapter_label" in book
            assert "chapters" in book
            assert isinstance(book["chapters"], list)

    def test_maths_book_has_chapters(self, test_client):
        resp = test_client.get("/api/books")
        body = resp.json()
        maths = next(b for b in body["books"] if b["id"] == "maths")
        assert maths["title"] == "Maths Mela"
        assert maths["chapter_label"] == "chapter"
        assert len(maths["chapters"]) == 2

    def test_english_book_has_units(self, test_client):
        resp = test_client.get("/api/books")
        body = resp.json()
        english = next(b for b in body["books"] if b["id"] == "english")
        assert english["chapter_label"] == "chapter"
        assert len(english.get("units", [])) == 1
        assert english["units"][0]["title"] == "Let's Have Fun"
        # Chapters reference their parent unit
        assert english["chapters"][0]["unit_number"] == 1


# ── POST /api/ask ────────────────────────────────────────────────────────────


class TestAsk:
    def test_returns_answer(self, test_client):
        resp = test_client.post(
            "/api/ask",
            json={"question": "What is addition?", "book_id": "maths", "stream": False},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "answer" in body
        assert len(body["answer"]) > 0

    def test_includes_sources(self, test_client):
        resp = test_client.post(
            "/api/ask",
            json={"question": "What is addition?", "book_id": "maths", "stream": False},
        )
        body = resp.json()
        assert "sources" in body
        assert isinstance(body["sources"], list)

    def test_rejects_empty_question(self, test_client):
        resp = test_client.post("/api/ask", json={"question": "", "book_id": "maths"})
        assert resp.status_code == 422

    def test_rejects_missing_question(self, test_client):
        resp = test_client.post("/api/ask", json={"book_id": "maths"})
        assert resp.status_code == 422

    def test_rejects_missing_book_id(self, test_client):
        resp = test_client.post("/api/ask", json={"question": "What is 2+2?"})
        assert resp.status_code == 422

    def test_rejects_invalid_book_id(self, test_client):
        resp = test_client.post(
            "/api/ask",
            json={"question": "Hello", "book_id": "nonexistent", "stream": False},
        )
        assert resp.status_code == 404


# ── POST /api/ask/stream ────────────────────────────────────────────────────


class TestAskStream:
    def test_streams_tokens(self, test_client):
        resp = test_client.post(
            "/api/ask/stream",
            json={"question": "What is 2+2?", "book_id": "maths"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        tokens = []
        is_done = False
        for line in resp.text.splitlines():
            if not line.startswith("data: "):
                continue
            payload = json.loads(line[6:])
            if "token" in payload:
                tokens.append(payload["token"])
            if payload.get("done"):
                is_done = True

        assert len(tokens) > 0, "Expected at least one token event"
        assert is_done, "Expected a done event with sources"


# ── POST /api/quiz ───────────────────────────────────────────────────────────


class TestQuiz:
    def test_returns_quiz(self, test_client):
        resp = test_client.post(
            "/api/quiz", json={"topic": "fractions", "book_id": "maths"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "quiz" in body
        assert len(body["quiz"]) > 0

    def test_rejects_empty_topic(self, test_client):
        resp = test_client.post(
            "/api/quiz", json={"topic": "", "book_id": "maths"}
        )
        assert resp.status_code == 422

    def test_rejects_missing_book_id(self, test_client):
        resp = test_client.post("/api/quiz", json={"topic": "fractions"})
        assert resp.status_code == 422


# ── POST /api/practice ──────────────────────────────────────────────────────


class TestPractice:
    def test_returns_practice(self, test_client):
        resp = test_client.post(
            "/api/practice",
            json={"topic": "multiplication", "book_id": "maths"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "practice" in body
        assert len(body["practice"]) > 0

    def test_rejects_empty_topic(self, test_client):
        resp = test_client.post(
            "/api/practice", json={"topic": "", "book_id": "maths"}
        )
        assert resp.status_code == 422


# ── POST /api/explain ───────────────────────────────────────────────────────


class TestExplain:
    def test_returns_explanation(self, test_client):
        resp = test_client.post(
            "/api/explain", json={"concept": "gravity", "book_id": "maths"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "explanation" in body
        assert "gravity" in body["explanation"]

    def test_rejects_empty_concept(self, test_client):
        resp = test_client.post(
            "/api/explain", json={"concept": "", "book_id": "maths"}
        )
        assert resp.status_code == 422


# ── GET /api/stats ──────────────────────────────────────────────────────────


class TestStats:
    def test_returns_stats(self, test_client):
        resp = test_client.get("/api/stats?book_id=maths")
        assert resp.status_code == 200
        body = resp.json()
        assert body["document_count"] == 42
        assert isinstance(body["unique_sources"], list)
