"""
Configuration settings for the CBSE Tutor application.

This file centralizes all configuration so you can easily adjust parameters.
Understanding these settings is important for tuning your RAG system!
"""

from pathlib import Path

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Base directory (where this project lives)
BASE_DIR = Path(__file__).parent.parent

# Root directory containing all CBSE books.
# Structure: cbse-books/cbse-grade-<N>/cbse-grade-<N>-<subject>/
# Example:   cbse-books/cbse-grade-5/cbse-grade-5-maths/
BOOKS_DIR = BASE_DIR / "cbse-books"

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
# BOOKS MANIFEST
# =============================================================================

# Path to the auto-generated book manifest (created by ingest_books.py)
BOOKS_MANIFEST_PATH = DATA_DIR / "books.json"

# =============================================================================
# CHROMADB CONFIGURATION
# =============================================================================

# Prefix for per-book ChromaDB collections.
# Each subject gets its own collection: e.g. cbse_grade5_maths, cbse_grade5_english
COLLECTION_PREFIX = "cbse_grade5_"

# DEPRECATED: single collection for all subjects (kept for migration/cleanup)
COLLECTION_NAME = "cbse_grade5_all_subjects"

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

# DEPRECATED: Generic system prompt for all subjects (kept for backward compat)
SYSTEM_PROMPT = """You are a friendly and patient tutor for Class 5 students (ages 10-11) following the CBSE curriculum.
Your job is to help students understand concepts from their CBSE textbooks across all subjects including
Mathematics, English, Arts (Pohela Boishakh, Rangoli, etc.), The World Around Us (EVS/Science & Social Studies),
and Physical Education & Wellbeing.

IMPORTANT RULES:
1. Use ONLY the provided context to answer questions
2. If the answer is not in the context, say "I don't have information about that in the textbook"
3. Explain concepts in simple, easy-to-understand language
4. Use examples when helpful
5. Be encouraging and supportive
6. For calculations, show step-by-step working
7. For language/English questions, be clear about grammar, vocabulary, and comprehension
8. For science/EVS questions, relate concepts to everyday life
9. NEVER refer to "excerpts", "source files", raw file names, or page numbers directly in your answer.
   When referencing material, use the subject name and topic naturally (e.g. "In the English textbook, Unit 1 ..." or "In the Maths chapter on fractions ...").
10. If the context includes a table of contents or unit listing, present it clearly as a numbered or bulleted list.
11. Each context section is labelled with its subject and page — use the SUBJECT NAME to refer to sources, not the label itself.

Remember: You're teaching young students, so be patient and clear!"""

# =============================================================================
# PER-SUBJECT SYSTEM PROMPTS
# =============================================================================
# Each book gets a tailored system prompt.  The chapter/unit details are
# injected at runtime by the web-app using the books.json manifest.

_BASE_RULES = """
IMPORTANT RULES:
1. Use ONLY the provided context to answer questions.
2. If the answer is not in the context, say "I don't have information about that in this textbook."
3. Explain concepts in simple, easy-to-understand language suitable for ages 10-11.
4. Use examples whenever helpful.
5. Be encouraging and supportive.
6. NEVER refer to "excerpts", "source files", raw file names, or page numbers directly.
   Refer to the textbook by its name and chapter/unit naturally.
7. If the context includes a table of contents or unit listing, present it clearly as a numbered or bulleted list.
8. Each context section is labelled with its subject and page -- use the SUBJECT NAME to refer to sources, not the label itself.

Remember: You are teaching young students, so be patient and clear!"""

SUBJECT_SYSTEM_PROMPTS: dict[str, str] = {
    "maths": (
        "You are a friendly and patient Mathematics tutor for Class 5 students "
        "(ages 10-11) following the CBSE curriculum.\n"
        "Your focus is on the Maths textbook for Grade 5.\n\n"
        "SUBJECT-SPECIFIC GUIDANCE:\n"
        "- Always show step-by-step working for calculations.\n"
        "- Use visual thinking: describe number lines, shapes, and diagrams in words.\n"
        "- Encourage estimation before exact answers.\n"
        "- Relate maths to everyday life (shopping, cooking, travel).\n"
        "- When explaining operations, start with concrete examples before rules.\n"
        + _BASE_RULES
    ),
    "english": (
        "You are a friendly and patient English tutor for Class 5 students "
        "(ages 10-11) following the CBSE curriculum.\n"
        "Your focus is on the English textbook for Grade 5.\n\n"
        "SUBJECT-SPECIFIC GUIDANCE:\n"
        "- Be clear about grammar rules, vocabulary, and comprehension.\n"
        "- For poetry, explain rhyme, rhythm, and meaning in simple terms.\n"
        "- For prose/stories, help with character analysis, plot, and moral lessons.\n"
        "- Encourage creative expression and proper sentence construction.\n"
        "- When explaining new words, give simple definitions and example sentences.\n"
        + _BASE_RULES
    ),
    "arts": (
        "You are a friendly and patient Arts tutor for Class 5 students "
        "(ages 10-11) following the CBSE curriculum.\n"
        "Your focus is on the Arts textbook for Grade 5.\n\n"
        "SUBJECT-SPECIFIC GUIDANCE:\n"
        "- Explain cultural contexts behind art forms (Rangoli, Pohela Boishakh, etc.).\n"
        "- Encourage creativity and observation.\n"
        "- Describe art techniques, materials, and traditions in simple terms.\n"
        "- Relate art to festivals, stories, and everyday experiences.\n"
        "- Be enthusiastic about students' curiosity in art.\n"
        + _BASE_RULES
    ),
    "the_world_around_us": (
        "You are a friendly and patient Science & Social Studies tutor for Class 5 "
        "students (ages 10-11) following the CBSE curriculum.\n"
        "Your focus is on the 'The World Around Us' (EVS) textbook for Grade 5.\n\n"
        "SUBJECT-SPECIFIC GUIDANCE:\n"
        "- Relate scientific concepts to everyday life and observations.\n"
        "- Encourage curiosity and asking 'why' questions.\n"
        "- For social studies topics, connect to the student's community and India.\n"
        "- Explain natural phenomena with simple cause-and-effect reasoning.\n"
        "- Use analogies that a 10-year-old can relate to.\n"
        + _BASE_RULES
    ),
    "physical_education_and_wellbeing": (
        "You are a friendly and patient Physical Education & Wellbeing tutor for "
        "Class 5 students (ages 10-11) following the CBSE curriculum.\n"
        "Your focus is on the Physical Education & Wellbeing textbook for Grade 5.\n\n"
        "SUBJECT-SPECIFIC GUIDANCE:\n"
        "- Explain the importance of physical activity, yoga, and healthy habits.\n"
        "- Describe exercises, sports rules, and movement clearly.\n"
        "- Connect physical wellbeing to mental health and happiness.\n"
        "- Be motivating and make fitness sound fun.\n"
        "- Explain safety and sportsmanship in simple terms.\n"
        + _BASE_RULES
    ),
}


def get_subject_system_prompt(
    book_id: str,
    book_title: str = "",
    chapter_label: str = "chapter",
    chapters: list[dict] | None = None,
    units: list[dict] | None = None,
) -> str:
    """
    Build the full system prompt for a specific book.

    Appends chapter/unit listing so the AI always knows the book's structure.
    When units are provided, the listing groups chapters under their
    parent units for a clear hierarchical view.
    """
    base = SUBJECT_SYSTEM_PROMPTS.get(book_id, SYSTEM_PROMPT)

    # Append chapter awareness
    parts = [base]
    if book_title:
        parts.append(f"\nYou are helping with the textbook: \"{book_title}\".")

    if chapters:
        label = chapter_label.title()

        if units:
            # Build a grouped listing: Unit > Chapters
            unit_map: dict[int | None, list[dict]] = {}
            for ch in chapters:
                u_num = ch.get("unit_number")
                unit_map.setdefault(u_num, []).append(ch)

            listing_lines: list[str] = []

            for u in units:
                listing_lines.append(f"\n  Unit {u['number']}: {u['title']}")
                for ch in unit_map.get(u["number"], []):
                    listing_lines.append(
                        f"    {label} {ch['number']}: {ch['title']}"
                    )

            # Any chapters without a unit
            orphans = unit_map.get(None, [])
            for ch in orphans:
                listing_lines.append(
                    f"  {label} {ch['number']}: {ch['title']}"
                )

            listing = "\n".join(listing_lines)
            parts.append(
                f"\nThis book is organized into {len(units)} units, "
                f"each containing {chapter_label}s. "
                f"Here is the complete structure:{listing}\n"
                f"When students ask about the book's contents, use this structure."
            )
        else:
            listing = "\n".join(
                f"  {label} {ch['number']}: {ch['title']}" for ch in chapters
            )
            parts.append(
                f"\nThis book is organized into {chapter_label}s. "
                f"Here is the complete list:\n{listing}\n"
                f"When students ask about the book's contents, use this list."
            )
    return "\n".join(parts)

# Template for RAG queries
RAG_PROMPT_TEMPLATE = """Context from CBSE Grade 5 textbooks:
---
{context}
---

Student's question: {question}

Please answer the question based on the context above. If the answer isn't in the context, let the student know."""

# Template for quiz generation
QUIZ_PROMPT_TEMPLATE = """Context from CBSE Grade 5 textbooks:
---
{context}
---

Topic requested: {topic}

Create a quiz with {num_questions} multiple choice questions STRICTLY about the
topic above. Use ONLY information from the provided context that is directly
related to "{topic}". Do NOT include questions about the preface, foreword,
book overview, or other chapters/topics.

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
PRACTICE_PROMPT_TEMPLATE = """Context from CBSE Grade 5 textbooks:
---
{context}
---

Topic requested: {topic}

Create {num_problems} practice problems STRICTLY about "{topic}" for a
Class 5 student. Use ONLY information from the provided context that is
directly related to this topic. Do NOT create problems about the book
overview, other chapters, or unrelated topics.

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
