#!/usr/bin/env python3
"""
Ingestion Script - Process all PDF files and store in per-subject vector databases.

This script orchestrates the full ingestion pipeline:
1. Discover all subject subdirectories under cbse-books/
2. Extract chapter/unit lists from each book's preface PDF (*ps.pdf)
3. Parse all PDF files from every subject
4. Split text into chunks
5. Generate embeddings for each chunk
6. Store in **per-subject** ChromaDB collections
7. Write a books.json manifest with metadata + chapter listings

Run this script ONCE to populate the vector databases:
    python scripts/ingest_books.py

The script is idempotent - running it again will clear and rebuild each collection.
"""

import json
import re
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from maths_tutor.config import (
    BOOKS_DIR,
    BOOKS_MANIFEST_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_PREFIX,
    DATA_DIR,
)
from maths_tutor.embeddings.embedder import Embedder
from maths_tutor.embeddings.vector_store import VectorStore
from maths_tutor.ingestion.chunker import TextChunker
from maths_tutor.ingestion.pdf_parser import PDFParser

# Rich console for beautiful output
console = Console()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _extract_subject_from_dirname(dirname: str) -> str:
    """
    Extract a human-readable subject name from a directory name.

    Examples:
        'cbse-grade-5-maths' -> 'maths'
        'cbse-grade-5-theWorldAroundUs' -> 'the world around us'
        'cbse-grade-5-physicalEducationAndWellbeing' -> 'physical education and wellbeing'
    """
    match = re.match(r"cbse-grade-\d+-(.+)", dirname)
    if not match:
        return dirname
    raw = match.group(1)
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", raw).lower()
    return words


def _subject_id_from_name(subject_name: str) -> str:
    """
    Convert a human-readable subject name to a stable identifier.

    Examples:
        'maths' -> 'maths'
        'the world around us' -> 'the_world_around_us'
        'physical education and wellbeing' -> 'physical_education_and_wellbeing'
    """
    return subject_name.strip().lower().replace(" ", "_")


def _extract_book_title(pdf_path: Path, parser: PDFParser) -> str:
    """
    Extract the book title from the first page of a preface/*ps.pdf file.
    """
    try:
        doc_content = parser.parse_pdf(pdf_path)
        if not doc_content.pages:
            return ""
        first_page = doc_content.pages[0].text.strip()
        for line in first_page.split("\n"):
            line = line.strip()
            if len(line) < 5 or line.isdigit():
                continue
            return line
    except Exception:
        pass
    return ""


def _detect_content_type(text: str, source_file: str) -> str:
    """
    Heuristically classify a chunk's content type.

    Returns one of: 'table_of_contents', 'preface', or 'chapter'.
    """
    lower = text.lower()
    is_preface_file = source_file.endswith("ps.pdf")

    unit_count = lower.count("unit ")
    has_contents_label = "contents" in lower
    has_numbered_list = bool(re.search(r"\d+\.\s+\S+.*\d+$", text, re.MULTILINE))

    if (unit_count >= 2 or has_contents_label) and has_numbered_list:
        return "table_of_contents"
    if is_preface_file:
        return "preface"
    return "chapter"


# =============================================================================
# CHAPTER / UNIT EXTRACTION
# =============================================================================


def _clean_toc_text(raw: str) -> str:
    """
    Clean raw PDF text for TOC parsing.

    Removes control characters, dot-leader patterns, and normalises
    whitespace while preserving meaningful Unicode (em-dashes, smart
    quotes, etc.).
    """
    # Replace thin-space (\u2009), NBSP (\xa0) with regular space
    cleaned = raw.replace("\u2009", " ").replace("\xa0", " ")
    # Remove control characters (except space, tab, newline)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)
    # Remove dot-leader sequences (middle dots, block chars, placeholder dots)
    cleaned = re.sub(r"[·█■…]+", " ", cleaned)
    # Remove runs of the Unicode replacement char
    cleaned = re.sub(r"\ufffd+", " ", cleaned)
    # Collapse multiple spaces/tabs
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned


def _clean_title(raw: str) -> str:
    """
    Clean a raw chapter title extracted from PDF text.

    Preserves meaningful characters (em-dashes, smart quotes/apostrophes)
    while removing control characters and artifacts.
    """
    # Replace thin-space (\u2009), NBSP (\xa0) with regular space
    cleaned = raw.replace("\u2009", " ").replace("\xa0", " ")
    # Normalise smart quotes/apostrophes to ASCII
    cleaned = cleaned.replace("\u2018", "'").replace("\u2019", "'")
    cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')
    # Remove control characters (except space)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)
    # Remove dot-leader characters
    cleaned = re.sub(r"[·█■…]+", "", cleaned)
    # Remove Unicode replacement char
    cleaned = re.sub(r"\ufffd+", "", cleaned)
    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Strip trailing page numbers (e.g. "Title 42")
    cleaned = re.sub(r"\s+\d+\s*$", "", cleaned)
    # Strip trailing/leading punctuation noise (but keep !, ?, ', ")
    cleaned = cleaned.strip(".,;:-")
    return cleaned.strip()


def _is_section_divider(first_page_text: str) -> bool:
    """
    Detect arts-style section divider pages (e.g. V I S U A L A R T S).

    These pages have mostly single-character lines.
    """
    lines = [l.strip() for l in first_page_text.split("\n") if l.strip()]
    if not lines:
        return True
    single_char_lines = sum(1 for l in lines if len(l) <= 2)
    return single_char_lines >= len(lines) * 0.6


def _is_non_chapter_pdf(pdf_name: str) -> bool:
    """
    Return True for PDFs that are known to NOT be chapter content.

    Patterns:
      *ps.pdf  - preface / title page
      *wc.pdf  - warm-up / cool-down appendix
    """
    return pdf_name.endswith("ps.pdf") or pdf_name.endswith("wc.pdf")


def _extract_chapter_number_from_filename(pdf_name: str) -> int | None:
    """
    Extract the chapter sequence number from the PDF filename.

    Pattern: ee<subj>1<NN>.pdf  ->  NN
    Examples: eemm101.pdf -> 1, eemm115.pdf -> 15, eesa102.pdf -> 2
    """
    m = re.match(r"[a-z]+1(\d{2})\.pdf$", pdf_name)
    if m:
        return int(m.group(1))
    return None


def _extract_title_from_first_page(first_page_text: str) -> str:
    """
    Extract the chapter/unit title from the first page of a chapter PDF.

    Handles various patterns across subjects:
    - English: "Let us Recite\\n1 Papa's Spectacles" or "1 Papa's Spectacles"
    - Maths:   "Chapter\\nChapter\\nFar and Near" or just "Fractions"
    - Arts:    "Chapter 16\\nDANCING WITH RHYTHM AND TEMPOS" or content start
    - EVS:     "Unit 1\\nAbout the Unit" or "Journey of a River"
    - PE:      "UNIT 1\\nBasic Motor Movements"
    """
    lines = [l.strip() for l in first_page_text.split("\n") if l.strip()]
    if not lines:
        return ""

    # Collect the first several meaningful lines (skip very short noise)
    candidate_lines: list[str] = []
    for line in lines[:12]:
        cleaned = _clean_title(line)
        if len(cleaned) < 2:
            continue
        candidate_lines.append(cleaned)
    if not candidate_lines:
        return ""

    # Strategy 1: Look for explicit "Unit N Title" or "Chapter N TITLE" pattern
    for line in candidate_lines[:6]:
        m = re.match(r"(?i)(?:unit|chapter)\s+(\d+)\s+(.+)", line)
        if m:
            title = m.group(2).strip()
            if title and len(title) > 2:
                return _clean_title(title)

    # Strategy 2: Look for "N Title" pattern (English books)
    # e.g. "1 Papa's Spectacles", "10 Glass Bangles"
    for line in candidate_lines[:4]:
        m = re.match(r"^(\d{1,2})\s+([A-Z].+)", line)
        if m:
            title = m.group(2).strip()
            if title and len(title) > 2:
                return _clean_title(title)

    # Strategy 3: Skip filler lines and grab the first substantive title
    # Skip lines that are just "Chapter", "Let us Recite", "Let us Read",
    # "About the Unit", "Unit N", etc.
    skip_patterns = re.compile(
        r"^(?:chapter|let us (?:recite|read)|about the unit|unit \d+|note (?:for|to) teachers?:?)$",
        re.IGNORECASE,
    )
    for line in candidate_lines:
        if skip_patterns.match(line):
            continue
        # Skip lines that are just a number
        if line.isdigit():
            continue
        # Skip very short lines (likely artifacts)
        if len(line) < 4:
            continue
        # Skip lines starting with "(1)" or similar footnote markers
        if re.match(r"^\(\d+\)", line):
            continue
        return _clean_title(line)

    # Fallback: return the first candidate
    return _clean_title(candidate_lines[0]) if candidate_lines else ""


def extract_chapters_from_pdfs(
    pdf_files: list[Path], parser: PDFParser
) -> tuple[str, list[dict]]:
    """
    Build the chapter list by scanning the first page of every chapter PDF.

    Skips non-chapter PDFs (preface, appendix, section dividers).
    Detects whether the book uses "Unit" or "Chapter" terminology.

    Args:
        pdf_files: Sorted list of PDF paths for a single subject.
        parser: PDFParser instance.

    Returns:
        Tuple of (chapter_label, chapters) where:
        - chapter_label is 'unit' or 'chapter'
        - chapters is a list of dicts: [{"number": int, "title": str}]
    """
    chapters: list[dict] = []
    unit_count = 0
    chapter_count = 0

    for pdf_path in pdf_files:
        # Skip known non-chapter PDFs
        if _is_non_chapter_pdf(pdf_path.name):
            continue

        try:
            doc_content = parser.parse_pdf(pdf_path)
        except Exception:
            continue

        if not doc_content.pages:
            continue

        first_page = doc_content.pages[0].text.strip()

        # Skip section divider pages (arts: V I S U A L A R T S, etc.)
        if _is_section_divider(first_page):
            continue

        # Count terminology across all chapter PDFs
        page_lower = first_page.lower()
        if re.search(r"\bunit\s+\d+", page_lower):
            unit_count += 1
        if re.search(r"\bchapter\s+\d+", page_lower):
            chapter_count += 1

        # Extract title
        title = _extract_title_from_first_page(first_page)
        if not title:
            continue

        # Get chapter number from filename (most reliable)
        ch_num = _extract_chapter_number_from_filename(pdf_path.name)
        if ch_num is None:
            continue

        chapters.append({"number": ch_num, "title": title})

    # Determine terminology
    chapter_label = "unit" if unit_count > chapter_count else "chapter"

    # Sort by chapter number
    chapters.sort(key=lambda c: c["number"])

    return (chapter_label, chapters)


def _find_toc_pages(doc_content) -> list[int]:
    """
    Find page indices that belong to the Table of Contents.

    The NCERT preface PDFs have a "Contents" label on (or near) the TOC
    page.  The label may appear at the top or bottom of the page.  The
    TOC can span up to 2 pages (e.g. arts).

    Returns a sorted list of 0-based page indices.
    """
    pages = doc_content.pages
    toc_indices: list[int] = []

    for i, page in enumerate(pages):
        text = page.text.strip()
        # "Contents" as a standalone word (not inside a sentence)
        if re.search(r"(?m)^\s*Contents\s*$", text, re.IGNORECASE):
            toc_indices.append(i)

    if not toc_indices:
        return []

    # Include the page after the last "Contents" marker for multi-page TOCs
    result_set = set(toc_indices)
    last_toc = max(toc_indices)
    if last_toc + 1 < len(pages):
        result_set.add(last_toc + 1)

    return sorted(result_set)


def extract_chapter_list_from_toc(
    ps_pdf_path: Path, parser: PDFParser
) -> tuple[str, list[dict], list[dict]]:
    """
    Parse the table of contents from a preface (*ps.pdf) file.

    First locates the actual Contents page(s) in the PDF, then parses
    ONLY those pages.  This avoids false matches from committee member
    lists, inline text references, and other numbered content that
    appears elsewhere in the preface.

    Captures the unit-chapter hierarchy:
      - PE:      Unit 1 — Basic Motor Movements  >  Chapter 1: Throwing and Catching
      - English: Unit 1: Let's Have Fun           >  1. Papa's Spectacles
      - Arts:    VISUAL ARTS (section)            >  1. Objects on the Move
      - EVS:     Unit 1: Life Around Us           >  Chapter 1: Water — The Essence of Life
      - Maths:   (no units, just chapters)

    Args:
        ps_pdf_path: Path to the *ps.pdf preface file.
        parser: PDFParser instance.

    Returns:
        Tuple of (chapter_label, chapters, units) where:
        - chapter_label is 'unit' or 'chapter'
        - chapters is a list of dicts:
            [{"number": int, "title": str, "unit_number": int|None, "unit_title": str|None}]
        - units is a list of dicts:
            [{"number": int, "title": str}]
    """
    try:
        doc_content = parser.parse_pdf(ps_pdf_path)
    except Exception:
        return ("chapter", [], [])

    # ── Locate TOC pages ───────────────────────────────────────────
    toc_page_indices = _find_toc_pages(doc_content)
    if not toc_page_indices:
        return ("chapter", [], [])

    # Combine ONLY the TOC pages
    raw_text = "\n".join(
        doc_content.pages[i].text
        for i in toc_page_indices
        if i < len(doc_content.pages) and doc_content.pages[i].text.strip()
    )
    toc_text = _clean_toc_text(raw_text)

    # ── Join continuation lines ────────────────────────────────────
    # Some TOC entries span two lines (e.g. arts chapter 18):
    #   18. My Dance Expresses Emotions and
    #       Narrates Stories  158
    joined_lines: list[str] = []
    entry_pattern = re.compile(
        r"^\s*(?:"
        r"\d{1,2}\.\s"           # "1. Title" numbered entry
        r"|[Cc]hapter\s+\d"     # "Chapter N" entry
        r"|[Uu]nit\s+\d"        # "Unit N" entry
        r"|[A-Z]{3,}\s"         # ALL-CAPS section headers (VISUAL ARTS, THEATRE, etc.)
        r"|Foreword|About|Contents|Self-assessment|Learning"
        r"|Session|Warm|Dear|Time|Gandhiji|Reprint|Annual"
        r"|[ivxlc]+$"           # Roman numeral page numbers (iv, vi, xxi)
        r")"
    )
    for line in toc_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if joined_lines and not entry_pattern.match(stripped):
            joined_lines[-1] = joined_lines[-1] + " " + stripped
        else:
            joined_lines.append(stripped)
    toc_text = "\n".join(joined_lines)

    # ── Parse units and chapters line by line ──────────────────────
    chapters: list[dict] = []
    units: list[dict] = []
    current_unit_num: int | None = None
    current_unit_title: str | None = None

    # Regex for unit headers:
    #   "Unit 1 — Basic Motor Movements"
    #   "Unit 1: Let's Have Fun"
    re_unit = re.compile(
        r"[Uu]nit\s+(\d+)\s*[:\u2014\u2013—–-]+\s*(.+)"
    )
    # Regex for ALL-CAPS section headers (arts: VISUAL ARTS, THEATRE, etc.)
    re_section = re.compile(r"^([A-Z]{3,}(?:\s+[A-Z]{3,})*)\s*\d*$")
    # Regex for "Chapter N: Title"
    re_chapter = re.compile(r"[Cc]hapter\s+(\d+)\s*:\s*(.+)")
    # Regex for "N. Title"
    re_numbered = re.compile(r"^\s*(\d{1,2})\.\s+([A-Z].+)")

    # Auto-number for ALL-CAPS sections (arts has no numeric unit numbers)
    section_counter = 0

    for line in toc_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        # Check for unit header
        m_unit = re_unit.search(stripped)
        if m_unit:
            current_unit_num = int(m_unit.group(1))
            current_unit_title = _clean_title(m_unit.group(2))
            if current_unit_title:
                units.append({"number": current_unit_num, "title": current_unit_title})
            continue

        # Check for ALL-CAPS section header (arts)
        m_section = re_section.match(stripped)
        if m_section:
            section_counter += 1
            current_unit_num = section_counter
            current_unit_title = _clean_title(m_section.group(1).title())
            units.append({"number": current_unit_num, "title": current_unit_title})
            continue

        # Check for "Chapter N: Title"
        m_ch = re_chapter.search(stripped)
        if m_ch:
            num = int(m_ch.group(1))
            title = _clean_title(m_ch.group(2))
            if title and len(title) > 1 and num <= 30:
                chapters.append({
                    "number": num,
                    "title": title,
                    "unit_number": current_unit_num,
                    "unit_title": current_unit_title,
                })
            continue

        # Check for "N. Title"
        m_num = re_numbered.match(stripped)
        if m_num:
            num = int(m_num.group(1))
            title = _clean_title(m_num.group(2))
            if title and len(title) > 1 and num <= 30:
                chapters.append({
                    "number": num,
                    "title": title,
                    "unit_number": current_unit_num,
                    "unit_title": current_unit_title,
                })
            continue

    # Deduplicate by chapter number (keep first occurrence)
    seen_nums: set[int] = set()
    deduped: list[dict] = []
    for ch in chapters:
        if ch["number"] not in seen_nums:
            seen_nums.add(ch["number"])
            deduped.append(ch)
    deduped.sort(key=lambda c: c["number"])

    # Deduplicate units
    seen_unit_nums: set[int] = set()
    deduped_units: list[dict] = []
    for u in units:
        if u["number"] not in seen_unit_nums:
            seen_unit_nums.add(u["number"])
            deduped_units.append(u)
    deduped_units.sort(key=lambda u: u["number"])

    # Determine chapter vs unit terminology for the individual entries.
    # When units exist, the individual items are always "chapters" (the
    # grouped containers are "units").  When there are no units, fall back
    # to whichever keyword is more frequent in the TOC text.
    if deduped_units:
        chapter_label = "chapter"
    else:
        unit_mentions = len(re.findall(r"(?i)\bunit\s+\d+", toc_text))
        chapter_mentions = len(re.findall(r"(?i)\bchapter\s+\d+", toc_text))
        chapter_label = "unit" if unit_mentions > chapter_mentions else "chapter"

    return (chapter_label, deduped, deduped_units)


# =============================================================================
# SUBJECT DISCOVERY
# =============================================================================


def discover_subjects(books_dir: Path) -> list[dict]:
    """
    Discover all subject directories and their PDF files.

    Supports both flat and nested layouts:
      Nested (current): cbse-books/cbse-grade-5/cbse-grade-5-maths/*.pdf
      Flat   (legacy):  cbse-books/cbse-grade-5-maths/*.pdf

    Returns:
        List of dicts with keys: subject, subject_id, directory, pdf_files
    """
    subjects = []
    if not books_dir.exists():
        return subjects

    # Collect candidate directories: direct children + one-level-deeper children
    candidate_dirs: list[Path] = []
    for child in sorted(books_dir.iterdir()):
        if not child.is_dir():
            continue
        # If this child matches the grade-grouping pattern (e.g. cbse-grade-5/)
        # look inside it for subject directories.
        if re.match(r"cbse-grade-\d+$", child.name):
            for grandchild in sorted(child.iterdir()):
                if grandchild.is_dir():
                    candidate_dirs.append(grandchild)
        else:
            candidate_dirs.append(child)

    for subdir in candidate_dirs:
        pdf_files = sorted(subdir.glob("*.pdf"))
        if pdf_files:
            subject_name = _extract_subject_from_dirname(subdir.name)
            subjects.append(
                {
                    "subject": subject_name,
                    "subject_id": _subject_id_from_name(subject_name),
                    "directory": subdir,
                    "pdf_files": pdf_files,
                }
            )
    return subjects


def discover_book_titles(subjects: list[dict], parser: PDFParser) -> dict[str, str]:
    """
    Scan each subject's *ps.pdf preface file to extract the book title.

    Returns:
        Mapping of subject_id -> book title (e.g. "english" -> "Santoor")
    """
    titles: dict[str, str] = {}
    for subj in subjects:
        for pdf_path in subj["pdf_files"]:
            if pdf_path.name.endswith("ps.pdf"):
                title = _extract_book_title(pdf_path, parser)
                if title:
                    titles[subj["subject_id"]] = title
                break
    return titles


def discover_chapter_lists(
    subjects: list[dict], parser: PDFParser
) -> dict[str, tuple[str, list[dict], list[dict]]]:
    """
    Extract chapter/unit lists for each subject.

    Strategy:
    1. Try parsing the Table of Contents from the preface PDF (*ps.pdf).
       This is the most reliable source — all NCERT preface PDFs include
       a Contents page with properly numbered chapter/unit titles.
    2. If the TOC extraction yields no chapters, fall back to scanning the
       first page of each individual chapter PDF.

    Returns:
        Mapping of subject_id -> (chapter_label, chapters_list, units_list)
    """
    result: dict[str, tuple[str, list[dict], list[dict]]] = {}
    for subj in subjects:
        label, chapters, units = "chapter", [], []

        # Strategy 1: Parse TOC from the preface PDF
        for pdf_path in subj["pdf_files"]:
            if pdf_path.name.endswith("ps.pdf"):
                label, chapters, units = extract_chapter_list_from_toc(
                    pdf_path, parser
                )
                break

        # Strategy 2: Fall back to first-page extraction from chapter PDFs
        if not chapters:
            label, chapters = extract_chapters_from_pdfs(subj["pdf_files"], parser)
            units = []

        result[subj["subject_id"]] = (label, chapters, units)
    return result


# =============================================================================
# HUMAN-READABLE SUBJECT DISPLAY NAME
# =============================================================================

_SUBJECT_DISPLAY_NAMES: dict[str, str] = {
    "maths": "Mathematics",
    "english": "English",
    "arts": "Arts",
    "the_world_around_us": "The World Around Us",
    "physical_education_and_wellbeing": "Physical Education & Wellbeing",
}


def _display_name(subject_id: str) -> str:
    return _SUBJECT_DISPLAY_NAMES.get(subject_id, subject_id.replace("_", " ").title())


# =============================================================================
# MAIN INGESTION
# =============================================================================


def ingest_books(clear_existing: bool = True) -> dict:
    """
    Main ingestion function that processes all PDFs across all subjects.

    Creates one ChromaDB collection per subject and writes a books.json manifest.

    Args:
        clear_existing: If True, clear existing collections before ingesting

    Returns:
        Dictionary with ingestion statistics
    """
    stats = {
        "pdfs_processed": 0,
        "total_pages": 0,
        "total_chunks": 0,
        "subjects_processed": 0,
        "subjects": [],
        "errors": [],
    }

    console.print(
        Panel.fit(
            "[bold blue]CBSE Grade 5 - Per-Subject Ingestion[/bold blue]\n"
            "Processing PDF files into per-subject ChromaDB collections",
            border_style="blue",
        )
    )

    if not BOOKS_DIR.exists():
        console.print(f"[red]Error: Books directory not found: {BOOKS_DIR}[/red]")
        return stats

    # Discover subjects
    subjects = discover_subjects(BOOKS_DIR)
    if not subjects:
        console.print(f"[red]Error: No subject directories with PDFs found in {BOOKS_DIR}[/red]")
        return stats

    # Show discovered subjects
    subject_table = Table(title="Discovered Subjects", show_header=True, header_style="bold cyan")
    subject_table.add_column("Subject", style="white")
    subject_table.add_column("ID", style="dim")
    subject_table.add_column("Collection", style="dim")
    subject_table.add_column("PDFs", style="green", justify="right")

    total_pdfs = 0
    for subj in subjects:
        collection_name = f"{COLLECTION_PREFIX}{subj['subject_id']}"
        subject_table.add_row(
            subj["subject"].title(),
            subj["subject_id"],
            collection_name,
            str(len(subj["pdf_files"])),
        )
        total_pdfs += len(subj["pdf_files"])

    console.print(subject_table)
    console.print(f"\n[cyan]Total: {len(subjects)} subjects, {total_pdfs} PDF files[/cyan]")
    console.print(f"[cyan]Chunk size: {CHUNK_SIZE} chars, Overlap: {CHUNK_OVERLAP} chars[/cyan]\n")

    # Initialize shared components
    console.print("[yellow]Initializing components...[/yellow]")
    parser = PDFParser()
    chunker = TextChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    embedder = Embedder()

    # Discover book titles and chapter lists
    book_titles = discover_book_titles(subjects, parser)
    chapter_data = discover_chapter_lists(subjects, parser)

    if book_titles:
        console.print("[cyan]Detected book titles:[/cyan]")
        for sid, title in book_titles.items():
            console.print(f"  {_display_name(sid)}: {title}")

    if chapter_data:
        console.print("\n[cyan]Detected chapters/units:[/cyan]")
        for sid, (label, chapters, units) in chapter_data.items():
            unit_info = f" in {len(units)} units" if units else ""
            console.print(
                f"  {_display_name(sid)}: {len(chapters)} {label}s{unit_info}"
            )

    # Books manifest data (will be written to books.json)
    books_manifest: list[dict] = []

    # Process each subject into its own collection
    for subj in subjects:
        subject_name = subj["subject"]
        subject_id = subj["subject_id"]
        pdf_files = subj["pdf_files"]
        book_title = book_titles.get(subject_id, "")
        chapter_label, chapters, units = chapter_data.get(
            subject_id, ("chapter", [], [])
        )
        collection_name = f"{COLLECTION_PREFIX}{subject_id}"

        # Build the preamble that will be prepended to every chunk
        preamble = f"[{subject_name.title()}"
        if book_title:
            preamble += f" - {book_title}"
        preamble += " (Grade 5)]"

        console.print(
            f"\n[bold magenta]--- {subject_name.title()} ({len(pdf_files)} PDFs) "
            f"-> collection: {collection_name} ---[/bold magenta]"
        )

        # Create per-subject vector store
        vector_store = VectorStore(collection_name=collection_name)

        # Clear existing data for this subject if requested
        if clear_existing and vector_store.count > 0:
            console.print(
                f"[yellow]  Clearing existing collection ({vector_store.count} documents)...[/yellow]"
            )
            vector_store.delete_collection()
            vector_store = VectorStore(collection_name=collection_name)

        # Collect chunks for this subject
        subject_chunks: list[str] = []
        subject_metadatas: list[dict] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            pdf_task = progress.add_task(
                f"[cyan]Processing {subject_name}...", total=len(pdf_files)
            )

            for pdf_path in pdf_files:
                progress.update(pdf_task, description=f"[cyan]Parsing {pdf_path.name}...")

                try:
                    doc_content = parser.parse_pdf(pdf_path)
                    stats["pdfs_processed"] += 1
                    stats["total_pages"] += doc_content.total_pages

                    for page in doc_content.pages:
                        if not page.text.strip():
                            continue

                        page_chunks = chunker.chunk_text(
                            page.text,
                            metadata={
                                "source_file": pdf_path.name,
                                "page_number": page.page_number,
                            },
                        )

                        for chunk in page_chunks:
                            content_type = _detect_content_type(chunk.text, pdf_path.name)
                            enriched_text = f"{preamble}\n{chunk.text}"

                            subject_chunks.append(enriched_text)
                            subject_metadatas.append(
                                {
                                    "source_file": pdf_path.name,
                                    "page_number": page.page_number,
                                    "chunk_index": chunk.chunk_index,
                                    "char_count": len(enriched_text),
                                    "subject": subject_name,
                                    "book_title": book_title,
                                    "content_type": content_type,
                                }
                            )

                except Exception as e:
                    stats["errors"].append(f"{pdf_path.name}: {str(e)}")
                    console.print(f"[red]Error processing {pdf_path.name}: {e}[/red]")

                progress.advance(pdf_task)

        if not subject_chunks:
            console.print(f"[yellow]  No chunks extracted for {subject_name}[/yellow]")
            stats["subjects_processed"] += 1
            stats["subjects"].append(subject_name)
            continue

        # Generate embeddings for this subject
        console.print(
            f"  [yellow]Embedding {len(subject_chunks)} chunks...[/yellow]"
        )

        all_embeddings: list[list[float]] = []
        batch_size = 32
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            embed_task = progress.add_task(
                f"[cyan]Embedding {subject_name}...", total=len(subject_chunks)
            )
            for i in range(0, len(subject_chunks), batch_size):
                batch = subject_chunks[i : i + batch_size]
                batch_embeddings = embedder.embed_batch(batch, show_progress=False)
                all_embeddings.extend(batch_embeddings)
                progress.advance(embed_task, len(batch))

        # Store in per-subject collection
        console.print(f"  [yellow]Storing in collection: {collection_name}...[/yellow]")
        ids = [f"{subject_id}_chunk_{i:06d}" for i in range(len(subject_chunks))]
        vector_store.add_documents(
            texts=subject_chunks,
            embeddings=all_embeddings,
            metadatas=subject_metadatas,
            ids=ids,
        )

        console.print(
            f"  [green]{len(subject_chunks)} chunks stored in {collection_name}[/green]"
        )

        stats["total_chunks"] += len(subject_chunks)
        stats["subjects_processed"] += 1
        stats["subjects"].append(subject_name)

        # Build manifest entry for this book
        manifest_entry: dict = {
            "id": subject_id,
            "title": book_title,
            "subject": _display_name(subject_id),
            "collection_name": collection_name,
            "chapter_label": chapter_label,
            "chapters": chapters,
            "chunk_count": len(subject_chunks),
            "pdf_count": len(pdf_files),
        }
        if units:
            manifest_entry["units"] = units
        books_manifest.append(manifest_entry)

    # Write books.json manifest
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"books": books_manifest}
    BOOKS_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    console.print(f"\n[green]Books manifest written to {BOOKS_MANIFEST_PATH}[/green]")

    # Print summary
    console.print("\n")
    table = Table(title="Ingestion Summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="white")
    table.add_column("Value", style="green")

    table.add_row("Subjects Processed", str(stats["subjects_processed"]))
    table.add_row("Subjects", ", ".join(s.title() for s in stats["subjects"]))
    table.add_row("PDFs Processed", str(stats["pdfs_processed"]))
    table.add_row("Total Pages", str(stats["total_pages"]))
    table.add_row("Total Chunks", str(stats["total_chunks"]))
    table.add_row("Errors", str(len(stats["errors"])))

    # Per-subject breakdown
    console.print()
    breakdown = Table(title="Per-Subject Breakdown", show_header=True, header_style="bold cyan")
    breakdown.add_column("Subject", style="white")
    breakdown.add_column("Collection", style="dim")
    breakdown.add_column("Chunks", style="green", justify="right")
    breakdown.add_column("Chapters", style="yellow", justify="right")

    for book in books_manifest:
        breakdown.add_row(
            book["subject"],
            book["collection_name"],
            str(book["chunk_count"]),
            str(len(book["chapters"])),
        )

    console.print(table)
    console.print(breakdown)

    if stats["errors"]:
        console.print("\n[red]Errors encountered:[/red]")
        for error in stats["errors"]:
            console.print(f"  [red]* {error}[/red]")

    console.print("\n[bold green]Ingestion complete![/bold green]")
    console.print("[dim]You can now use the chatbot to ask questions about the textbooks.[/dim]")

    return stats


def verify_ingestion() -> None:
    """
    Verify that ingestion was successful by running a test query per subject.
    """
    console.print("\n[yellow]Verifying ingestion...[/yellow]")

    if not BOOKS_MANIFEST_PATH.exists():
        console.print("[red]books.json not found! Run ingestion first.[/red]")
        return

    manifest = json.loads(BOOKS_MANIFEST_PATH.read_text())
    embedder = Embedder()
    test_query = "What topics are covered in this textbook?"
    query_embedding = embedder.embed(test_query)

    for book in manifest["books"]:
        collection_name = book["collection_name"]
        vector_store = VectorStore(collection_name=collection_name)

        if vector_store.count == 0:
            console.print(f"[red]{book['subject']}: collection is empty![/red]")
            continue

        results = vector_store.search(query_embedding, top_k=2)
        console.print(
            f"\n[cyan]{book['subject']} ({collection_name}): "
            f"{vector_store.count} chunks, {len(book['chapters'])} {book['chapter_label']}s[/cyan]"
        )
        for i, result in enumerate(results, 1):
            preview = result.text[:120].replace("\n", " ")
            console.print(f"  Result {i} (score: {result.score:.4f}): {preview}...")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest CBSE Grade 5 book PDFs into per-subject vector databases"
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="Only verify existing ingestion"
    )
    parser.add_argument(
        "--no-clear", action="store_true", help="Don't clear existing data before ingesting"
    )

    args = parser.parse_args()

    if args.verify_only:
        verify_ingestion()
    else:
        ingest_books(clear_existing=not args.no_clear)
        verify_ingestion()


if __name__ == "__main__":
    main()
