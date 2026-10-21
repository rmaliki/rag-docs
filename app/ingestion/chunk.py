"""Paragraph-aware chunking with a word-based size and overlap."""

from collections.abc import Sequence
from dataclasses import dataclass

from app.ingestion.extract import ExtractedPage


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    page: int | None


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Pack whole paragraphs into chunks of at most `size` words.

    Paragraphs longer than `size` are split with `overlap` words of overlap; consecutive
    chunks also share the last `overlap` words of the previous chunk when they fit.
    """
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")

    units: list[list[str]] = []
    for paragraph in text.split("\n\n"):
        words = paragraph.split()
        if not words:
            continue
        step = size - overlap
        for start in range(0, max(len(words) - overlap, 1), step):
            units.append(words[start : start + size])

    chunks: list[list[str]] = []
    window: list[str] = []
    for unit in units:
        if window and len(window) + len(unit) > size:
            chunks.append(window)
            tail = window[-overlap:] if overlap else []
            window = tail if len(tail) + len(unit) <= size else []
        window = window + unit
    if window:
        chunks.append(window)
    return [" ".join(words) for words in chunks]


def chunk_pages(pages: Sequence[ExtractedPage], size: int, overlap: int) -> list[Chunk]:
    """Chunk page by page so every chunk keeps a page number for citations."""
    chunks: list[Chunk] = []
    for page in pages:
        for text in chunk_text(page.text, size, overlap):
            chunks.append(Chunk(index=len(chunks), text=text, page=page.page))
    return chunks
