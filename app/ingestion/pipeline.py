"""Ingestion: extract -> clean -> chunk -> embed -> store, independent of user questions."""

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.database.repositories import DocumentRepository
from app.database.session import Database
from app.ingestion.chunk import Chunk, chunk_pages
from app.ingestion.clean import clean_text
from app.ingestion.embed import Embedder
from app.ingestion.extract import ExtractedPage, UnsupportedDocumentError, extract
from app.ingestion.metadata import DocumentMetadata

logger = logging.getLogger(__name__)


class IngestionError(Exception):
    pass


@dataclass(frozen=True)
class PreparedDocument:
    chunks: list[Chunk]
    embeddings: list[list[float]]


@dataclass(frozen=True)
class IngestionResult:
    document_id: str
    status: Literal["ingested", "metadata_updated", "unchanged"]
    chunk_count: int | None = None


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_document(path: Path, embedder: Embedder, *, chunk_size: int, chunk_overlap: int) -> PreparedDocument:
    """CPU-bound part of ingestion; run it in a worker thread from async code."""
    try:
        pages = [ExtractedPage(page.page, clean_text(page.text)) for page in extract(path)]
    except UnsupportedDocumentError as exc:
        raise IngestionError(str(exc)) from exc
    chunks = chunk_pages(pages, chunk_size, chunk_overlap)
    if not chunks:
        raise IngestionError(f"{path.name}: no extractable text (scanned documents need OCR first)")
    return PreparedDocument(chunks=chunks, embeddings=embedder.embed_documents([c.text for c in chunks]))


class IngestionPipeline:
    def __init__(self, database: Database, embedder: Embedder, *, chunk_size: int, chunk_overlap: int):
        self._database = database
        self._embedder = embedder
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    async def ingest_file(
        self, path: Path, metadata: DocumentMetadata, *, source_uri: str | None = None
    ) -> IngestionResult:
        digest = await asyncio.to_thread(file_hash, path)

        # Same content: apply metadata/ACL changes only. Permission changes take effect
        # immediately without paying for re-extraction and re-embedding.
        async with self._database.admin_sessions() as session, session.begin():
            repository = DocumentRepository(session)
            existing = await repository.get(metadata.document_id)
            if existing is not None and existing.content_hash == digest:
                changed = repository.update_metadata(existing, metadata)
                status = "metadata_updated" if changed else "unchanged"
                logger.info("document %s", status, extra={"document_id": metadata.document_id})
                return IngestionResult(metadata.document_id, status)

        prepared = await asyncio.to_thread(
            prepare_document,
            path,
            self._embedder,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        async with self._database.admin_sessions() as session, session.begin():
            await DocumentRepository(session).replace(
                metadata,
                content_hash=digest,
                source_uri=source_uri or path.name,
                chunks=prepared.chunks,
                embeddings=prepared.embeddings,
            )
        logger.info(
            "document ingested",
            extra={"document_id": metadata.document_id, "chunks": len(prepared.chunks)},
        )
        return IngestionResult(metadata.document_id, "ingested", len(prepared.chunks))
