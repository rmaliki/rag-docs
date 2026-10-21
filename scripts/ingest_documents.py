"""Ingest every supported document under a directory. Each file needs a `<file>.meta.json` sidecar.

Usage: python -m scripts.ingest_documents data/demo
"""

import argparse
import asyncio
import sys
from pathlib import Path

from pydantic import ValidationError

from app.config import get_settings
from app.database.session import Database
from app.ingestion.embed import SentenceTransformerEmbedder
from app.ingestion.extract import SUPPORTED_EXTENSIONS
from app.ingestion.metadata import load_metadata
from app.ingestion.pipeline import IngestionError, IngestionPipeline
from app.monitoring.logging import configure_logging


async def ingest_directory(root: Path) -> int:
    settings = get_settings()
    configure_logging("WARNING")
    database = Database(settings)
    pipeline = IngestionPipeline(
        database,
        SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_dimension),
        chunk_size=settings.chunk_size_words,
        chunk_overlap=settings.chunk_overlap_words,
    )
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
    failures = 0
    try:
        for path in paths:
            relative = path.relative_to(root)
            try:
                result = await pipeline.ingest_file(path, load_metadata(path), source_uri=str(relative))
            except (IngestionError, ValidationError, FileNotFoundError) as exc:
                failures += 1
                print(f"{'failed':>16}  {relative}: {exc}", file=sys.stderr)
                continue
            chunks = f"{result.chunk_count} chunks" if result.chunk_count is not None else ""
            print(f"{result.status:>16}  {result.document_id:<14} {relative}  {chunks}")
    finally:
        await database.dispose()
    print(f"\n{len(paths) - failures}/{len(paths)} documents processed")
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    sys.exit(asyncio.run(ingest_directory(args.directory)))


if __name__ == "__main__":
    main()
