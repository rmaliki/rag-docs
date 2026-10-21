import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import text

from app.auth.permissions import Permissions
from app.config import Settings
from app.database.repositories import DocumentRepository
from app.database.session import Database
from app.ingestion.metadata import load_metadata
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.hybrid_search import HybridRetriever, fuse_results, reciprocal_rank_fusion
from app.retrieval.reranker import NoopReranker
from tests.conftest import DEV_SECRET, HashingEmbedder, make_chunk

# ---------------------------------------------------------------- unit tests


def test_rrf_ranks_chunks_found_by_both_retrievers_first():
    vector = [make_chunk(1, vector_distance=0.2), make_chunk(2, vector_distance=0.3)]
    keyword = [make_chunk(3, keyword_rank=0.5), make_chunk(2, keyword_rank=0.4)]

    fused = reciprocal_rank_fusion([vector, keyword])

    assert [chunk.chunk_id for chunk in fused][0] == 2
    assert len(fused) == 3  # chunk 2 is de-duplicated
    merged = next(chunk for chunk in fused if chunk.chunk_id == 2)
    assert (merged.vector_distance, merged.keyword_rank) == (0.3, 0.4)


def test_distant_vector_hits_are_not_evidence():
    assert fuse_results([make_chunk(1, vector_distance=0.95)], [], max_vector_distance=0.75, rrf_k=60) == []


def test_keyword_match_survives_a_distant_vector_hit():
    fused = fuse_results(
        [make_chunk(1, vector_distance=0.95)],
        [make_chunk(1, keyword_rank=0.3)],
        max_vector_distance=0.75,
        rrf_k=60,
    )
    assert [chunk.chunk_id for chunk in fused] == [1]


# ------------------------------------------------- integration tests (pgvector)
# Run against a THROWAWAY database: the fixture drops and recreates the schema.

ADMIN_URL = os.environ.get("RAG_TEST_ADMIN_DATABASE_URL")
APP_URL = os.environ.get("RAG_TEST_APP_DATABASE_URL")
ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "data" / "demo"

requires_database = pytest.mark.skipif(
    not (ADMIN_URL and APP_URL),
    reason="set RAG_TEST_ADMIN_DATABASE_URL and RAG_TEST_APP_DATABASE_URL",
)


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        database_url=APP_URL,
        admin_database_url=ADMIN_URL,
        auth_mode="dev",
        dev_jwt_secret=DEV_SECRET,
        warm_up_models=False,
    )


def _permissions(*groups: str, object_id: str = "u-test") -> Permissions:
    principals = {f"user:{object_id}", *(f"group:{group}" for group in groups)}
    return Permissions(user_id=object_id, principals=frozenset(principals))


def _pipeline(database: Database) -> IngestionPipeline:
    return IngestionPipeline(database, HashingEmbedder(), chunk_size=200, chunk_overlap=40)


@pytest.fixture(scope="module")
def demo_database():
    env = {**os.environ, "ADMIN_DATABASE_URL": ADMIN_URL}
    for command in (["downgrade", "base"], ["upgrade", "head"]):
        subprocess.run([sys.executable, "-m", "alembic", *command], cwd=ROOT, env=env, check=True)  # noqa: S603

    async def ingest_demo() -> None:
        database = Database(_settings())
        try:
            for path in sorted(DEMO_DIR.rglob("*.md")):
                await _pipeline(database).ingest_file(path, load_metadata(path))
        finally:
            await database.dispose()

    asyncio.run(ingest_demo())


async def _retrieve(question: str, permissions: Permissions) -> set[str]:
    database = Database(_settings())
    retriever = HybridRetriever(
        database, HashingEmbedder(), NoopReranker(), top_k=10, candidates=20, rrf_k=60, max_vector_distance=2.0
    )
    try:
        return {chunk.document_id for chunk in await retriever.retrieve(question, permissions)}
    finally:
        await database.dispose()


@pytest.mark.integration
@requires_database
async def test_marketing_cannot_retrieve_executive_salaries(demo_database):
    documents = await _retrieve("What are the executive salaries?", _permissions("all-employees", "marketing-team"))
    assert "doc-hr-002" not in documents
    assert documents <= {"doc-hr-001", "doc-mkt-001"}


@pytest.mark.integration
@requires_database
async def test_executive_can_retrieve_executive_salaries(demo_database):
    documents = await _retrieve("What are the executive salaries?", _permissions("all-employees", "executives"))
    assert "doc-hr-002" in documents


@pytest.mark.integration
@requires_database
async def test_row_level_security_blocks_unfiltered_queries(demo_database):
    database = Database(_settings())
    try:
        marketing = _permissions("all-employees", "marketing-team")
        async with database.authorized_session(marketing) as session:
            visible = set((await session.execute(text("SELECT DISTINCT document_id FROM document_chunks"))).scalars())
        async with database.app_sessions() as session:
            without_principals = (await session.execute(text("SELECT count(*) FROM document_chunks"))).scalar_one()
    finally:
        await database.dispose()

    assert visible == {"doc-hr-001", "doc-mkt-001"}
    assert without_principals == 0


@pytest.mark.integration
@requires_database
async def test_acl_change_applies_without_reingesting(demo_database):
    path = DEMO_DIR / "finance" / "q3-budget.md"
    original = load_metadata(path)
    widened = original.model_copy(update={"allowed_groups": (*original.allowed_groups, "marketing-team")})
    marketing = _permissions("marketing-team")

    async def visible_ids(database: Database) -> set[str]:
        async with database.authorized_session(marketing) as session:
            documents = await DocumentRepository(session).list_visible(sorted(marketing.principals))
        return {document.id for document in documents}

    database = Database(_settings())
    try:
        assert (await _pipeline(database).ingest_file(path, widened)).status == "metadata_updated"
        assert "doc-fin-001" in await visible_ids(database)
        assert (await _pipeline(database).ingest_file(path, original)).status == "metadata_updated"
        assert "doc-fin-001" not in await visible_ids(database)
    finally:
        await database.dispose()
