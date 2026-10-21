"""Semantic search with pgvector (cosine distance over normalised embeddings)."""

from collections.abc import Collection, Sequence

from pgvector.sqlalchemy import Vector
from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.retrieval.common import ACL_FILTER, RetrievedChunk

_VECTOR_SEARCH = text(
    f"""
    SELECT c.id, c.document_id, d.title, c.chunk_index, c.page, c.content,
           c.embedding <=> CAST(:query_embedding AS vector) AS distance
    FROM document_chunks c
    JOIN documents d ON d.id = c.document_id
    WHERE {ACL_FILTER}
    ORDER BY distance
    LIMIT :limit
    """
).bindparams(bindparam("query_embedding", type_=Vector()))


async def vector_search(
    session: AsyncSession,
    query_embedding: Sequence[float],
    principals: Collection[str],
    limit: int,
) -> list[RetrievedChunk]:
    result = await session.execute(
        _VECTOR_SEARCH,
        {"query_embedding": list(query_embedding), "principals": sorted(principals), "limit": limit},
    )
    return [
        RetrievedChunk(
            chunk_id=row.id,
            document_id=row.document_id,
            title=row.title,
            chunk_index=row.chunk_index,
            page=row.page,
            content=row.content,
            vector_distance=float(row.distance),
        )
        for row in result
    ]
