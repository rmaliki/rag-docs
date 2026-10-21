"""Keyword search with PostgreSQL full-text search.

Catches what embeddings miss: exact terms, codes, names, acronyms ("Q3", "FY26").
"""

from collections.abc import Collection

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.retrieval.common import ACL_FILTER, RetrievedChunk

# plainto_tsquery ANDs every term, which is too strict for natural-language questions;
# rewriting `&` to `|` matches any term and lets ts_rank_cd reward chunks matching more.
_KEYWORD_SEARCH = text(
    f"""
    WITH q AS (
        SELECT to_tsquery('english', replace(plainto_tsquery('english', :query)::text, '&', '|')) AS query
    )
    SELECT c.id, c.document_id, d.title, c.chunk_index, c.page, c.content,
           ts_rank_cd(c.search_vector, q.query) AS rank
    FROM document_chunks c
    JOIN documents d ON d.id = c.document_id
    CROSS JOIN q
    WHERE c.search_vector @@ q.query
      AND {ACL_FILTER}
    ORDER BY rank DESC
    LIMIT :limit
    """
)


async def keyword_search(
    session: AsyncSession, query: str, principals: Collection[str], limit: int
) -> list[RetrievedChunk]:
    result = await session.execute(
        _KEYWORD_SEARCH, {"query": query, "principals": sorted(principals), "limit": limit}
    )
    return [
        RetrievedChunk(
            chunk_id=row.id,
            document_id=row.document_id,
            title=row.title,
            chunk_index=row.chunk_index,
            page=row.page,
            content=row.content,
            keyword_rank=float(row.rank),
        )
        for row in result
    ]
