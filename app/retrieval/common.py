"""Types and SQL shared by the vector and keyword retrievers."""

from dataclasses import dataclass

# Application-level authorization filter, applied in every retrieval query.
# PostgreSQL row-level security enforces the same rule underneath (defense in depth).
# EXISTS rather than a JOIN on document_acl, so a chunk visible through several
# groups is returned once.
ACL_FILTER = """
    (
        d.is_public
        OR EXISTS (
            SELECT 1 FROM document_acl a
            WHERE a.document_id = d.id
              AND a.principal = ANY (CAST(:principals AS text[]))
        )
    )
"""


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    document_id: str
    title: str
    chunk_index: int
    page: int | None
    content: str
    vector_distance: float | None = None
    keyword_rank: float | None = None
    score: float = 0.0
