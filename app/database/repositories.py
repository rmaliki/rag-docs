"""Document persistence. Writes go through the admin pool; reads through an authorized session."""

from collections.abc import Sequence

from sqlalchemy import delete, exists, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database.models import Document, DocumentAcl, DocumentChunk
from app.ingestion.chunk import Chunk
from app.ingestion.metadata import DocumentMetadata


class DocumentRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get(self, document_id: str) -> Document | None:
        result = await self._session.execute(
            select(Document).options(selectinload(Document.acl)).where(Document.id == document_id)
        )
        return result.scalar_one_or_none()

    def update_metadata(self, document: Document, metadata: DocumentMetadata) -> bool:
        """Apply metadata and ACL changes without re-embedding. Returns True if anything changed."""
        changed = False
        for attribute, value in (
            ("title", metadata.title),
            ("department", metadata.department),
            ("classification", metadata.classification),
            ("version", metadata.version),
            ("is_public", metadata.public),
        ):
            if getattr(document, attribute) != value:
                setattr(document, attribute, value)
                changed = True

        # Diff rather than replace the ACL list: deleting and re-inserting the same
        # (document_id, principal) key in one flush would violate the primary key.
        wanted = set(metadata.acl_principals())
        current = {entry.principal: entry for entry in document.acl}
        for principal in current.keys() - wanted:
            document.acl.remove(current[principal])
            changed = True
        for principal in sorted(wanted - current.keys()):
            document.acl.append(DocumentAcl(principal=principal))
            changed = True
        return changed

    async def replace(
        self,
        metadata: DocumentMetadata,
        *,
        content_hash: str,
        source_uri: str,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Replace a document, its ACL and all its chunks atomically (caller owns the transaction)."""
        await self._session.execute(delete(Document).where(Document.id == metadata.document_id))
        self._session.add(
            Document(
                id=metadata.document_id,
                title=metadata.title,
                department=metadata.department,
                classification=metadata.classification,
                version=metadata.version,
                is_public=metadata.public,
                source_uri=source_uri,
                content_hash=content_hash,
                acl=[DocumentAcl(principal=p) for p in metadata.acl_principals()],
                chunks=[
                    DocumentChunk(
                        chunk_index=chunk.index,
                        page=chunk.page,
                        content=chunk.text,
                        embedding=list(embedding),
                    )
                    for chunk, embedding in zip(chunks, embeddings, strict=True)
                ],
            )
        )

    async def list_visible(self, principals: Sequence[str]) -> list[Document]:
        """Documents readable by the principals (row-level security enforces the same rule)."""
        allowed = exists().where(
            DocumentAcl.document_id == Document.id, DocumentAcl.principal.in_(list(principals))
        )
        result = await self._session.execute(
            select(Document)
            .where(or_(Document.is_public, allowed))
            .order_by(Document.department, Document.title)
        )
        return list(result.scalars())
