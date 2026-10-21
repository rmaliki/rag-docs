import tempfile
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, ValidationError

from app.api.dependencies import AdminPermissions, CurrentPermissions
from app.database.repositories import DocumentRepository
from app.ingestion.extract import SUPPORTED_EXTENSIONS
from app.ingestion.metadata import DocumentMetadata
from app.ingestion.pipeline import IngestionError

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentOut(BaseModel):
    document_id: str
    title: str
    department: str
    classification: str
    version: int
    updated_at: datetime


class IngestionOut(BaseModel):
    document_id: str
    status: str
    chunk_count: int | None


@router.get("", response_model=list[DocumentOut])
async def list_documents(request: Request, permissions: CurrentPermissions) -> list[DocumentOut]:
    """The documents the caller may read (and therefore may receive answers from)."""
    async with request.app.state.database.authorized_session(permissions) as session:
        documents = await DocumentRepository(session).list_visible(sorted(permissions.principals))
    return [
        DocumentOut(
            document_id=d.id,
            title=d.title,
            department=d.department,
            classification=d.classification,
            version=d.version,
            updated_at=d.updated_at,
        )
        for d in documents
    ]


@router.post("", response_model=IngestionOut, status_code=status.HTTP_201_CREATED)
async def upload_document(
    request: Request,
    _admin: AdminPermissions,
    file: Annotated[UploadFile, File()],
    metadata: Annotated[str, Form(description="DocumentMetadata as JSON")],
) -> IngestionOut:
    try:
        parsed = DocumentMetadata.model_validate_json(metadata)
    except ValidationError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            jsonable_encoder(exc.errors(include_url=False, include_context=False)),
        ) from None

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Supported types: {sorted(SUPPORTED_EXTENSIONS)}")

    max_bytes = request.app.state.settings.max_upload_bytes
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / f"upload{suffix}"
        size = 0
        with target.open("wb") as out:
            while block := await file.read(1024 * 1024):
                size += len(block)
                if size > max_bytes:
                    raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "File too large")
                out.write(block)
        try:
            result = await request.app.state.ingestion_pipeline.ingest_file(
                target, parsed, source_uri=file.filename
            )
        except IngestionError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from None
    return IngestionOut(document_id=result.document_id, status=result.status, chunk_count=result.chunk_count)
