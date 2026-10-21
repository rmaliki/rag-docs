import logging

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from app.api.dependencies import CurrentPermissions
from app.generation.llm_client import LLMError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    # extra="forbid": the body carries the question only. Groups or user ids sent by
    # the client are rejected, never trusted: permissions come from the token.
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)


class CitationOut(BaseModel):
    source_id: str
    document_id: str
    title: str
    chunk_index: int
    page: int | None


class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    insufficient_evidence: bool


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request, permissions: CurrentPermissions) -> ChatResponse:
    try:
        result = await request.app.state.rag_service.answer_question(body.question.strip(), permissions)
    except LLMError:
        logger.exception("LLM call failed")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "The assistant is temporarily unavailable") from None
    return ChatResponse(
        answer=result.answer,
        citations=[CitationOut(**vars(citation)) for citation in result.citations],
        insufficient_evidence=result.insufficient_evidence,
    )
