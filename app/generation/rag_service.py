"""Question answering: authorized retrieval -> grounded generation -> citation validation."""

import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from app.auth.permissions import Permissions
from app.generation.llm_client import LLMClient
from app.generation.prompts import INSUFFICIENT_EVIDENCE_ANSWER, Source, build_messages, select_sources
from app.monitoring import metrics
from app.retrieval.common import RetrievedChunk

logger = logging.getLogger(__name__)

CITATION_PATTERN = re.compile(r"\[(S\d+)\]")


class Retriever(Protocol):
    async def retrieve(self, question: str, permissions: Permissions) -> list[RetrievedChunk]: ...


@dataclass(frozen=True)
class Citation:
    source_id: str
    document_id: str
    title: str
    chunk_index: int
    page: int | None


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    insufficient_evidence: bool = False


def insufficient_evidence() -> RagAnswer:
    return RagAnswer(answer=INSUFFICIENT_EVIDENCE_ANSWER, citations=[], insufficient_evidence=True)


def validate_citations(answer: str, sources: Sequence[Source]) -> tuple[str, list[Citation]]:
    """Keep citations that point at a source we actually sent; strip invented ones."""
    by_id = {source.source_id: source for source in sources}
    cited: dict[str, Citation] = {}
    invalid = 0

    def keep_or_strip(match: re.Match[str]) -> str:
        nonlocal invalid
        source = by_id.get(match.group(1))
        if source is None:
            invalid += 1
            return ""
        chunk = source.chunk
        cited.setdefault(
            source.source_id,
            Citation(source.source_id, chunk.document_id, chunk.title, chunk.chunk_index, chunk.page),
        )
        return match.group(0)

    cleaned = CITATION_PATTERN.sub(keep_or_strip, answer).strip()
    if invalid:
        metrics.INVALID_CITATIONS.inc(invalid)
    return cleaned, list(cited.values())


class RagService:
    def __init__(self, retriever: Retriever, llm: LLMClient, *, max_context_chars: int, require_citations: bool):
        self._retriever = retriever
        self._llm = llm
        self._max_context_chars = max_context_chars
        self._require_citations = require_citations

    async def answer_question(self, question: str, permissions: Permissions) -> RagAnswer:
        started = time.perf_counter()
        try:
            answer = await self._answer(question, permissions)
        except Exception:
            metrics.CHAT_REQUESTS.labels(outcome="error").inc()
            raise
        outcome = "insufficient_evidence" if answer.insufficient_evidence else "answered"
        metrics.CHAT_REQUESTS.labels(outcome=outcome).inc()
        metrics.RESPONSE_LATENCY.observe(time.perf_counter() - started)
        return answer

    async def _answer(self, question: str, permissions: Permissions) -> RagAnswer:
        with metrics.RETRIEVAL_LATENCY.time():
            chunks = await self._retriever.retrieve(question, permissions)
        metrics.RETRIEVED_CHUNKS.observe(len(chunks))
        if not chunks:
            # Nothing authorized and relevant: answer without calling the LLM at all.
            return insufficient_evidence()

        sources = select_sources(chunks, self._max_context_chars)
        result = await self._llm.complete(build_messages(question, sources))
        if result.time_to_first_token is not None:
            metrics.LLM_TIME_TO_FIRST_TOKEN.observe(result.time_to_first_token)
        if result.prompt_tokens:
            metrics.LLM_TOKENS.labels(kind="prompt").inc(result.prompt_tokens)
        if result.completion_tokens:
            metrics.LLM_TOKENS.labels(kind="completion").inc(result.completion_tokens)

        text = result.text.strip()
        if not text or INSUFFICIENT_EVIDENCE_ANSWER.lower().rstrip(".") in text.lower():
            return insufficient_evidence()

        text, citations = validate_citations(text, sources)
        if self._require_citations and not citations:
            # An answer that cites nothing cannot be traced to evidence: treat it as unsupported.
            metrics.UNCITED_ANSWERS.inc()
            logger.warning("model answer had no valid citation; returning insufficient evidence")
            return insufficient_evidence()
        return RagAnswer(answer=text, citations=citations)
