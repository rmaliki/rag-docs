"""Optional second-stage reranking with a cross-encoder (reads question and chunk together)."""

import threading
from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

from app.config import Settings
from app.retrieval.common import RetrievedChunk


class Reranker(Protocol):
    def rerank(self, question: str, chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]: ...


class NoopReranker:
    def rerank(self, question: str, chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        return list(chunks)


class CrossEncoderReranker:
    """More accurate than bi-encoder similarity, but costs one model pass per candidate."""

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self._model_name)
            return self._model

    def rerank(self, question: str, chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []
        scores = self._load().predict([(question, chunk.content) for chunk in chunks])
        rescored = [replace(chunk, score=float(score)) for chunk, score in zip(chunks, scores, strict=True)]
        return sorted(rescored, key=lambda chunk: chunk.score, reverse=True)


def create_reranker(settings: Settings) -> Reranker:
    return CrossEncoderReranker(settings.reranker_model) if settings.reranker_enabled else NoopReranker()
