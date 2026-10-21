"""Hybrid retrieval: vector + keyword search fused with Reciprocal Rank Fusion, all ACL-filtered."""

import asyncio
from collections.abc import Sequence
from dataclasses import replace

from app.auth.permissions import Permissions
from app.database.session import Database
from app.ingestion.embed import Embedder
from app.retrieval.common import RetrievedChunk
from app.retrieval.keyword_search import keyword_search
from app.retrieval.reranker import Reranker
from app.retrieval.vector_search import vector_search


def _first_not_none(first: float | None, second: float | None) -> float | None:
    return first if first is not None else second


def reciprocal_rank_fusion(result_lists: Sequence[Sequence[RetrievedChunk]], k: int = 60) -> list[RetrievedChunk]:
    """Combine ranked lists without having to calibrate their incomparable raw scores.

    score(chunk) = sum over lists of 1 / (k + rank). Chunks are de-duplicated by id.
    """
    scores: dict[int, float] = {}
    merged: dict[int, RetrievedChunk] = {}
    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            previous = merged.get(chunk.chunk_id)
            if previous is None:
                merged[chunk.chunk_id] = chunk
            else:
                # The same chunk found by both retrievers: keep both signals.
                merged[chunk.chunk_id] = replace(
                    previous,
                    vector_distance=_first_not_none(previous.vector_distance, chunk.vector_distance),
                    keyword_rank=_first_not_none(previous.keyword_rank, chunk.keyword_rank),
                )
    fused = [replace(chunk, score=scores[chunk_id]) for chunk_id, chunk in merged.items()]
    return sorted(fused, key=lambda chunk: chunk.score, reverse=True)


def fuse_results(
    vector_hits: Sequence[RetrievedChunk],
    keyword_hits: Sequence[RetrievedChunk],
    *,
    max_vector_distance: float,
    rrf_k: int,
) -> list[RetrievedChunk]:
    # Vector search always returns its nearest neighbours, however unrelated. Dropping
    # distant ones lets an off-topic question end in "insufficient information".
    relevant_vector_hits = [
        hit for hit in vector_hits if hit.vector_distance is not None and hit.vector_distance <= max_vector_distance
    ]
    return reciprocal_rank_fusion([relevant_vector_hits, keyword_hits], k=rrf_k)


class HybridRetriever:
    def __init__(
        self,
        database: Database,
        embedder: Embedder,
        reranker: Reranker,
        *,
        top_k: int,
        candidates: int,
        rrf_k: int,
        max_vector_distance: float,
    ):
        self._database = database
        self._embedder = embedder
        self._reranker = reranker
        self._top_k = top_k
        self._candidates = candidates
        self._rrf_k = rrf_k
        self._max_vector_distance = max_vector_distance

    async def retrieve(self, question: str, permissions: Permissions) -> list[RetrievedChunk]:
        query_embedding = await asyncio.to_thread(self._embedder.embed_query, question)
        async with self._database.authorized_session(permissions) as session:
            vector_hits = await vector_search(session, query_embedding, permissions.principals, self._candidates)
            keyword_hits = await keyword_search(session, question, permissions.principals, self._candidates)
        fused = fuse_results(
            vector_hits,
            keyword_hits,
            max_vector_distance=self._max_vector_distance,
            rrf_k=self._rrf_k,
        )
        reranked = await asyncio.to_thread(self._reranker.rerank, question, fused)
        return reranked[: self._top_k]
