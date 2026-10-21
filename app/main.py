"""FastAPI application factory. Run with: uvicorn app.main:create_app --factory"""

import asyncio
import re
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from prometheus_client import make_asgi_app

from app.api.routes import chat, documents, health
from app.auth.entra import GraphGroupResolver, TokenValidator, create_group_resolver, create_token_validator
from app.config import Settings, get_settings
from app.database.session import Database
from app.generation.llm_client import create_llm_client
from app.generation.rag_service import RagService
from app.ingestion.embed import Embedder, SentenceTransformerEmbedder
from app.ingestion.pipeline import IngestionPipeline
from app.monitoring import metrics
from app.monitoring.logging import configure_logging, request_id_var
from app.retrieval.hybrid_search import HybridRetriever
from app.retrieval.reranker import create_reranker

_REQUEST_ID = re.compile(r"^[A-Za-z0-9-]{1,64}$")


def create_app(
    settings: Settings | None = None,
    *,
    token_validator: TokenValidator | None = None,
    group_resolver: GraphGroupResolver | None = None,
    database: Database | None = None,
    embedder: Embedder | None = None,
    rag_service: RagService | None = None,
    ingestion_pipeline: IngestionPipeline | None = None,
) -> FastAPI:
    """Build the app. Keyword arguments let tests replace any component with a fake."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    database = database or Database(settings)
    embedder = embedder or SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_dimension)
    if token_validator is None:
        token_validator = create_token_validator(settings)
    if group_resolver is None:
        group_resolver = create_group_resolver(settings)
    if rag_service is None:
        retriever = HybridRetriever(
            database,
            embedder,
            create_reranker(settings),
            top_k=settings.retrieval_top_k,
            candidates=settings.retrieval_candidates,
            rrf_k=settings.rrf_k,
            max_vector_distance=settings.max_vector_distance,
        )
        rag_service = RagService(
            retriever,
            create_llm_client(settings),
            max_context_chars=settings.max_context_chars,
            require_citations=settings.require_citations,
        )
    if ingestion_pipeline is None:
        ingestion_pipeline = IngestionPipeline(
            database,
            embedder,
            chunk_size=settings.chunk_size_words,
            chunk_overlap=settings.chunk_overlap_words,
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.warm_up_models:
            # Load the embedding model before taking traffic, not on the first question.
            await asyncio.to_thread(embedder.warm_up)
        yield
        await database.dispose()

    app = FastAPI(title="Enterprise RAG — Secure Knowledge Assistant", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.token_validator = token_validator
    app.state.group_resolver = group_resolver
    app.state.database = database
    app.state.rag_service = rag_service
    app.state.ingestion_pipeline = ingestion_pipeline

    @app.middleware("http")
    async def observe(request: Request, call_next):
        incoming = request.headers.get("x-request-id", "")
        request_id = incoming if _REQUEST_ID.match(incoming) else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["x-request-id"] = request_id
            return response
        finally:
            route = request.scope.get("route")
            path = getattr(route, "path", "unmatched")
            metrics.HTTP_REQUESTS.labels(request.method, path, str(status)).inc()
            metrics.HTTP_LATENCY.labels(request.method, path).observe(time.perf_counter() - started)
            request_id_var.reset(token)

    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(documents.router)
    # Unauthenticated by design (scraped in-cluster); keep it off the public ingress.
    app.mount("/metrics", make_asgi_app())
    return app
