"""Application settings, loaded from environment variables (and `.env` in development)."""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"

    # Two database roles: the API reads through `rag_app`, which is subject to
    # row-level security; ingestion and migrations write through `rag_admin`.
    database_url: str = "postgresql+psycopg://rag_app:rag_app@localhost:5432/rag"
    admin_database_url: str = "postgresql+psycopg://rag_admin:rag_admin@localhost:5432/rag"
    db_pool_size: int = 10
    db_max_overflow: int = 5
    # pgvector >= 0.8: keep scanning the HNSW index when ACL filters drop candidates.
    hnsw_iterative_scan: bool = True

    auth_mode: Literal["entra", "dev"] = "entra"
    entra_tenant_id: str = ""
    entra_audience: str = ""
    graph_client_id: str = ""
    graph_client_secret: SecretStr | None = None
    dev_jwt_secret: SecretStr | None = None
    admin_role: str = "Documents.Admin"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    warm_up_models: bool = True
    chunk_size_words: int = 200
    chunk_overlap_words: int = 40
    max_upload_bytes: int = 20 * 1024 * 1024

    retrieval_top_k: int = 5
    retrieval_candidates: int = 20
    rrf_k: int = 60
    max_vector_distance: float = 0.75
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    llm_provider: Literal["azure_openai", "openai_compatible"] = "azure_openai"
    azure_openai_endpoint: str = ""
    azure_openai_api_key: SecretStr | None = None
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_deployment: str = ""
    openai_base_url: str = ""
    openai_api_key: SecretStr | None = None
    openai_model: str = ""
    llm_temperature: float = 0.0
    llm_max_tokens: int = 800
    llm_timeout_seconds: float = 30.0
    max_context_chars: int = 12_000
    require_citations: bool = True

    @model_validator(mode="after")
    def _validate(self) -> "Settings":
        if self.chunk_overlap_words >= self.chunk_size_words:
            raise ValueError("CHUNK_OVERLAP_WORDS must be smaller than CHUNK_SIZE_WORDS")
        if self.auth_mode == "dev":
            if self.environment == "production":
                raise ValueError("AUTH_MODE=dev is not allowed in production")
            secret = self.dev_jwt_secret.get_secret_value() if self.dev_jwt_secret else ""
            if len(secret) < 32:
                raise ValueError("DEV_JWT_SECRET must be at least 32 characters when AUTH_MODE=dev")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
