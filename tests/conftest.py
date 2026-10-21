import hashlib
import math
import re

import pytest

from app.config import Settings
from app.retrieval.common import RetrievedChunk

DEV_SECRET = "test-only-secret-" + "x" * 32


class HashingEmbedder:
    """Deterministic bag-of-words embedder: no model download, same 384 dimensions as MiniLM."""

    dimension = 384

    def warm_up(self) -> None:
        pass

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        vector = [0.0] * self.dimension
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            index = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=4).digest(), "big") % self.dimension
            vector[index] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


def make_chunk(chunk_id: int, document_id: str = "doc-1", content: str = "text", **fields) -> RetrievedChunk:
    fields.setdefault("title", "Document")
    fields.setdefault("chunk_index", 0)
    fields.setdefault("page", None)
    return RetrievedChunk(chunk_id=chunk_id, document_id=document_id, content=content, **fields)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        auth_mode="dev",
        dev_jwt_secret=DEV_SECRET,
        warm_up_models=False,
    )
