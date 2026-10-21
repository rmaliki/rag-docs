"""Embedding models. Vectors are L2-normalised so cosine distance is the natural metric."""

import threading
from collections.abc import Sequence
from typing import Protocol


class Embedder(Protocol):
    dimension: int

    def warm_up(self) -> None: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, dimension: int, batch_size: int = 32):
        self.model_name = model_name
        self.dimension = dimension
        self._batch_size = batch_size
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                # Imported lazily: torch is heavy and not needed by most unit tests.
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(self.model_name)
                actual = model.get_sentence_embedding_dimension()
                if actual != self.dimension:
                    raise ValueError(
                        f"{self.model_name} produces {actual}-dimensional vectors, "
                        f"but EMBEDDING_DIMENSION is {self.dimension}"
                    )
                self._model = model
            return self._model

    def warm_up(self) -> None:
        self._load()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._load().encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
