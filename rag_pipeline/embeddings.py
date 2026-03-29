"""Embedding providers for the RAG pipeline.

Supports a hash-based fallback (zero dependencies) and an interface
for real embedding models.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    """Abstract embedder interface."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts, returning an (N, D) array."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimensionality."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query text, returning a (D,) array."""
        return self.embed([text])[0]


class HashEmbedder(BaseEmbedder):
    """Deterministic hash-based embedder for testing and demos.

    NOT a real embedding model — produces pseudo-embeddings from SHA-512
    hashes. Useful for testing the full pipeline without downloading models.
    """

    def __init__(self, dimension: int = 128):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        embeddings = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            h = hashlib.sha512(text.lower().strip().encode()).digest()
            while len(h) < self._dimension * 4:
                h += hashlib.sha512(h).digest()
            raw = np.frombuffer(h[: self._dimension * 4], dtype=np.uint8).copy()
            vec = (raw.astype(np.float32) / 255.0) - 0.5
            vec = vec[: self._dimension]
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings[i] = vec
        return embeddings


class SentenceTransformerEmbedder(BaseEmbedder):
    """Real embedding model via sentence-transformers.

    Requires: pip install sentence-transformers
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers required. "
                "Install with: pip install sentence-transformers"
            )
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True)


def get_embedder(name: str = "hash", **kwargs: object) -> BaseEmbedder:
    """Factory function to create an embedder."""
    embedders = {
        "hash": HashEmbedder,
        "sentence-transformer": SentenceTransformerEmbedder,
    }
    if name not in embedders:
        raise ValueError(f"Unknown embedder: {name}. Available: {list(embedders.keys())}")
    return embedders[name](**kwargs)
