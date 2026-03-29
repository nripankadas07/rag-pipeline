"""Vector store abstraction with pluggable backends.

Provides a unified interface for storing and querying embeddings, with
an in-memory backend (always available) and optional FAISS backend
for production workloads.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SearchResult:
    """A single search result from a vector store query."""

    text: str
    score: float
    index: int
    metadata: dict[str, object]


class BaseVectorStore(ABC):
    """Abstract vector store interface."""

    @abstractmethod
    def add(self, texts: list[str], embeddings: np.ndarray, metadata: list[dict] | None = None) -> None:
        """Add texts and their embeddings to the store."""
        ...

    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[SearchResult]:
        """Find the k most similar items to the query embedding."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    def save(self, path: str | Path) -> None:
        """Persist the store to disk."""
        raise NotImplementedError(f"{type(self).__name__} does not support persistence")

    @classmethod
    def load(cls, path: str | Path) -> "BaseVectorStore":
        """Load a store from disk."""
        raise NotImplementedError(f"{cls.__name__} does not support persistence")


class InMemoryVectorStore(BaseVectorStore):
    """Simple numpy-based vector store.

    Good for prototyping and small datasets (< 100k vectors).
    Uses brute-force cosine similarity — no index structure.
    """

    def __init__(self) -> None:
        self._texts: list[str] = []
        self._embeddings: np.ndarray | None = None
        self._metadata: list[dict] = []

    def add(self, texts: list[str], embeddings: np.ndarray, metadata: list[dict] | None = None) -> None:
        if len(texts) != embeddings.shape[0]:
            raise ValueError(f"texts ({len(texts)}) and embeddings ({embeddings.shape[0]}) must have same length")

        self._texts.extend(texts)
        if metadata:
            self._metadata.extend(metadata)
        else:
            self._metadata.extend([{}] * len(texts))

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = embeddings / norms

        if self._embeddings is None:
            self._embeddings = normalized
        else:
            self._embeddings = np.vstack([self._embeddings, normalized])

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[SearchResult]:
        if self._embeddings is None or len(self._texts) == 0:
            return []

        # Normalize query
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []
        query_normalized = query_embedding / query_norm

        # Cosine similarity via dot product (both vectors are normalized)
        similarities = self._embeddings @ query_normalized

        # Get top-k indices
        k = min(k, len(self._texts))
        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            results.append(SearchResult(
                text=self._texts[idx],
                score=float(similarities[idx]),
                index=int(idx),
                metadata=self._metadata[idx],
            ))
        return results

    def __len__(self) -> int:
        return len(self._texts)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "embeddings.npy", self._embeddings)
        with open(path / "data.json", "w") as f:
            json.dump({"texts": self._texts, "metadata": self._metadata}, f)

    @classmethod
    def load(cls, path: str | Path) -> "InMemoryVectorStore":
        path = Path(path)
        store = cls()
        store._embeddings = np.load(path / "embeddings.npy")
        with open(path / "data.json") as f:
            data = json.load(f)
        store._texts = data["texts"]
        store._metadata = data["metadata"]
        return store


class FAISSVectorStore(BaseVectorStore):
    """FAISS-backed vector store for production workloads.

    Supports large-scale similarity search with various index types.
    Requires: pip install faiss-cpu
    """

    def __init__(self, dimension: int | None = None) -> None:
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for FAISSVectorStore. "
                "Install with: pip install faiss-cpu"
            )
        self._faiss = faiss
        self._dimension = dimension
        self._index = None
        self._texts: list[str] = []
        self._metadata: list[dict] = []

    def _ensure_index(self, dimension: int) -> None:
        if self._index is None:
            self._dimension = dimension
            self._index = self._faiss.IndexFlatIP(dimension)  # Inner product (cosine after normalization)

    def add(self, texts: list[str], embeddings: np.ndarray, metadata: list[dict] | None = None) -> None:
        if len(texts) != embeddings.shape[0]:
            raise ValueError("texts and embeddings must have same length")

        self._ensure_index(embeddings.shape[1])

        # Normalize for cosine similarity
        self._faiss.normalize_L2(embeddings.astype(np.float32))

        self._index.add(embeddings.astype(np.float32))
        self._texts.extend(texts)
        if metadata:
            self._metadata.extend(metadata)
        else:
            self._metadata.extend([{}] * len(texts))

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[SearchResult]:
        if self._index is None or self._index.ntotal == 0:
            return []

        query = query_embedding.astype(np.float32).reshape(1, -1)
        self._faiss.normalize_L2(query)

        k = min(k, self._index.ntotal)
        scores, indices = self._index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append(SearchResult(
                    text=self._texts[idx],
                    score=float(score),
                    index=int(idx),
                    metadata=self._metadata[idx],
                ))
        return results

    def __len__(self) -> int:
        return self._index.ntotal if self._index else 0


# Registry
STORES: dict[str, type[BaseVectorStore]] = {
    "memory": InMemoryVectorStore,
    "faiss": FAISSVectorStore,
}


def get_vectorstore(name: str, **kwargs: object) -> BaseVectorStore:
    """Factory function to create a vector store by name."""
    if name not in STORES:
        raise ValueError(f"Unknown store: {name}. Available: {list(STORES.keys())}")
    return STORES[name](**kwargs)
