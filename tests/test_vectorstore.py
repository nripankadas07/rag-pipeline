"""Tests for vector store implementations."""

import numpy as np
import pytest

from rag_pipeline.vectorstore import InMemoryVectorStore, SearchResult, get_vectorstore


class TestInMemoryVectorStore:
    def test_add_and_search(self):
        store = InMemoryVectorStore()
        texts = ["hello world", "foo bar"]
        embeddings = np.random.randn(2, 128).astype(np.float32)
        store.add(texts, embeddings)

        results = store.search(embeddings[0], k=2)
        assert len(results) == 2
        assert results[0].text == "hello world"  # Most similar to itself

    def test_length(self):
        store = InMemoryVectorStore()
        assert len(store) == 0
        store.add(["a"], np.random.randn(1, 64).astype(np.float32))
        assert len(store) == 1

    def test_mismatched_lengths(self):
        store = InMemoryVectorStore()
        with pytest.raises(ValueError):
            store.add(["a", "b"], np.random.randn(3, 64).astype(np.float32))

    def test_empty_search(self):
        store = InMemoryVectorStore()
        results = store.search(np.random.randn(64).astype(np.float32))
        assert results == []

    def test_metadata(self):
        store = InMemoryVectorStore()
        store.add(
            ["test"],
            np.random.randn(1, 64).astype(np.float32),
            metadata=[{"source": "doc1"}],
        )
        results = store.search(np.random.randn(64).astype(np.float32), k=1)
        assert results[0].metadata["source"] == "doc1"

    def test_save_and_load(self, tmp_path):
        store = InMemoryVectorStore()
        texts = ["hello", "world"]
        embeddings = np.random.randn(2, 64).astype(np.float32)
        store.add(texts, embeddings, metadata=[{"id": 1}, {"id": 2}])

        store.save(tmp_path / "test_store")
        loaded = InMemoryVectorStore.load(tmp_path / "test_store")

        assert len(loaded) == 2
        results = loaded.search(embeddings[0], k=1)
        assert results[0].text == "hello"

    def test_cosine_similarity_ordering(self):
        store = InMemoryVectorStore()
        # Create two very different vectors
        v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        store.add(["similar", "different"], np.vstack([v1, v2]))

        # Query with v1 — should rank "similar" first
        results = store.search(v1, k=2)
        assert results[0].text == "similar"


class TestGetVectorStore:
    def test_memory(self):
        store = get_vectorstore("memory")
        assert isinstance(store, InMemoryVectorStore)

    def test_invalid(self):
        with pytest.raises(ValueError):
            get_vectorstore("nonexistent")
