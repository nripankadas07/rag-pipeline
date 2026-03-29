"""Tests for chunking strategies."""

import pytest

from rag_pipeline.chunkers import (
    Chunk,
    FixedSizeChunker,
    SentenceChunker,
    RecursiveChunker,
    get_chunker,
)


class TestChunk:
    def test_token_estimate(self):
        c = Chunk(text="hello world foo bar", index=0)
        assert c.token_estimate > 0

    def test_len(self):
        c = Chunk(text="hello", index=0)
        assert len(c) == 5


class TestFixedSizeChunker:
    def test_basic_split(self):
        text = "a" * 1000
        chunker = FixedSizeChunker(chunk_size=200, overlap=0)
        chunks = chunker.chunk(text)
        assert len(chunks) == 5

    def test_overlap(self):
        text = "a" * 500
        chunker = FixedSizeChunker(chunk_size=200, overlap=50)
        chunks = chunker.chunk(text)
        assert len(chunks) > 2

    def test_invalid_overlap(self):
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=100, overlap=100)

    def test_empty_text(self):
        chunker = FixedSizeChunker()
        assert chunker.chunk("") == []

    def test_metadata(self):
        chunker = FixedSizeChunker(chunk_size=100, overlap=0)
        chunks = chunker.chunk("hello world " * 20)
        assert chunks[0].metadata["strategy"] == "fixed_size"


class TestSentenceChunker:
    def test_sentence_boundaries(self):
        text = "First sentence. Second sentence. Third sentence."
        chunker = SentenceChunker(max_chunk_size=1000)
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1

    def test_respects_max_size(self):
        text = "Short. " * 100
        chunker = SentenceChunker(max_chunk_size=50)
        chunks = chunker.chunk(text)
        for c in chunks:
            # Allow some overflow for single long sentences
            assert len(c) < 200

    def test_empty_text(self):
        chunker = SentenceChunker()
        assert chunker.chunk("") == []


class TestRecursiveChunker:
    def test_paragraph_splitting(self):
        text = "Para one content.\n\nPara two content.\n\nPara three."
        chunker = RecursiveChunker(chunk_size=100)
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1

    def test_respects_chunk_size(self):
        text = "word " * 500
        chunker = RecursiveChunker(chunk_size=100, overlap=0)
        chunks = chunker.chunk(text)
        # Most chunks should be near the target size
        for c in chunks[:-1]:  # last chunk may be smaller
            assert len(c) <= 200  # Some tolerance

    def test_empty_text(self):
        chunker = RecursiveChunker()
        assert chunker.chunk("") == []


class TestGetChunker:
    def test_valid_names(self):
        for name in ["fixed", "sentence", "recursive"]:
            chunker = get_chunker(name)
            assert chunker is not None

    def test_invalid_name(self):
        with pytest.raises(ValueError):
            get_chunker("nonexistent")
