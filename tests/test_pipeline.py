"""Tests for the RAG pipeline orchestrator."""

import pytest

from rag_pipeline.pipeline import Pipeline


SAMPLE_TEXT = """
Artificial intelligence has transformed how we process information.
Machine learning models can now understand natural language with remarkable accuracy.

Deep learning architectures like transformers have revolutionized NLP tasks.
Models such as BERT and GPT have set new benchmarks across many domains.

Retrieval-augmented generation combines the strengths of search and generation.
By grounding responses in retrieved documents, RAG reduces hallucination.
""".strip()


class TestPipeline:
    def test_ingest(self):
        pipe = Pipeline()
        count = pipe.ingest(SAMPLE_TEXT, source="test")
        assert count > 0
        assert pipe.num_chunks > 0

    def test_query(self):
        pipe = Pipeline()
        pipe.ingest(SAMPLE_TEXT, source="test")
        result = pipe.query("What is RAG?", k=3)
        assert len(result.results) > 0
        assert result.context != ""
        assert result.query == "What is RAG?"

    def test_ingest_many(self):
        pipe = Pipeline()
        docs = [
            ("First document about cats.", "doc1"),
            ("Second document about dogs.", "doc2"),
        ]
        total = pipe.ingest_many(docs)
        assert total >= 2

    def test_empty_ingest(self):
        pipe = Pipeline()
        count = pipe.ingest("")
        assert count == 0

    def test_repr(self):
        pipe = Pipeline()
        r = repr(pipe)
        assert "Pipeline" in r
        assert "RecursiveChunker" in r

    def test_top_score(self):
        pipe = Pipeline()
        pipe.ingest(SAMPLE_TEXT, source="test")
        result = pipe.query("transformers")
        assert result.top_score > 0

    def test_custom_components(self):
        from rag_pipeline.chunkers import FixedSizeChunker
        from rag_pipeline.embeddings import HashEmbedder
        from rag_pipeline.vectorstore import InMemoryVectorStore

        pipe = Pipeline(
            chunker=FixedSizeChunker(chunk_size=100, overlap=10),
            embedder=HashEmbedder(dimension=64),
            store=InMemoryVectorStore(),
        )
        pipe.ingest(SAMPLE_TEXT)
        result = pipe.query("machine learning")
        assert len(result.results) > 0
