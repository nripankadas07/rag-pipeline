"""Core RAG pipeline orchestrator.

Wires together chunking, embedding, storage, and retrieval into
a single Pipeline object that can ingest documents and answer queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .chunkers import BaseChunker, Chunk, get_chunker
from .embeddings import BaseEmbedder, get_embedder
from .vectorstore import BaseVectorStore, SearchResult, get_vectorstore


@dataclass
class RetrievalResult:
    """A query result with retrieved context."""

    query: str
    results: list[SearchResult]
    context: str

    @property
    def top_score(self) -> float:
        return self.results[0].score if self.results else 0.0


class Pipeline:
    """End-to-end RAG pipeline.

    Usage:
        pipe = Pipeline()
        pipe.ingest("Your long document text here...")
        result = pipe.query("What is the main topic?", k=3)
        print(result.context)
    """

    def __init__(
        self,
        chunker: BaseChunker | None = None,
        embedder: BaseEmbedder | None = None,
        store: BaseVectorStore | None = None,
        chunker_name: str = "recursive",
        embedder_name: str = "hash",
        store_name: str = "memory",
        **kwargs: object,
    ):
        self.chunker = chunker or get_chunker(chunker_name)
        self.embedder = embedder or get_embedder(embedder_name)
        self.store = store or get_vectorstore(store_name)
        self._chunks: list[Chunk] = []

    def ingest(self, text: str, source: str = "unknown") -> int:
        """Chunk and embed a document, adding it to the vector store.

        Returns the number of chunks created.
        """
        chunks = self.chunker.chunk(text)
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed(texts)

        metadata = [
            {**c.metadata, "source": source, "chunk_index": c.index}
            for c in chunks
        ]

        self.store.add(texts, embeddings, metadata)
        self._chunks.extend(chunks)
        return len(chunks)

    def ingest_many(self, documents: list[tuple[str, str]]) -> int:
        """Ingest multiple (text, source) pairs. Returns total chunks."""
        total = 0
        for text, source in documents:
            total += self.ingest(text, source)
        return total

    def query(self, question: str, k: int = 5) -> RetrievalResult:
        """Retrieve relevant context for a question."""
        query_embedding = self.embedder.embed_query(question)
        results = self.store.search(query_embedding, k=k)

        # Build context from retrieved chunks
        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(f"[{i}] (score: {r.score:.3f}) {r.text}")
        context = "

".join(context_parts)

        return RetrievalResult(
            query=question,
            results=results,
            context=context,
        )

    @property
    def num_chunks(self) -> int:
        return len(self.store)

    def __repr__(self) -> str:
        return (
            f"Pipeline(chunker={type(self.chunker).__name__}, "
            f"embedder={type(self.embedder).__name__}, "
            f"store={type(self.store).__name__}, "
            f"chunks={self.num_chunks})"
        )
