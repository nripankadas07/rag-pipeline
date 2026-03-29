"""Text chunking strategies for RAG pipelines.

Each chunker implements a consistent interface: split a document into
overlapping or non-overlapping chunks that can be embedded and retrieved.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A single chunk of text with metadata."""

    text: str
    index: int
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        """Rough token count (words ÷ 0.75)."""
        return max(1, int(len(self.text.split()) / 0.75))

    def __len__(self) -> int:
        return len(self.text)


class BaseChunker(ABC):
    """Abstract base class for text chunkers."""

    @abstractmethod
    def chunk(self, text: str, **kwargs: object) -> list[Chunk]:
        """Split text into chunks."""
        ...

    def _make_chunk(self, text: str, index: int, **meta: object) -> Chunk:
        return Chunk(text=text.strip(), index=index, metadata=dict(meta))


class FixedSizeChunker(BaseChunker):
    """Split text into fixed-size character windows with overlap.

    Good baseline strategy — predictable chunk sizes, simple to reason about.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, **kwargs: object) -> list[Chunk]:
        chunks: list[Chunk] = []
        start = 0
        idx = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append(self._make_chunk(
                    chunk_text, idx, start=start, end=min(end, len(text)),
                    strategy="fixed_size",
                ))
                idx += 1
            start += self.chunk_size - self.overlap
        return chunks


class SentenceChunker(BaseChunker):
    """Group sentences into chunks up to a target size.

    Preserves sentence boundaries — better semantic coherence than
    fixed-size splitting for most natural-language documents.
    """

    SENTENCE_PATTERN = re.compile(r"(?<=[.!?])s+")

    def __init__(self, max_chunk_size: int = 512, overlap_sentences: int = 1):
        self.max_chunk_size = max_chunk_size
        self.overlap_sentences = overlap_sentences

    def chunk(self, text: str, **kwargs: object) -> list[Chunk]:
        sentences = self.SENTENCE_PATTERN.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: list[Chunk] = []
        idx = 0
        i = 0

        while i < len(sentences):
            current: list[str] = []
            current_len = 0

            while i < len(sentences):
                sent_len = len(sentences[i])
                if current and current_len + sent_len + 1 > self.max_chunk_size:
                    break
                current.append(sentences[i])
                current_len += sent_len + 1
                i += 1

            chunk_text = " ".join(current)
            if chunk_text:
                chunks.append(self._make_chunk(
                    chunk_text, idx,
                    sentence_count=len(current),
                    strategy="sentence",
                ))
                idx += 1

            # Overlap: step back by overlap_sentences
            i -= self.overlap_sentences

        return chunks


class RecursiveChunker(BaseChunker):
    """Recursively split on separators, falling through to smaller ones.

    Mirrors LangChain's RecursiveCharacterTextSplitter logic but with
    a cleaner implementation. Tries paragraph → sentence → word boundaries.
    """

    DEFAULT_SEPARATORS = ["

", "
", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def chunk(self, text: str, **kwargs: object) -> list[Chunk]:
        pieces = self._split_recursive(text, self.separators)
        # Merge small pieces back into chunks with overlap
        chunks: list[Chunk] = []
        idx = 0
        current = ""

        for piece in pieces:
            if len(current) + len(piece) + 1 <= self.chunk_size:
                current = f"{current} {piece}".strip() if current else piece
            else:
                if current:
                    chunks.append(self._make_chunk(current, idx, strategy="recursive"))
                    idx += 1
                    # Keep overlap from end of current
                    if self.overlap > 0:
                        current = current[-self.overlap:] + " " + piece
                    else:
                        current = piece
                else:
                    # Single piece exceeds chunk_size — include anyway
                    chunks.append(self._make_chunk(piece, idx, strategy="recursive"))
                    idx += 1
                    current = ""

        if current.strip():
            chunks.append(self._make_chunk(current, idx, strategy="recursive"))

        return chunks

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return [text] if text else []

        sep = separators[0]
        remaining = separators[1:]

        if sep == "":
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        parts = text.split(sep)
        result: list[str] = []

        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) <= self.chunk_size:
                result.append(part)
            else:
                result.extend(self._split_recursive(part, remaining))

        return result


# Registry for easy access
CHUNKERS: dict[str, type[BaseChunker]] = {
    "fixed": FixedSizeChunker,
    "sentence": SentenceChunker,
    "recursive": RecursiveChunker,
}


def get_chunker(name: str, **kwargs: object) -> BaseChunker:
    """Factory function to create a chunker by name."""
    if name not in CHUNKERS:
        raise ValueError(f"Unknown chunker: {name}. Available: {list(CHUNKERS.keys())}")
    return CHUNKERS[name](**kwargs)
