# rag-pipeline

A production-ready Retrieval-Augmented Generation pipeline with pluggable chunking strategies, vector store backends, and built-in retrieval evaluation metrics.

## Why This Exists

Most RAG tutorials glue together API calls and call it done. Real production RAG needs: configurable chunking that respects document structure, a vector store abstraction so you can swap backends without rewriting code, and evaluation metrics to know if retrieval actually works. This library provides all three.

## Installation

```bash
pip install -e .
```

With FAISS support:

```bash
pip install -e ".[faiss]"
```

## Quick Start

```python
from rag_pipeline.pipeline import Pipeline

# Create a pipeline with sensible defaults
pipe = Pipeline()

# Ingest a document
pipe.ingest("""
    Retrieval-augmented generation combines search with language models.
    By grounding responses in retrieved documents, RAG reduces hallucination
    and provides traceable, up-to-date answers.
""", source="rag_intro.txt")

# Query it
result = pipe.query("How does RAG reduce hallucination?", k=3)
print(result.context)
```

## Chunking Strategies

Three built-in chunkers, each suited to different document types:

```python
from rag_pipeline.chunkers import get_chunker

# Fixed-size windows — predictable, fast
chunker = get_chunker("fixed", chunk_size=512, overlap=64)

# Sentence-aware — preserves semantic boundaries
chunker = get_chunker("sentence", max_chunk_size=512)

# Recursive — tries paragraph → sentence → word boundaries
chunker = get_chunker("recursive", chunk_size=512, overlap=64)
```

## Vector Stores

Pluggable backends behind a common interface:

```python
from rag_pipeline.vectorstore import get_vectorstore

# In-memory (numpy) — great for prototyping
store = get_vectorstore("memory")

# FAISS — production-scale similarity search
store = get_vectorstore("faiss")
```

## Retrieval Evaluation

Built-in IR metrics to measure retrieval quality:

```python
from rag_pipeline.evaluation import evaluate_query, evaluate_batch

# Single query evaluation
eval_result = evaluate_query(
    results=search_results,
    relevant_texts={"doc1 text", "doc2 text"},
    query="test query",
    k=5,
)
print(f"P@5: {eval_result.precision_at_k:.2f}")
print(f"MRR: {eval_result.mrr:.2f}")

# Batch evaluation across many queries
metrics = evaluate_batch(queries_with_relevance, k=5)
print(f"Hit rate: {metrics['hit_rate']:.1%}")
```

## CLI

```bash
# Chunk a document
ragpipe chunk document.txt --chunker recursive --chunk-size 512

# Ingest and search
ragpipe search document.txt "What is the main argument?"
```

## Project Structure

```
rag-pipeline/
├── rag_pipeline/
│   ├── __init__.py
│   ├── chunkers.py         # Chunking strategies
│   ├── embeddings.py        # Embedding providers
│   ├── vectorstore.py       # Vector store backends
│   ├── pipeline.py          # Pipeline orchestrator
│   ├── evaluation.py        # Retrieval metrics
│   └── cli.py               # CLI interface
├── tests/
│   ├── test_chunkers.py
│   ├── test_vectorstore.py
│   ├── test_pipeline.py
│   └── test_evaluation.py
├── pyproject.toml
├── README.md
└── LICENSE
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
