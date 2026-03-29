"""Retrieval evaluation metrics for RAG pipelines.

Provides standard IR metrics to measure how well the retrieval
component surfaces relevant context.
"""

from __future__ import annotations

from dataclasses import dataclass

from .vectorstore import SearchResult


@dataclass
class EvalResult:
    """Evaluation metrics for a single query."""

    query: str
    precision_at_k: float
    recall: float
    mrr: float  # Mean Reciprocal Rank
    hit: bool  # Was any relevant doc in top-k?
    relevant_found: int
    relevant_total: int


def precision_at_k(results: list[SearchResult], relevant_texts: set[str], k: int) -> float:
    """Fraction of top-k results that are relevant."""
    if k == 0:
        return 0.0
    top_k = results[:k]
    hits = sum(1 for r in top_k if r.text in relevant_texts)
    return hits / k


def recall(results: list[SearchResult], relevant_texts: set[str], k: int) -> float:
    """Fraction of relevant documents found in top-k."""
    if not relevant_texts:
        return 0.0
    top_k = results[:k]
    hits = sum(1 for r in top_k if r.text in relevant_texts)
    return hits / len(relevant_texts)


def mean_reciprocal_rank(results: list[SearchResult], relevant_texts: set[str]) -> float:
    """Reciprocal of the rank of the first relevant result."""
    for i, r in enumerate(results, 1):
        if r.text in relevant_texts:
            return 1.0 / i
    return 0.0


def evaluate_query(
    results: list[SearchResult],
    relevant_texts: set[str],
    query: str = "",
    k: int = 5,
) -> EvalResult:
    """Compute all retrieval metrics for a single query."""
    p_at_k = precision_at_k(results, relevant_texts, k)
    rec = recall(results, relevant_texts, k)
    mrr = mean_reciprocal_rank(results, relevant_texts)
    found = sum(1 for r in results[:k] if r.text in relevant_texts)

    return EvalResult(
        query=query,
        precision_at_k=p_at_k,
        recall=rec,
        mrr=mrr,
        hit=found > 0,
        relevant_found=found,
        relevant_total=len(relevant_texts),
    )


def evaluate_batch(
    queries: list[tuple[str, list[SearchResult], set[str]]],
    k: int = 5,
) -> dict[str, float]:
    """Compute aggregate metrics across multiple queries.

    Args:
        queries: List of (query_text, search_results, relevant_texts) tuples.
        k: Number of top results to consider.

    Returns:
        Dict with mean_precision, mean_recall, mean_mrr, hit_rate.
    """
    if not queries:
        return {"mean_precision": 0, "mean_recall": 0, "mean_mrr": 0, "hit_rate": 0}

    evals = [
        evaluate_query(results, relevant, query, k)
        for query, results, relevant in queries
    ]

    n = len(evals)
    return {
        "mean_precision": sum(e.precision_at_k for e in evals) / n,
        "mean_recall": sum(e.recall for e in evals) / n,
        "mean_mrr": sum(e.mrr for e in evals) / n,
        "hit_rate": sum(1 for e in evals if e.hit) / n,
    }
