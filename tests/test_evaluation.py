"""Tests for retrieval evaluation metrics."""

import pytest

from rag_pipeline.evaluation import (
    precision_at_k,
    recall,
    mean_reciprocal_rank,
    evaluate_query,
    evaluate_batch,
)
from rag_pipeline.vectorstore import SearchResult


def _make_result(text: str, score: float = 0.5) -> SearchResult:
    return SearchResult(text=text, score=score, index=0, metadata={})


class TestPrecisionAtK:
    def test_all_relevant(self):
        results = [_make_result("a"), _make_result("b")]
        relevant = {"a", "b"}
        assert precision_at_k(results, relevant, k=2) == 1.0

    def test_none_relevant(self):
        results = [_make_result("a"), _make_result("b")]
        relevant = {"c", "d"}
        assert precision_at_k(results, relevant, k=2) == 0.0

    def test_half_relevant(self):
        results = [_make_result("a"), _make_result("b")]
        relevant = {"a"}
        assert precision_at_k(results, relevant, k=2) == 0.5

    def test_k_zero(self):
        assert precision_at_k([], set(), k=0) == 0.0


class TestRecall:
    def test_all_found(self):
        results = [_make_result("a"), _make_result("b")]
        relevant = {"a", "b"}
        assert recall(results, relevant, k=2) == 1.0

    def test_partial(self):
        results = [_make_result("a"), _make_result("c")]
        relevant = {"a", "b"}
        assert recall(results, relevant, k=2) == 0.5

    def test_empty_relevant(self):
        results = [_make_result("a")]
        assert recall(results, set(), k=1) == 0.0


class TestMRR:
    def test_first_is_relevant(self):
        results = [_make_result("a"), _make_result("b")]
        assert mean_reciprocal_rank(results, {"a"}) == 1.0

    def test_second_is_relevant(self):
        results = [_make_result("a"), _make_result("b")]
        assert mean_reciprocal_rank(results, {"b"}) == 0.5

    def test_none_relevant(self):
        results = [_make_result("a")]
        assert mean_reciprocal_rank(results, {"z"}) == 0.0


class TestEvaluateQuery:
    def test_full_eval(self):
        results = [_make_result("a"), _make_result("b"), _make_result("c")]
        relevant = {"a", "c"}
        ev = evaluate_query(results, relevant, query="test", k=3)
        assert ev.hit is True
        assert ev.relevant_found == 2
        assert ev.precision_at_k == pytest.approx(2 / 3)
        assert ev.recall == pytest.approx(1.0)


class TestEvaluateBatch:
    def test_aggregate(self):
        r1 = [_make_result("a"), _make_result("b")]
        r2 = [_make_result("c"), _make_result("d")]
        queries = [
            ("q1", r1, {"a"}),
            ("q2", r2, {"c"}),
        ]
        metrics = evaluate_batch(queries, k=2)
        assert metrics["hit_rate"] == 1.0
        assert metrics["mean_precision"] == 0.5
        assert metrics["mean_mrr"] == 1.0

    def test_empty(self):
        metrics = evaluate_batch([], k=5)
        assert metrics["hit_rate"] == 0
