from __future__ import annotations

import math

import pytest

from hierarchical_rag.hotpotqa import HotpotExample
from hierarchical_rag.retrieval import (
    BM25Index,
    Document,
    evaluate_bm25_candidate_reranking,
    recall_at_k,
)


def _documents() -> tuple[Document, ...]:
    return (
        Document("france", "France", "France is a country in Europe."),
        Document("paris", "Paris", "Paris is the capital city of France."),
        Document("berlin", "Berlin", "Berlin is the capital city of Germany."),
    )


def test_bm25_ranks_lexically_relevant_document_first():
    index = BM25Index(_documents())

    ranking = index.search("capital of France Paris", top_k=3)

    assert ranking[0].document.identifier == "paris"
    assert ranking[0].score > ranking[1].score


def test_bm25_ties_are_broken_by_document_identifier():
    documents = (
        Document("b", "B", "same tokens"),
        Document("a", "A", "same tokens"),
    )

    ranking = BM25Index(documents).search("absent", top_k=2)

    assert [item.document.identifier for item in ranking] == ["a", "b"]


def test_recall_at_k_uses_unique_relevant_documents():
    ranking = BM25Index(_documents()).search("capital France Paris", top_k=3)

    assert math.isclose(recall_at_k(ranking, {"france", "paris"}, 1), 0.5)
    assert recall_at_k(ranking, {"france", "paris"}, 2) == 1.0


def test_recall_requires_relevant_documents():
    ranking = BM25Index(_documents()).search("Paris", top_k=1)

    with pytest.raises(ValueError, match="relevant"):
        recall_at_k(ranking, set(), 1)


def test_bm25_candidate_reranking_reports_paragraph_and_fact_recall(
    hotpot_records,
):
    examples = tuple(
        HotpotExample.from_mapping(record) for record in hotpot_records[:1]
    )

    metrics, rows = evaluate_bm25_candidate_reranking(examples, top_ks=[1, 2])

    assert metrics["candidate_scope"] == "benchmark_provided_distractor_context"
    assert metrics["paragraph_recall_at_2"] == 0.5
    assert metrics["supporting_fact_recall_at_2"] == 0.5
    assert rows[0]["retrieved"][0]["document_id"] == "Paris"
