from __future__ import annotations

from hierarchical_rag.audit_e1 import summarize_e1_records
from hierarchical_rag.hotpotqa import HotpotExample, SupportingFact


def _example(
    identifier: str, question_type: str, level: str, titles: tuple[str, str]
) -> HotpotExample:
    return HotpotExample(
        identifier=identifier,
        question=f"Question {identifier}",
        answer=None,
        question_type=question_type,
        level=level,
        supporting_facts=(
            SupportingFact(titles[0], 0),
            SupportingFact(titles[1], 0),
        ),
        context=(),
    )


def _retrieval(
    example: HotpotExample, retrieved: tuple[str, ...], metrics: dict[str, float]
) -> dict:
    return {
        "example_id": example.identifier,
        "query": example.question,
        "gold_titles": [fact.title for fact in example.supporting_facts],
        "latency_seconds": 1.0,
        "retrieved": [
            {
                "rank": rank,
                "document_id": title,
                "title": title,
                "score": 3.0 - rank,
            }
            for rank, title in enumerate(retrieved, start=1)
        ],
        "metrics": metrics,
    }


def _prediction(example: HotpotExample) -> dict:
    return {
        "example_id": example.identifier,
        "status": "not_applicable",
        "reason": "E1 evaluates retrieval without a reader.",
    }


def test_e1_summary_recomputes_metrics_and_error_groups():
    complete = _example("a", "comparison", "easy", ("A", "B"))
    missing = _example("b", "bridge", "hard", ("C", "D"))
    complete_metrics = {
        "paragraph_recall_at_10": 1.0,
        "supporting_fact_recall_at_10": 1.0,
        "all_supporting_paragraphs_at_10": 1.0,
    }
    missing_metrics = {
        key: 0.0 for key in complete_metrics
    }
    complete_ranking = ("A", "B", *(f"X{index}" for index in range(8)))
    missing_ranking = tuple(f"Y{index}" for index in range(10))
    metrics, analysis, latencies = summarize_e1_records(
        examples=(complete, missing),
        retrieval=(
            _retrieval(complete, complete_ranking, complete_metrics),
            _retrieval(missing, missing_ranking, missing_metrics),
        ),
        predictions=(_prediction(complete), _prediction(missing)),
        cutoffs=(10,),
    )

    assert metrics["paragraph_recall_at_10"] == 0.5
    assert metrics["all_supporting_paragraphs_at_10"] == 0.5
    assert analysis["paragraph_outcomes_at_10"]["complete"]["count"] == 1
    assert analysis["paragraph_outcomes_at_10"]["none"]["count"] == 1
    assert analysis["gold_paragraph_rank_bins"]["missing"]["count"] == 2
    assert analysis["by_question_type"]["bridge"]["paragraph_recall_at_10"] == 0.0
    assert latencies == [1.0, 1.0]


def test_e1_summary_rejects_tampered_per_example_metric():
    example = _example("a", "bridge", "hard", ("A", "B"))
    metrics = {
        "paragraph_recall_at_10": 0.0,
        "supporting_fact_recall_at_10": 1.0,
        "all_supporting_paragraphs_at_10": 1.0,
    }
    ranking = ("A", "B", *(f"X{index}" for index in range(8)))

    try:
        summarize_e1_records(
            examples=(example,),
            retrieval=(_retrieval(example, ranking, metrics),),
            predictions=(_prediction(example),),
            cutoffs=(10,),
        )
    except ValueError as error:
        assert "per-example metrics differ" in str(error)
    else:
        raise AssertionError("tampered E1 metric was accepted")
