from __future__ import annotations

import math

from hierarchical_rag.hotpotqa import HotpotExample, SupportingFact
from hierarchical_rag.metrics import (
    aggregate_answer_metrics,
    answer_score,
    evaluate_hotpotqa,
    normalize_answer,
    supporting_fact_score,
)


def test_official_answer_normalization():
    assert normalize_answer("The U.S.-based cat!") == "usbased cat"
    assert normalize_answer("  An   Answer ") == "answer"


def test_answer_em_and_partial_f1():
    exact = answer_score("The Paris", "Paris")
    partial = answer_score("capital city Paris", "Paris")

    assert exact.exact_match == 1.0
    assert exact.f1 == 1.0
    assert partial.exact_match == 0.0
    assert math.isclose(partial.precision, 1 / 3)
    assert math.isclose(partial.recall, 1.0)
    assert math.isclose(partial.f1, 0.5)


def test_yes_no_mismatch_has_zero_overlap_score():
    score = answer_score("no", "yes")

    assert score.exact_match == 0.0
    assert score.f1 == 0.0


def test_supporting_fact_metrics_match_set_semantics():
    gold = [SupportingFact("A", 0), SupportingFact("B", 1)]
    predicted = [SupportingFact("A", 0), SupportingFact("C", 2)]

    score = supporting_fact_score(predicted, gold)

    assert score.exact_match == 0.0
    assert score.precision == 0.5
    assert score.recall == 0.5
    assert score.f1 == 0.5


def test_aggregate_counts_missing_predictions_as_empty(hotpot_records):
    examples = tuple(HotpotExample.from_mapping(record) for record in hotpot_records[:2])

    metrics = aggregate_answer_metrics({"example-a": "Paris"}, examples)

    assert metrics.count == 2
    assert metrics.exact_match == 0.5
    assert metrics.missing_ids == ("example-b",)


def test_full_evaluator_matches_official_joint_formula(hotpot_records):
    examples = tuple(HotpotExample.from_mapping(record) for record in hotpot_records[:2])
    answers = {"example-a": "Paris", "example-b": "no"}
    supporting = {
        "example-a": [["France", 0], ["Paris", 0]],
        "example-b": [["Wrong", 0]],
    }

    metrics = evaluate_hotpotqa(answers, supporting, examples)

    assert metrics.answer_exact_match == 0.5
    assert metrics.supporting_exact_match == 0.5
    assert metrics.joint_exact_match == 0.5
    assert metrics.official_dict()["joint_f1"] == 0.5


def test_full_evaluator_records_missing_fields(hotpot_records):
    examples = tuple(HotpotExample.from_mapping(record) for record in hotpot_records[:1])

    metrics = evaluate_hotpotqa({}, {}, examples)

    assert metrics.missing_answer_ids == ("example-a",)
    assert metrics.missing_supporting_fact_ids == ("example-a",)
    assert metrics.joint_f1 == 0.0
