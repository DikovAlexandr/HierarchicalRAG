"""Official-compatible HotpotQA answer and supporting-fact metrics."""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from hierarchical_rag.hotpotqa import HotpotExample, SupportingFact


@dataclass(frozen=True, slots=True)
class Score:
    exact_match: float
    f1: float
    precision: float
    recall: float


@dataclass(frozen=True, slots=True)
class AggregateAnswerMetrics:
    count: int
    exact_match: float
    f1: float
    precision: float
    recall: float
    missing_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AggregateHotpotMetrics:
    """The twelve metrics emitted by the official HotpotQA evaluator."""

    count: int
    answer_exact_match: float
    answer_f1: float
    answer_precision: float
    answer_recall: float
    supporting_exact_match: float
    supporting_f1: float
    supporting_precision: float
    supporting_recall: float
    joint_exact_match: float
    joint_f1: float
    joint_precision: float
    joint_recall: float
    missing_answer_ids: tuple[str, ...]
    missing_supporting_fact_ids: tuple[str, ...]

    def official_dict(self) -> dict[str, float]:
        """Return official scorer field names for direct comparison."""

        return {
            "em": self.answer_exact_match,
            "f1": self.answer_f1,
            "prec": self.answer_precision,
            "recall": self.answer_recall,
            "sp_em": self.supporting_exact_match,
            "sp_f1": self.supporting_f1,
            "sp_prec": self.supporting_precision,
            "sp_recall": self.supporting_recall,
            "joint_em": self.joint_exact_match,
            "joint_f1": self.joint_f1,
            "joint_prec": self.joint_precision,
            "joint_recall": self.joint_recall,
        }


def normalize_answer(text: str) -> str:
    """Match the official lower/punctuation/article/whitespace normalization."""

    lowered = text.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in set(string.punctuation)
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def answer_score(prediction: str, ground_truth: str) -> Score:
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    exact_match = float(normalized_prediction == normalized_ground_truth)

    special = {"yes", "no", "noanswer"}
    if (
        normalized_prediction in special or normalized_ground_truth in special
    ) and normalized_prediction != normalized_ground_truth:
        return Score(exact_match=exact_match, f1=0.0, precision=0.0, recall=0.0)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return Score(exact_match=exact_match, f1=0.0, precision=0.0, recall=0.0)

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(ground_truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return Score(exact_match=exact_match, f1=f1, precision=precision, recall=recall)


def supporting_fact_score(
    prediction: Iterable[SupportingFact],
    ground_truth: Iterable[SupportingFact],
) -> Score:
    predicted = set(prediction)
    gold = set(ground_truth)
    true_positives = len(predicted & gold)
    false_positives = len(predicted - gold)
    false_negatives = len(gold - predicted)
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact_match = float(not false_positives and not false_negatives)
    return Score(exact_match=exact_match, f1=f1, precision=precision, recall=recall)


def aggregate_answer_metrics(
    predictions: Mapping[str, str],
    examples: Sequence[HotpotExample],
) -> AggregateAnswerMetrics:
    scores: list[Score] = []
    missing: list[str] = []
    for example in examples:
        if example.answer is None:
            raise ValueError(f"{example.identifier}: ground-truth answer is unavailable")
        prediction = predictions.get(example.identifier)
        if prediction is None:
            missing.append(example.identifier)
            prediction = ""
        scores.append(answer_score(prediction, example.answer))

    if not scores:
        raise ValueError("at least one example is required")
    return AggregateAnswerMetrics(
        count=len(scores),
        exact_match=fmean(score.exact_match for score in scores),
        f1=fmean(score.f1 for score in scores),
        precision=fmean(score.precision for score in scores),
        recall=fmean(score.recall for score in scores),
        missing_ids=tuple(missing),
    )


def evaluate_hotpotqa(
    answers: Mapping[str, str],
    supporting_facts: Mapping[str, Sequence[SupportingFact | Sequence[Any]]],
    examples: Sequence[HotpotExample],
) -> AggregateHotpotMetrics:
    """Evaluate predictions with the official answer, support, and joint formulas."""

    if not examples:
        raise ValueError("at least one example is required")

    answer_scores: list[Score] = []
    supporting_scores: list[Score] = []
    joint_exact_matches: list[float] = []
    joint_f1s: list[float] = []
    joint_precisions: list[float] = []
    joint_recalls: list[float] = []
    missing_answers: list[str] = []
    missing_supporting_facts: list[str] = []

    for example in examples:
        if example.answer is None:
            raise ValueError(f"{example.identifier}: ground-truth answer is unavailable")

        predicted_answer = answers.get(example.identifier)
        has_answer = predicted_answer is not None
        if not has_answer:
            missing_answers.append(example.identifier)
            predicted_answer = ""
        answer = answer_score(predicted_answer, example.answer)
        answer_scores.append(answer)

        raw_supporting = supporting_facts.get(example.identifier)
        has_supporting = raw_supporting is not None
        if not has_supporting:
            missing_supporting_facts.append(example.identifier)
            raw_supporting = ()
        predicted_supporting = tuple(
            _coerce_supporting_fact(item, example.identifier)
            for item in raw_supporting
        )
        supporting = supporting_fact_score(
            predicted_supporting, example.supporting_facts
        )
        supporting_scores.append(supporting)

        # The official scorer only accumulates joint metrics when both fields exist.
        if has_answer and has_supporting:
            joint_precision = answer.precision * supporting.precision
            joint_recall = answer.recall * supporting.recall
            joint_f1 = (
                2 * joint_precision * joint_recall
                / (joint_precision + joint_recall)
                if joint_precision + joint_recall
                else 0.0
            )
            joint_exact_match = answer.exact_match * supporting.exact_match
        else:
            joint_precision = joint_recall = joint_f1 = joint_exact_match = 0.0

        joint_precisions.append(joint_precision)
        joint_recalls.append(joint_recall)
        joint_f1s.append(joint_f1)
        joint_exact_matches.append(joint_exact_match)

    return AggregateHotpotMetrics(
        count=len(examples),
        answer_exact_match=fmean(score.exact_match for score in answer_scores),
        answer_f1=fmean(score.f1 for score in answer_scores),
        answer_precision=fmean(score.precision for score in answer_scores),
        answer_recall=fmean(score.recall for score in answer_scores),
        supporting_exact_match=fmean(
            score.exact_match for score in supporting_scores
        ),
        supporting_f1=fmean(score.f1 for score in supporting_scores),
        supporting_precision=fmean(score.precision for score in supporting_scores),
        supporting_recall=fmean(score.recall for score in supporting_scores),
        joint_exact_match=fmean(joint_exact_matches),
        joint_f1=fmean(joint_f1s),
        joint_precision=fmean(joint_precisions),
        joint_recall=fmean(joint_recalls),
        missing_answer_ids=tuple(missing_answers),
        missing_supporting_fact_ids=tuple(missing_supporting_facts),
    )


def _coerce_supporting_fact(
    raw: SupportingFact | Sequence[Any], identifier: str
) -> SupportingFact:
    if isinstance(raw, SupportingFact):
        return raw
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        if len(raw) == 2 and isinstance(raw[0], str) and isinstance(raw[1], int):
            return SupportingFact(title=raw[0], sentence_id=raw[1])
    raise ValueError(
        f"{identifier}: predicted supporting fact must be [title, sentence_id]"
    )
