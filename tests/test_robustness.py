from __future__ import annotations

import pytest

from hierarchical_rag.retrieval import Document
from hierarchical_rag.robustness import append_distractors, select_distractors


def _document(identifier: str) -> Document:
    return Document(identifier, identifier.title(), f"Text for {identifier}.")


def test_distractor_selection_is_repeatable_and_excludes_base():
    base = (_document("gold-a"), _document("gold-b"))
    candidates = tuple(_document(name) for name in ("gold-a", "x", "y", "z"))

    first = select_distractors(
        base, candidates, count=2, seed=42, example_id="question-1"
    )
    second = select_distractors(
        base, reversed(candidates), count=2, seed=42, example_id="question-1"
    )

    assert first == second
    assert not {item.identifier for item in first} & {"gold-a", "gold-b"}


def test_append_preserves_evidence_order():
    base = (_document("gold-a"), _document("gold-b"))
    candidates = (_document("x"), _document("y"))

    merged = append_distractors(
        base, candidates, count=1, seed=5, example_id="question-1"
    )

    assert merged[:2] == base
    assert len(merged) == 3


def test_selection_fails_when_noise_pool_is_too_small():
    with pytest.raises(ValueError, match="not enough"):
        select_distractors(
            (_document("gold"),),
            (_document("x"),),
            count=2,
            seed=1,
            example_id="question-1",
        )
