from __future__ import annotations

from dataclasses import replace

from hierarchical_rag.hotpotqa import HotpotExample, Paragraph, gold_paragraphs
from hierarchical_rag.hrm_text import (
    DIRECT_CONDITION,
    build_direct_prompt,
    extract_short_answer,
)


def _examples(hotpot_records) -> tuple[HotpotExample, ...]:
    return tuple(HotpotExample.from_mapping(record) for record in hotpot_records)


def test_direct_prompt_contains_demonstrations_but_not_target_answer(hotpot_records):
    first, second, target = _examples(hotpot_records)

    built = build_direct_prompt(
        (first, second),
        target,
        gold_paragraphs(target),
        token_count=len,
        max_input_tokens=10_000,
    )

    assert built.prompt.startswith(f"<|im_start|>{DIRECT_CONDITION}")
    assert "Answer: Paris" in built.prompt
    assert "Answer: yes" in built.prompt
    assert built.prompt.endswith("Answer:<|im_end|>")
    assert f"Answer: {target.answer}" not in built.prompt
    assert built.truncated is False


def test_direct_prompt_truncates_target_at_sentence_boundary(hotpot_records):
    first, second, target = _examples(hotpot_records)
    target = replace(
        target,
        context=(
            Paragraph("Mars", ("Mars is red.", "This sentence must be dropped.")),
        ),
        supporting_facts=(),
    )
    one_sentence = replace(
        target,
        context=(Paragraph("Mars", ("Mars is red.",)),),
    )
    budget = build_direct_prompt(
        (first, second),
        one_sentence,
        one_sentence.context,
        token_count=len,
        max_input_tokens=10_000,
    ).input_tokens

    built = build_direct_prompt(
        (first, second),
        target,
        target.context,
        token_count=len,
        max_input_tokens=budget,
    )

    assert built.truncated is True
    assert built.included_sentence_count == 1
    assert built.dropped_sentence_count == 1
    assert "Mars is red." in built.prompt
    assert "This sentence must be dropped." not in built.prompt


def test_extract_short_answer_from_nested_latex_box():
    extracted = extract_short_answer(
        "Reasoning. $\\boxed{\\text{Paris}}$<|box_end|>ignored"
    )

    assert extracted.answer == "Paris"
    assert extracted.status == "ok"
    assert extracted.method == "latex_boxed"


def test_extract_short_answer_from_declared_answer_line():
    extracted = extract_short_answer("Answer: Nile\nAdditional text")

    assert extracted.answer == "Nile"
    assert extracted.method == "first_line"


def test_extract_short_answer_preserves_unstructured_first_line():
    extracted = extract_short_answer("The answer may be Paris.\nExplanation")

    assert extracted.answer == "The answer may be Paris."


def test_extract_short_answer_records_empty_output():
    extracted = extract_short_answer("  <|box_end|>  ")

    assert extracted.answer is None
    assert extracted.status == "empty"
