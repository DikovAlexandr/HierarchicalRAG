from __future__ import annotations

from dataclasses import replace

import pytest

from hierarchical_rag.hotpotqa import HotpotExample, Paragraph, gold_paragraphs
from hierarchical_rag.qwen_baseline import (
    SYSTEM_INSTRUCTION,
    build_cot_chat_prompt,
    extract_final_answer,
    has_explicit_reasoning,
)
from hierarchical_rag.run_qwen_baseline_smoke import _validate_frozen_protocol


def _examples(hotpot_records) -> tuple[HotpotExample, ...]:
    return tuple(HotpotExample.from_mapping(record) for record in hotpot_records)


def _render(messages) -> str:
    blocks = [f"<{item['role']}>\n{item['content']}" for item in messages]
    return "\n".join(blocks) + "\n<assistant>\n"


def test_chat_prompt_uses_same_evidence_and_hides_target_answer(hotpot_records):
    first, second, target = _examples(hotpot_records)

    built = build_cot_chat_prompt(
        (first, second),
        target,
        gold_paragraphs(target),
        render_chat=_render,
        token_count=len,
        max_input_tokens=10_000,
    )

    assert SYSTEM_INSTRUCTION in built.prompt
    assert "Answer: Paris" in built.prompt
    assert "Answer: yes" in built.prompt
    assert f"Answer: {target.answer}" not in built.prompt
    assert built.truncated is False


def test_chat_prompt_truncates_only_target_on_sentence_boundaries(hotpot_records):
    first, second, target = _examples(hotpot_records)
    target = replace(
        target,
        context=(Paragraph("Mars", ("Mars is red.", "Drop this sentence.")),),
        supporting_facts=(),
    )
    one_sentence = replace(
        target,
        context=(Paragraph("Mars", ("Mars is red.",)),),
    )
    budget = build_cot_chat_prompt(
        (first, second),
        one_sentence,
        one_sentence.context,
        render_chat=_render,
        token_count=len,
        max_input_tokens=10_000,
    ).input_tokens

    built = build_cot_chat_prompt(
        (first, second),
        target,
        target.context,
        render_chat=_render,
        token_count=len,
        max_input_tokens=budget,
    )

    assert built.truncated is True
    assert built.included_sentence_count == 1
    assert "Mars is red." in built.prompt
    assert "Drop this sentence." not in built.prompt


def test_final_answer_parser_uses_last_declared_answer_line():
    extracted = extract_final_answer(
        "First I combine both documents.\nAnswer: draft\nFinal Answer: defender<|im_end|>"
    )

    assert extracted.answer == "defender"
    assert extracted.status == "ok"
    assert extracted.method == "last_answer_line"


def test_final_answer_parser_unwraps_latex_box():
    extracted = extract_final_answer("Reasoning.\nAnswer: $\\boxed{Paris}$")

    assert extracted.answer == "Paris"


def test_final_answer_parser_rejects_unmarked_output():
    extracted = extract_final_answer("The answer appears to be Paris.")

    assert extracted.answer is None
    assert extracted.status == "missing_final_answer"


def test_final_answer_parser_records_empty_output():
    extracted = extract_final_answer(" <|im_end|> ")

    assert extracted.answer is None
    assert extracted.status == "empty"


def test_explicit_reasoning_detection_excludes_answer_only_output():
    assert has_explicit_reasoning("Combine the two facts.\nAnswer: Paris") is True
    assert has_explicit_reasoning("Answer: Paris<|im_end|>") is False


def test_baseline_protocol_rejects_a_larger_generation_budget():
    config = {
        "experiment": {"stage": "train_only_baseline_smoke"},
        "dataset": {
            "split": "train",
            "evidence": "gold_supporting_paragraphs_in_original_context_order",
        },
        "model": {
            "role": "primary_size_matched_cot_baseline",
            "frozen": True,
            "dtype": "bfloat16",
            "max_position_embeddings": 32768,
        },
        "prompt": {
            "style": "brief_explicit_cot_with_final_answer",
            "do_sample": False,
            "max_input_tokens": 4032,
            "max_new_tokens": 65,
        },
    }

    with pytest.raises(ValueError, match=r"4032\+64"):
        _validate_frozen_protocol(config)
