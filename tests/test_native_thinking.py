from __future__ import annotations

from copy import deepcopy

import pytest

from hierarchical_rag.native_thinking import (
    MODEL_PROTOCOLS,
    detect_native_reasoning,
    validate_native_thinking_protocol,
)
from hierarchical_rag.run_native_thinking_smoke import (
    _claim_eligibility,
    _example_progress,
    _generation_seed,
)


def _config(model_id: str) -> dict:
    protocol = MODEL_PROTOCOLS[model_id]
    example_ids = [f"example-{index:02d}" for index in range(18)]
    decision_ids = ["D001", "D013", "D014"]
    if model_id.startswith("Qwen/Qwen3.5-"):
        decision_ids.extend(("D015", "D016"))
    return {
        "experiment": {
            "stage": "train_only_native_thinking_smoke",
            "decision_ids": decision_ids,
            "seeds": [0],
        },
        "dataset": {
            "split": "train",
            "evidence": "gold_supporting_paragraphs_in_original_context_order",
            "selection": {
                "demonstration_count": 2,
                "evaluated_count": 16,
                "example_ids": example_ids,
            },
        },
        "model": {
            "id": model_id,
            "revision": protocol["revision"],
            "architecture": protocol["architecture"],
            "backend": protocol["backend"],
            "role": protocol["role"],
            "parameter_count_expected": protocol["parameter_count_expected"],
            "checkpoint_tensor_element_count": protocol[
                "checkpoint_tensor_element_count"
            ],
            "mtp_checkpoint_tensor_element_count": protocol.get(
                "mtp_checkpoint_tensor_element_count",
                0,
            ),
            "max_position_embeddings": protocol["max_position_embeddings"],
            "frozen": True,
            "dtype": "bfloat16",
        },
        "prompt": {
            "style": "few_shot_supporting_fact_cot_final_answer",
            "demonstration_rationale": "supporting_fact_sentences",
            "demonstration_count": 2,
            "demonstration_ids": example_ids[:2],
            "final_answer_format": "Answer: <shortest answer>",
            "target_truncation": "sentence_boundary_from_end_only",
            "chat_template_kwargs": deepcopy(protocol["chat_template_kwargs"]),
            "max_input_tokens": 3584,
            "max_new_tokens": 512,
            "total_reader_tokens": 4096,
            "decoding": deepcopy(protocol["decoding"]),
        },
        "reader": {
            "type": "frozen_native_thinking_cot",
            "answer_extraction": "last_declared_answer_line_v1",
            "reasoning_detection": "native_think_or_pre_answer_v1",
            "semantic_postprocessing": "forbidden",
            "presence_penalty_semantics": "generated_tokens_only_v1",
        },
        "runtime": {"deterministic_decoding": False, "fixed_sampling_seed": 0},
    }


@pytest.mark.parametrize("model_id", sorted(MODEL_PROTOCOLS))
def test_native_protocol_accepts_every_preregistered_model(model_id):
    validate_native_thinking_protocol(_config(model_id))


@pytest.mark.parametrize("model_id", sorted(MODEL_PROTOCOLS))
def test_native_protocol_accepts_d017_final_gate(model_id):
    config = _config(model_id)
    config["experiment"]["stage"] = "train_only_native_thinking_final_gate"
    config["experiment"]["decision_ids"].append("D017")
    config["prompt"]["max_input_tokens"] = 2048
    config["prompt"]["max_new_tokens"] = 2048

    validate_native_thinking_protocol(config)


def test_native_protocol_rejects_d017_with_old_allocation():
    config = _config("LiquidAI/LFM2.5-1.2B-Thinking")
    config["experiment"]["stage"] = "train_only_native_thinking_final_gate"
    config["experiment"]["decision_ids"].append("D017")

    with pytest.raises(ValueError, match=r"D017.*2048\+2048"):
        validate_native_thinking_protocol(config)


@pytest.mark.parametrize("max_new_tokens", [4096, 8192])
@pytest.mark.parametrize(
    "model_id",
    ["LiquidAI/LFM2.5-1.2B-Thinking", "Qwen/Qwen3.5-2B"],
)
def test_native_protocol_accepts_d023_budget_sensitivity(
    model_id, max_new_tokens
):
    config = _config(model_id)
    config["experiment"]["stage"] = (
        "train_only_native_thinking_budget_sensitivity"
    )
    config["experiment"]["decision_ids"].extend(("D017", "D023"))
    config["prompt"]["max_input_tokens"] = 2048
    config["prompt"]["max_new_tokens"] = max_new_tokens
    config["prompt"]["total_reader_tokens"] = 2048 + max_new_tokens
    config["runtime"]["sampling_seed_scope"] = "per_example_sha256_v1"

    validate_native_thinking_protocol(config)


def test_native_protocol_rejects_unapproved_d023_allocation():
    config = _config("LiquidAI/LFM2.5-1.2B-Thinking")
    config["experiment"]["stage"] = (
        "train_only_native_thinking_budget_sensitivity"
    )
    config["experiment"]["decision_ids"].extend(("D017", "D023"))
    config["prompt"]["max_input_tokens"] = 2048
    config["prompt"]["max_new_tokens"] = 16384
    config["prompt"]["total_reader_tokens"] = 18432
    config["runtime"]["sampling_seed_scope"] = "per_example_sha256_v1"

    with pytest.raises(ValueError, match="4096- or 8192-token output"):
        validate_native_thinking_protocol(config)


def test_native_protocol_rejects_d023_lower_scale_model():
    config = _config("Qwen/Qwen3.5-0.8B")
    config["experiment"]["stage"] = (
        "train_only_native_thinking_budget_sensitivity"
    )
    config["experiment"]["decision_ids"].extend(("D017", "D023"))
    config["prompt"]["max_input_tokens"] = 2048
    config["prompt"]["max_new_tokens"] = 4096
    config["prompt"]["total_reader_tokens"] = 6144
    config["runtime"]["sampling_seed_scope"] = "per_example_sha256_v1"

    with pytest.raises(ValueError, match="two preregistered core models"):
        validate_native_thinking_protocol(config)


def test_native_protocol_rejects_disabled_qwen_thinking():
    config = _config("Qwen/Qwen3.5-2B")
    config["prompt"]["chat_template_kwargs"]["enable_thinking"] = False

    with pytest.raises(ValueError, match="thinking mode"):
        validate_native_thinking_protocol(config)


def test_native_protocol_rejects_qwen_without_count_decision():
    config = _config("Qwen/Qwen3.5-2B")
    config["experiment"]["decision_ids"].remove("D016")

    with pytest.raises(ValueError, match="must cite D015/D016"):
        validate_native_thinking_protocol(config)


def test_native_protocol_rejects_model_specific_extra_output_budget():
    config = _config("LiquidAI/LFM2.5-1.2B-Thinking")
    config["prompt"]["max_new_tokens"] = 513

    with pytest.raises(ValueError, match=r"3584\+512"):
        validate_native_thinking_protocol(config)


def test_native_protocol_rejects_sampler_drift():
    config = _config("Qwen/Qwen3.5-0.8B")
    config["prompt"]["decoding"]["temperature"] = 0.6

    with pytest.raises(ValueError, match="decoding parameters"):
        validate_native_thinking_protocol(config)


def test_native_protocol_rejects_demonstration_drift():
    config = _config("LiquidAI/LFM2.5-1.2B-Thinking")
    config["prompt"]["demonstration_ids"].reverse()

    with pytest.raises(ValueError, match="first two frozen examples"):
        validate_native_thinking_protocol(config)


def test_reasoning_detection_handles_template_prefilled_opening_tag():
    detected = detect_native_reasoning(
        "Combine the first and second facts.\n</think>\nAnswer: Paris<|im_end|>"
    )

    assert detected.present is True
    assert detected.method == "closed_think_block"
    assert detected.content == "Combine the first and second facts."


def test_reasoning_detection_does_not_count_empty_markup():
    detected = detect_native_reasoning("<think>\n</think>\nAnswer: Paris")

    assert detected.present is False
    assert detected.content == ""


def test_reasoning_detection_supports_untagged_cot():
    detected = detect_native_reasoning(
        "The evidence identifies France, then its capital.\nAnswer: Paris"
    )

    assert detected.present is True
    assert detected.method == "pre_answer_text"


def test_example_progress_reports_bounded_bar_and_identity():
    message = _example_progress(
        current=4,
        total=16,
        example_id="example-04",
        status="complete",
        detail="generated_tokens=42",
    )

    assert "[######------------------]" in message
    assert "examples=4/16" in message
    assert "example_id=example-04" in message


def test_per_example_seed_is_stable_and_example_specific():
    first = _generation_seed(
        base_seed=0,
        example_id="example-01",
        scope="per_example_sha256_v1",
    )
    repeated = _generation_seed(
        base_seed=0,
        example_id="example-01",
        scope="per_example_sha256_v1",
    )
    second = _generation_seed(
        base_seed=0,
        example_id="example-02",
        scope="per_example_sha256_v1",
    )

    assert first == repeated
    assert first != second
    assert 0 <= first < 2**64


def test_claim_eligibility_uses_validated_token_allocation():
    assert _claim_eligibility(
        {"max_input_tokens": 2048, "max_new_tokens": 2048}
    ) == "none; D013/D014/D017 final train-only compatibility gate"
    assert _claim_eligibility(
        {"max_input_tokens": 3584, "max_new_tokens": 512}
    ) == "none; D013/D014 train-only compatibility gate"
    assert _claim_eligibility(
        {"max_input_tokens": 2048, "max_new_tokens": 4096}
    ) == (
        "none; D023 exploratory train-only expanded-output "
        "budget-sensitivity study"
    )
    assert _claim_eligibility(
        {"max_input_tokens": 2048, "max_new_tokens": 8192}
    ) == (
        "none; D023 exploratory train-only expanded-output "
        "budget-sensitivity study"
    )
    with pytest.raises(ValueError, match="unsupported native-thinking token allocation"):
        _claim_eligibility(
            {"max_input_tokens": 2048, "max_new_tokens": 512}
        )
