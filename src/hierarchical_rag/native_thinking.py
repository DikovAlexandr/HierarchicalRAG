"""Frozen protocol and output checks for native-thinking CoT readers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ReasoningDetection:
    present: bool
    method: str
    content: str


MODEL_PROTOCOLS: dict[str, dict[str, Any]] = {
    "LiquidAI/LFM2.5-1.2B-Thinking": {
        "revision": "95053d21d8e0b7ca99421a2127ae39c64f685ff3",
        "architecture": "lfm2",
        "backend": "causal_lm_tokenizer",
        "role": "primary_size_matched_cot_baseline",
        "parameter_count_expected": 1_170_340_608,
        "checkpoint_tensor_element_count": 1_170_340_608,
        "max_position_embeddings": 128_000,
        "chat_template_kwargs": {},
        "decoding": {
            "do_sample": True,
            "temperature": 0.05,
            "top_k": 50,
            "repetition_penalty": 1.05,
        },
    },
    "Qwen/Qwen3.5-2B": {
        "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
        "architecture": "qwen3_5",
        "backend": "multimodal_lm_processor_text_only",
        "role": "contemporary_cot_reference",
        "parameter_count_expected": 2_213_241_664,
        "checkpoint_tensor_element_count": 2_274_069_824,
        "mtp_checkpoint_tensor_element_count": 60_828_160,
        "max_position_embeddings": 262_144,
        "chat_template_kwargs": {"enable_thinking": True},
        "decoding": {
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
        },
    },
    "Qwen/Qwen3.5-0.8B": {
        "revision": "2fc06364715b967f1860aea9cf38778875588b17",
        "architecture": "qwen3_5",
        "backend": "multimodal_lm_processor_text_only",
        "role": "lower_scale_cot_control",
        "parameter_count_expected": 852_985_920,
        "checkpoint_tensor_element_count": 873_438_784,
        "mtp_checkpoint_tensor_element_count": 20_452_864,
        "max_position_embeddings": 262_144,
        "chat_template_kwargs": {"enable_thinking": True},
        "decoding": {
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
        },
    },
}


def detect_native_reasoning(generated_text: str) -> ReasoningDetection:
    """Detect non-empty native thinking without counting markup as reasoning."""

    text = generated_text
    for marker in ("<|im_end|>", "<|endoftext|>"):
        text = text.split(marker, 1)[0]

    if "</think>" in text:
        reasoning = text.split("</think>", 1)[0]
        if "<think>" in reasoning:
            reasoning = reasoning.rsplit("<think>", 1)[-1]
        content = _strip_control_tokens(reasoning)
        return ReasoningDetection(bool(content), "closed_think_block", content)

    if "<think>" in text:
        content = _strip_control_tokens(text.rsplit("<think>", 1)[-1])
        return ReasoningDetection(bool(content), "unclosed_think_block", content)

    answer_match = re.search(
        r"(?im)^\s*(?:final\s+)?answer\s*:",
        text,
    )
    prefix = text[: answer_match.start()] if answer_match else text
    content = _strip_control_tokens(prefix)
    return ReasoningDetection(bool(content), "pre_answer_text", content)


def validate_native_thinking_protocol(config: Mapping[str, Any]) -> None:
    """Reject any unrecorded change to the D013/D014 train-only gate."""

    experiment = config["experiment"]
    dataset = config["dataset"]
    model = config["model"]
    prompt = config["prompt"]
    reader = config["reader"]
    runtime = config["runtime"]

    decision_ids = set(experiment["decision_ids"])
    stage = experiment["stage"]
    if stage not in {
        "train_only_native_thinking_smoke",
        "train_only_native_thinking_final_gate",
    }:
        raise ValueError("native-thinking runner is restricted to its train-only gates")
    is_final_gate = stage == "train_only_native_thinking_final_gate"
    if is_final_gate != ("D017" in decision_ids):
        raise ValueError("only the final 2048+2048 gate may cite D017")
    if not {"D013", "D014"} <= decision_ids:
        raise ValueError("native-thinking configs must cite D013 and D014")
    if experiment["seeds"] != [0]:
        raise ValueError("the compatibility gate must run exactly once with seed 0")
    if dataset["split"] != "train" or dataset["evidence"] != (
        "gold_supporting_paragraphs_in_original_context_order"
    ):
        raise ValueError("native-thinking development is restricted to train gold evidence")
    if int(dataset["selection"]["demonstration_count"]) != 2 or int(
        dataset["selection"]["evaluated_count"]
    ) != 16:
        raise ValueError("the compatibility gate requires 2 demos and 16 targets")
    example_ids = list(dataset["selection"]["example_ids"])
    if len(example_ids) != 18 or len(set(example_ids)) != 18:
        raise ValueError("the frozen train slice must contain 18 unique examples")

    model_id = model["id"]
    if model_id not in MODEL_PROTOCOLS:
        raise ValueError(f"model is not preregistered by D013: {model_id}")
    if model_id.startswith("Qwen/Qwen3.5-") and not {"D015", "D016"} <= decision_ids:
        raise ValueError("Qwen3.5 configs with corrected counts must cite D015/D016")
    expected = MODEL_PROTOCOLS[model_id]
    for field in (
        "revision",
        "architecture",
        "backend",
        "role",
        "parameter_count_expected",
        "max_position_embeddings",
    ):
        if model[field] != expected[field]:
            raise ValueError(f"{model_id}: {field} differs from D013")
    checkpoint_elements = model.get(
        "checkpoint_tensor_element_count",
        model["parameter_count_expected"],
    )
    if checkpoint_elements != expected["checkpoint_tensor_element_count"]:
        raise ValueError(f"{model_id}: checkpoint tensor count differs from D015")
    if model_id.startswith("Qwen/Qwen3.5-") and model.get(
        "mtp_checkpoint_tensor_element_count"
    ) != expected["mtp_checkpoint_tensor_element_count"]:
        raise ValueError(f"{model_id}: MTP tensor count differs from D016")
    if not model["frozen"] or model["dtype"] != "bfloat16":
        raise ValueError("D013 requires a frozen BF16 checkpoint")

    if prompt["style"] != "few_shot_supporting_fact_cot_final_answer":
        raise ValueError("native-thinking gate requires the frozen CoT prompt style")
    if prompt["demonstration_rationale"] != "supporting_fact_sentences":
        raise ValueError("native-thinking gate requires deterministic demo rationales")
    if int(prompt["demonstration_count"]) != 2 or list(
        prompt["demonstration_ids"]
    ) != example_ids[:2]:
        raise ValueError("prompt demonstrations must be the first two frozen examples")
    if prompt["final_answer_format"] != "Answer: <shortest answer>":
        raise ValueError("final-answer instruction differs from the shared contract")
    if prompt["target_truncation"] != "sentence_boundary_from_end_only":
        raise ValueError("target truncation differs from the shared contract")
    if prompt["chat_template_kwargs"] != expected["chat_template_kwargs"]:
        raise ValueError("chat-template thinking mode differs from D013")
    expected_allocation = (2048, 2048) if is_final_gate else (3584, 512)
    if (int(prompt["max_input_tokens"]), int(prompt["max_new_tokens"])) != (
        expected_allocation
    ):
        label = "D017" if is_final_gate else "D014"
        allocation = "+".join(str(value) for value in expected_allocation)
        raise ValueError(f"{label} requires the shared {allocation} token allocation")
    if int(prompt["total_reader_tokens"]) != 4096:
        raise ValueError("D014 requires a 4096-token total reader ceiling")
    if int(prompt["total_reader_tokens"]) > int(model["max_position_embeddings"]):
        raise ValueError("reader budget exceeds the pinned model context window")
    if prompt["decoding"] != expected["decoding"]:
        raise ValueError("decoding parameters differ from the pinned native profile")

    if reader["type"] != "frozen_native_thinking_cot":
        raise ValueError("reader type must identify the native-thinking interface")
    if reader["answer_extraction"] != "last_declared_answer_line_v1":
        raise ValueError("answer extraction differs from the shared frozen parser")
    if reader["reasoning_detection"] != "native_think_or_pre_answer_v1":
        raise ValueError("reasoning detection differs from the native gate contract")
    if reader["semantic_postprocessing"] != "forbidden":
        raise ValueError("semantic answer postprocessing is forbidden")
    if reader["presence_penalty_semantics"] != "generated_tokens_only_v1":
        raise ValueError("presence-penalty semantics must be explicit and frozen")
    if runtime["deterministic_decoding"] is not False:
        raise ValueError("native samplers are stochastic even when their seed is fixed")
    if int(runtime["fixed_sampling_seed"]) != int(experiment["seeds"][0]):
        raise ValueError("runtime sampling seed must match the preregistered seed")


def _strip_control_tokens(value: str) -> str:
    value = re.sub(r"<\|[^>]+\|>", " ", value)
    value = value.replace("<think>", " ").replace("</think>", " ")
    return " ".join(value.split())
