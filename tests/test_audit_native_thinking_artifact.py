from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import pytest

from hierarchical_rag.audit_native_thinking_artifact import (
    _validate_complete_outputs,
    summarize_records,
)
from hierarchical_rag.hotpotqa import HotpotExample
from hierarchical_rag.metrics import answer_score
from hierarchical_rag.native_thinking import detect_native_reasoning
from hierarchical_rag.qwen_baseline import extract_final_answer


def _prediction(example_id: str, answer: str, generated_tokens: int) -> dict:
    raw_output = f"<think>Evidence supports the answer.</think>\nAnswer: {answer}"
    extraction = extract_final_answer(raw_output)
    reasoning = detect_native_reasoning(raw_output)
    prompt = f"Question for {example_id}"
    return {
        "example_id": example_id,
        "prediction": extraction.answer,
        "extraction": asdict(extraction),
        "reasoning_detection": asdict(reasoning),
        "raw_output": raw_output,
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "input_tokens": 100,
        "generated_tokens": generated_tokens,
        "budget_exhausted": generated_tokens >= 32,
        "latency_seconds": 2.0,
        "answer_score": asdict(answer_score(extraction.answer or "", answer)),
    }


def test_summarize_records_recovers_failed_gate_without_model_rerun():
    examples = [
        HotpotExample(
            identifier="a",
            question="Q1",
            answer="Paris",
            question_type=None,
            level=None,
            supporting_facts=(),
            context=(),
        ),
        HotpotExample(
            identifier="b",
            question="Q2",
            answer="Rome",
            question_type=None,
            level=None,
            supporting_facts=(),
            context=(),
        ),
    ]
    predictions = [
        _prediction("a", "Paris", 10),
        _prediction("b", "Rome", 32),
    ]
    retrieval = [
        {
            "example_id": example.identifier,
            "dropped_document_count": 0,
            "dropped_sentence_count": 0,
            "truncated": False,
        }
        for example in examples
    ]

    summary = summarize_records(
        config={"prompt": {"max_new_tokens": 32}},
        predictions=predictions,
        retrieval=retrieval,
        examples=examples,
    )

    assert summary["interface_gate_passed"] is False
    assert summary["budget_exhausted_ids"] == ["b"]
    assert summary["valid_extraction_count"] == 2
    assert summary["explicit_reasoning_count"] == 2
    assert summary["answer_exact_match"] == 1.0


def test_complete_output_validation_matches_raw_summary(tmp_path):
    recovered = {
        "interface_gate_passed": False,
        "evaluated_count": 2,
        "valid_extraction_count": 1,
        "budget_exhausted_count": 1,
        "explicit_reasoning_count": 2,
        "reasoning_detection_methods": {"closed_think_block": 2},
        "truncated_example_count": 0,
        "answer_exact_match": 0.5,
        "answer_f1": 0.5,
        "answer_precision": 0.5,
        "answer_recall": 0.5,
        "generated_tokens_total": 42,
        "generated_tokens_mean": 21.0,
        "generation_latency_sum_seconds": 4.0,
        "generation_latency_mean_seconds": 2.0,
    }
    config = {
        "experiment": {"id": "example", "seeds": [0]},
        "model": {
            "id": "model",
            "revision": "a" * 40,
            "architecture": "architecture",
            "backend": "backend",
            "torch_version": "2.5.1+cu121",
            "transformers_version": "5.9.0",
            "parameter_count_expected": 100,
            "checkpoint_tensor_element_count": 110,
            "mtp_checkpoint_tensor_element_count": 10,
            "max_position_embeddings": 4096,
        },
        "prompt": {
            "max_input_tokens": 2048,
            "max_new_tokens": 2048,
            "decoding": {"do_sample": True},
        },
    }
    reported = {
        "experiment_id": "example",
        "status": "exploratory_train_only",
        "interface_gate_passed": False,
        "evaluated_count": 2,
        "valid_extraction_count": 1,
        "valid_extraction_rate": 0.5,
        "invalid_output_count": 1,
        "budget_exhausted_count": 1,
        "explicit_reasoning_count": 2,
        "explicit_reasoning_rate": 1.0,
        "reasoning_detection_methods": {"closed_think_block": 2},
        "truncated_example_count": 0,
        "answer_exact_match": 0.5,
        "answer_f1": 0.5,
        "answer_precision": 0.5,
        "answer_recall": 0.5,
        "claim_eligibility": (
            "none; D013/D014/D017 final train-only compatibility gate"
        ),
        "runtime": {
            "evaluation_seconds": 5.0,
            "throughput_examples_per_second": 0.4,
            "generated_tokens_total": 42,
            "generated_tokens_mean": 21.0,
            "latency_mean_seconds": 2.0,
            "peak_allocated_bytes": 1,
            "peak_reserved_bytes": 2,
        },
    }
    environment = {
        "torch": "2.5.1+cu121",
        "transformers": "5.9.0",
        "model_id": "model",
        "model_revision": "a" * 40,
        "model_type": "architecture",
        "model_backend": "backend",
        "model_dtype": "torch.bfloat16",
        "parameter_count": 100,
        "checkpoint_tensor_element_count": 110,
        "mtp_checkpoint_tensor_element_count": 10,
        "language_model_parameter_count": 90,
        "max_position_embeddings": 4096,
        "decoding": {"do_sample": True},
        "presence_penalty_semantics": "generated_tokens_only_v1",
        "seed": 0,
        "pip_freeze": "package==1",
        "nvidia_smi": "GPU",
    }
    (tmp_path / "statistics.json").write_text(
        json.dumps({"status": "not_applicable"}), encoding="utf-8"
    )
    (tmp_path / "environment.txt").write_text(
        json.dumps(environment), encoding="utf-8"
    )

    _validate_complete_outputs(
        run_dir=tmp_path,
        config=config,
        reported_metrics=reported,
        recovered=recovered,
    )

    config["prompt"]["max_new_tokens"] = 4096
    reported["claim_eligibility"] = (
        "none; D023 exploratory train-only expanded-output "
        "budget-sensitivity study"
    )
    _validate_complete_outputs(
        run_dir=tmp_path,
        config=config,
        reported_metrics=reported,
        recovered=recovered,
    )

    reported["answer_f1"] = 0.75
    with pytest.raises(ValueError, match="answer_f1"):
        _validate_complete_outputs(
            run_dir=tmp_path,
            config=config,
            reported_metrics=reported,
            recovered=recovered,
        )
