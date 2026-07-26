from __future__ import annotations

import hashlib
from dataclasses import asdict

from hierarchical_rag.audit_native_thinking_artifact import summarize_records
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
