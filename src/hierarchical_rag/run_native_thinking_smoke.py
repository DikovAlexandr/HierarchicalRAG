"""Run a D013 native-thinking reader on the fixed D009 train slice."""

from __future__ import annotations

import argparse
import hashlib
import platform
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from hierarchical_rag.hotpotqa import gold_paragraphs
from hierarchical_rag.metrics import aggregate_answer_metrics, answer_score
from hierarchical_rag.native_thinking import (
    detect_native_reasoning,
    validate_native_thinking_protocol,
)
from hierarchical_rag.qwen_baseline import (
    build_cot_chat_prompt,
    extract_final_answer,
)
from hierarchical_rag.run_qwen_baseline_smoke import execute_baseline_smoke


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a preregistered native-thinking train-only smoke."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return execute_baseline_smoke(
        config_path=args.config.resolve(),
        validate_protocol=validate_native_thinking_protocol,
        run_model=_run_model,
    )


def _run_model(
    *,
    run_dir: Path,
    experiment_id: str,
    demonstrations: Sequence[Any],
    targets: Sequence[Any],
    model_config: Mapping[str, Any],
    prompt_config: Mapping[str, Any],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    import transformers
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForMultimodalLM,
        AutoProcessor,
        AutoTokenizer,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if transformers.__version__ != model_config["transformers_version"]:
        raise RuntimeError("Transformers version differs from the config")
    if torch.__version__ != model_config["torch_version"]:
        raise RuntimeError("PyTorch version differs from the config")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    backend = model_config["backend"]
    load_started = time.perf_counter()
    if backend == "causal_lm_tokenizer":
        frontend = AutoTokenizer.from_pretrained(
            model_config["id"],
            revision=model_config["revision"],
        )
        tokenizer = frontend
        model_loader = AutoModelForCausalLM
    elif backend == "multimodal_lm_processor_text_only":
        frontend = AutoProcessor.from_pretrained(
            model_config["id"],
            revision=model_config["revision"],
        )
        tokenizer = frontend.tokenizer
        model_loader = AutoModelForMultimodalLM
    else:
        raise ValueError(f"unsupported native-thinking backend: {backend}")

    model = model_loader.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        dtype=torch.bfloat16,
    ).cuda().eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    resolved_revision = getattr(model.config, "_commit_hash", None)
    if resolved_revision != model_config["revision"]:
        raise RuntimeError("resolved model revision differs from the config")
    if model.config.model_type != model_config["architecture"]:
        raise RuntimeError("loaded model architecture differs from the config")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != int(model_config["parameter_count_expected"]):
        raise RuntimeError(
            "model parameter count differs from the config: "
            f"observed={parameter_count}, "
            f"expected={model_config['parameter_count_expected']}"
        )
    max_position_embeddings = _max_position_embeddings(model.config)
    if max_position_embeddings != int(model_config["max_position_embeddings"]):
        raise RuntimeError("model context length differs from the config")
    language_model_parameter_count = _language_model_parameter_count(model)

    def render_chat(messages: Sequence[Mapping[str, str]]) -> str:
        return frontend.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
            **dict(prompt_config["chat_template_kwargs"]),
        )

    predictions: dict[str, str] = {}
    latencies: list[float] = []
    generated_token_counts: list[int] = []
    invalid_count = 0
    budget_exhausted_count = 0
    explicit_reasoning_count = 0
    truncated_count = 0
    peak_allocated = 0
    peak_reserved = 0
    reasoning_methods: Counter[str] = Counter()
    started = time.perf_counter()
    with (run_dir / "predictions.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as prediction_stream, (run_dir / "retrieval.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as retrieval_stream:
        for target in targets:
            evidence = gold_paragraphs(target)
            built = build_cot_chat_prompt(
                demonstrations,
                target,
                evidence,
                render_chat=render_chat,
                token_count=lambda text: len(
                    tokenizer(text, add_special_tokens=False)["input_ids"]
                ),
                max_input_tokens=int(prompt_config["max_input_tokens"]),
                demonstration_rationale=prompt_config["demonstration_rationale"],
            )
            inputs = tokenizer(
                built.prompt,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(model.device)
            actual_input_tokens = int(inputs["input_ids"].shape[-1])
            if actual_input_tokens != built.input_tokens:
                raise RuntimeError("prompt token count changed before inference")

            generation_kwargs = _generation_kwargs(
                prompt_config=prompt_config,
                tokenizer=tokenizer,
                prompt_length=actual_input_tokens,
                torch_module=torch,
                transformers_module=transformers,
            )
            torch.cuda.reset_peak_memory_stats()
            generation_started = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(**inputs, **generation_kwargs)
            torch.cuda.synchronize()
            latency = time.perf_counter() - generation_started
            generated_ids = output[0, actual_input_tokens:]
            generated_tokens = int(generated_ids.shape[-1])
            raw_output = tokenizer.decode(
                generated_ids,
                skip_special_tokens=False,
            )
            extraction = extract_final_answer(raw_output)
            reasoning = detect_native_reasoning(raw_output)
            prediction = extraction.answer or ""
            predictions[target.identifier] = prediction
            reasoning_methods[reasoning.method] += 1
            if extraction.status != "ok":
                invalid_count += 1
            if generated_tokens >= int(prompt_config["max_new_tokens"]):
                budget_exhausted_count += 1
            if reasoning.present:
                explicit_reasoning_count += 1
            if built.truncated:
                truncated_count += 1
            latencies.append(latency)
            generated_token_counts.append(generated_tokens)
            peak_allocated = max(peak_allocated, int(torch.cuda.max_memory_allocated()))
            peak_reserved = max(peak_reserved, int(torch.cuda.max_memory_reserved()))
            score = answer_score(prediction, target.answer or "")

            prediction_stream.write(
                _json_line(
                    {
                        "example_id": target.identifier,
                        "gold_answer": target.answer,
                        "prediction": prediction,
                        "extraction": asdict(extraction),
                        "reasoning_detection": asdict(reasoning),
                        "raw_output": raw_output,
                        "prompt": built.prompt,
                        "prompt_sha256": _sha256_text(built.prompt),
                        "input_tokens": actual_input_tokens,
                        "generated_tokens": generated_tokens,
                        "budget_exhausted": generated_tokens
                        >= int(prompt_config["max_new_tokens"]),
                        "latency_seconds": latency,
                        "answer_score": asdict(score),
                    }
                )
            )
            retrieval_stream.write(
                _json_line(
                    {
                        "example_id": target.identifier,
                        "source": "gold_supporting_paragraphs",
                        "original_document_count": len(evidence),
                        "included_document_count": built.included_document_count,
                        "included_sentence_count": built.included_sentence_count,
                        "dropped_document_count": built.dropped_document_count,
                        "dropped_sentence_count": built.dropped_sentence_count,
                        "truncated": built.truncated,
                        "documents": [
                            {
                                "rank": index,
                                "title": paragraph.title,
                                "sentences": list(paragraph.sentences),
                            }
                            for index, paragraph in enumerate(
                                built.included_paragraphs,
                                start=1,
                            )
                        ],
                    }
                )
            )

    elapsed = time.perf_counter() - started
    aggregate = aggregate_answer_metrics(predictions, targets)
    evaluated_count = len(targets)
    interface_gate_passed = all(
        (
            invalid_count == 0,
            explicit_reasoning_count == evaluated_count,
            budget_exhausted_count == 0,
            truncated_count == 0,
        )
    )
    device = torch.cuda.get_device_properties(0)
    metrics = {
        "experiment_id": experiment_id,
        "status": "exploratory_train_only",
        "interface_gate_passed": interface_gate_passed,
        "evaluated_count": evaluated_count,
        "valid_extraction_count": evaluated_count - invalid_count,
        "valid_extraction_rate": (evaluated_count - invalid_count) / evaluated_count,
        "invalid_output_count": invalid_count,
        "budget_exhausted_count": budget_exhausted_count,
        "explicit_reasoning_count": explicit_reasoning_count,
        "explicit_reasoning_rate": explicit_reasoning_count / evaluated_count,
        "reasoning_detection_methods": dict(sorted(reasoning_methods.items())),
        "truncated_example_count": truncated_count,
        "answer_exact_match": aggregate.exact_match,
        "answer_f1": aggregate.f1,
        "answer_precision": aggregate.precision,
        "answer_recall": aggregate.recall,
        "runtime": {
            "model_load_seconds": load_seconds,
            "evaluation_seconds": elapsed,
            "throughput_examples_per_second": evaluated_count / elapsed,
            "latency_mean_seconds": fmean(latencies),
            "generated_tokens_total": sum(generated_token_counts),
            "generated_tokens_mean": fmean(generated_token_counts),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
        "claim_eligibility": (
            "none; D013/D014/D017 final train-only compatibility gate"
            if "D017" in experiment_config["decision_ids"]
            else "none; D013/D014 train-only compatibility gate"
        ),
    }
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "pip_freeze": _pip_freeze(),
        "nvidia_smi": _nvidia_smi(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "model_id": model_config["id"],
        "model_revision": resolved_revision,
        "model_type": model.config.model_type,
        "model_backend": backend,
        "model_dtype": str(model.dtype),
        "parameter_count": parameter_count,
        "checkpoint_tensor_element_count": int(
            model_config["checkpoint_tensor_element_count"]
        ),
        "mtp_checkpoint_tensor_element_count": int(
            model_config.get("mtp_checkpoint_tensor_element_count", 0)
        ),
        "language_model_parameter_count": language_model_parameter_count,
        "max_position_embeddings": max_position_embeddings,
        "decoding": dict(prompt_config["decoding"]),
        "presence_penalty_semantics": "generated_tokens_only_v1",
        "seed": seed,
        "gpu_name": device.name,
        "gpu_total_memory_bytes": int(device.total_memory),
        "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
    }
    return metrics, environment


def _generation_kwargs(
    *,
    prompt_config: Mapping[str, Any],
    tokenizer: Any,
    prompt_length: int,
    torch_module: Any,
    transformers_module: Any,
) -> dict[str, Any]:
    decoding = dict(prompt_config["decoding"])
    presence_penalty = float(decoding.pop("presence_penalty", 0.0))
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    kwargs: dict[str, Any] = {
        "max_new_tokens": int(prompt_config["max_new_tokens"]),
        "pad_token_id": pad_token_id,
        **decoding,
    }
    if presence_penalty:
        processor = _GeneratedTokenPresencePenalty(
            torch_module=torch_module,
            penalty=presence_penalty,
            prompt_length=prompt_length,
        )
        kwargs["logits_processor"] = transformers_module.LogitsProcessorList(
            [processor]
        )
    return kwargs


class _GeneratedTokenPresencePenalty:
    """Subtract a fixed penalty from tokens already generated in this response."""

    def __init__(self, *, torch_module: Any, penalty: float, prompt_length: int):
        if penalty < 0:
            raise ValueError("presence penalty must be non-negative")
        if prompt_length < 0:
            raise ValueError("prompt length must be non-negative")
        self._torch = torch_module
        self._penalty = penalty
        self._prompt_length = prompt_length

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        generated_ids = input_ids[:, self._prompt_length :]
        if generated_ids.numel() == 0:
            return scores
        seen = self._torch.zeros_like(scores, dtype=self._torch.bool)
        seen.scatter_(1, generated_ids, True)
        return scores - seen.to(dtype=scores.dtype) * self._penalty


def _max_position_embeddings(config: Any) -> int:
    text_config = getattr(config, "text_config", None)
    value = getattr(text_config, "max_position_embeddings", None)
    if value is None:
        value = getattr(config, "max_position_embeddings", None)
    if value is None:
        raise RuntimeError("model config does not declare max_position_embeddings")
    return int(value)


def _language_model_parameter_count(model: Any) -> int | None:
    inner = getattr(model, "model", None)
    language_model = getattr(inner, "language_model", None)
    if language_model is None:
        return None
    return sum(parameter.numel() for parameter in language_model.parameters())


def _json_line(value: Mapping[str, Any]) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _pip_freeze() -> str:
    return subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _nvidia_smi() -> str:
    return subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,compute_cap",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
