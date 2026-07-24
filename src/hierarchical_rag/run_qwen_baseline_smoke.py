"""Run the frozen Qwen size-matched CoT baseline on the D009 train slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

import yaml

from hierarchical_rag.experiment import (
    REQUIRED_RUN_FILES,
    build_manifest,
    file_inventory,
    load_experiment_config,
    prepare_run_directory,
    sha256_file,
    write_json_atomic,
)
from hierarchical_rag.hotpotqa import gold_paragraphs, load_hotpotqa
from hierarchical_rag.metrics import aggregate_answer_metrics, answer_score
from hierarchical_rag.qwen_baseline import (
    build_cot_chat_prompt,
    extract_final_answer,
    has_explicit_reasoning,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the preregistered Qwen train-only CoT baseline smoke."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    config = load_experiment_config(config_path)
    _validate_frozen_protocol(config)
    root = Path.cwd().resolve()
    experiment = config["experiment"]
    dataset_config = config["dataset"]
    model_config = config["model"]
    prompt_config = config["prompt"]
    runtime = config["runtime"]

    revision = _source_revision()
    dataset_path = _resolve(root, dataset_config["source_file"])
    dataset_manifest_path = _resolve(root, dataset_config["manifest_file"])
    lock_path = _resolve(root, runtime["dependency_lock_file"])
    _verify_checksum(dataset_path, dataset_config["source_sha256"])
    _verify_checksum(dataset_manifest_path, dataset_config["manifest_sha256"])
    _verify_checksum(lock_path, runtime["dependency_lock_sha256"])

    dataset_manifest = json.loads(
        dataset_manifest_path.read_text(encoding="utf-8")
    )
    expected_ids = dataset_config["selection"]["example_ids"]
    if dataset_manifest.get("example_ids") != expected_ids:
        raise ValueError("dataset manifest example IDs differ from the config")
    examples = load_hotpotqa(dataset_path)
    if [example.identifier for example in examples] != expected_ids:
        raise ValueError("development-pool order differs from the config")
    demonstration_count = int(prompt_config["demonstration_count"])
    demonstrations = examples[:demonstration_count]
    targets = examples[demonstration_count:]
    evaluated_count = int(dataset_config["selection"]["evaluated_count"])
    if len(demonstrations) != 2 or len(targets) != evaluated_count:
        raise ValueError("D010 requires exactly 2 demonstrations and 16 targets")

    run_dir = prepare_run_directory(_resolve(root, runtime["output_dir"]))
    command = shlex.join([sys.executable, *sys.argv])
    resolved = json.loads(json.dumps(config))
    resolved["experiment"]["source_revision"] = revision
    resolved["experiment"]["actual_command"] = command
    resolved["dataset"]["source_path_resolved"] = str(dataset_path)
    resolved["dataset"]["manifest_path_resolved"] = str(dataset_manifest_path)
    resolved["runtime"]["dependency_lock_path_resolved"] = str(lock_path)
    resolved_config_path = run_dir / "resolved-config.yaml"
    resolved_config_path.write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )
    (run_dir / "command.txt").write_text(
        f"Source revision: {revision}\nCommand: {command}\n",
        encoding="utf-8",
        newline="\n",
    )
    log_lines = [
        f"experiment_id={experiment['id']}",
        f"source_revision={revision}",
        "status=running",
    ]

    try:
        metrics, environment = _run_model(
            run_dir=run_dir,
            experiment_id=experiment["id"],
            demonstrations=demonstrations,
            targets=targets,
            model_config=model_config,
            prompt_config=prompt_config,
            seed=int(experiment["seeds"][0]),
        )
        write_json_atomic(run_dir / "metrics.json", metrics)
        write_json_atomic(
            run_dir / "statistics.json",
            {
                "status": "not_applicable",
                "reason": (
                    "Train-only n=16 baseline interface smoke; no model "
                    "comparison or confirmatory claim."
                ),
            },
        )
        log_lines.extend(
            [
                f"evaluated_count={metrics['evaluated_count']}",
                f"answer_em={metrics['answer_exact_match']}",
                f"answer_f1={metrics['answer_f1']}",
                f"invalid_output_count={metrics['invalid_output_count']}",
                f"explicit_reasoning_count={metrics['explicit_reasoning_count']}",
                f"budget_exhausted_count={metrics['budget_exhausted_count']}",
                f"truncated_example_count={metrics['truncated_example_count']}",
                "status=complete",
            ]
        )
        write_json_atomic(run_dir / "environment.txt", environment)
        (run_dir / "run.log").write_text(
            "\n".join(log_lines) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        manifest = build_manifest(
            experiment_id=experiment["id"],
            owner=experiment["owner"],
            command=command,
            git_commit=revision,
            config_path=resolved_config_path,
            extra={
                "status": "complete",
                "source_config_path": str(config_path),
                "source_config_sha256": sha256_file(config_path),
                "input_files": {
                    "dataset_sha256": sha256_file(dataset_path),
                    "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
                    "dependency_lock_sha256": sha256_file(lock_path),
                },
                "dataset_manifest": dataset_manifest,
                "file_inventory": file_inventory(
                    run_dir,
                    exclude={"manifest.json"},
                ),
            },
        )
        write_json_atomic(run_dir / "manifest.json", manifest)
        _assert_required_files(run_dir)
        return 0
    except Exception as error:
        _preserve_failure(
            run_dir=run_dir,
            error=error,
            log_lines=log_lines,
            experiment=experiment,
            command=command,
            revision=revision,
            resolved_config_path=resolved_config_path,
            config_path=config_path,
        )
        raise


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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if transformers.__version__ != model_config["transformers_version"]:
        raise RuntimeError("Transformers version differs from the config")
    if torch.__version__ != model_config["torch_version"]:
        raise RuntimeError("PyTorch version differs from the config")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        dtype=torch.bfloat16,
    ).cuda().eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    if getattr(model.config, "_commit_hash", None) != model_config["revision"]:
        raise RuntimeError("resolved model revision differs from the config")
    if model.config.model_type != model_config["architecture"]:
        raise RuntimeError("loaded model architecture differs from the config")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != int(model_config["parameter_count_expected"]):
        raise RuntimeError("model parameter count differs from the config")
    if int(model.config.max_position_embeddings) != int(
        model_config["max_position_embeddings"]
    ):
        raise RuntimeError("model context length differs from the config")

    def render_chat(messages: Sequence[Mapping[str, str]]) -> str:
        return tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
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
                demonstration_rationale=prompt_config.get(
                    "demonstration_rationale",
                    "answer_only",
                ),
            )
            inputs = tokenizer(
                built.prompt,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(model.device)
            actual_input_tokens = int(inputs["input_ids"].shape[-1])
            if actual_input_tokens != built.input_tokens:
                raise RuntimeError("prompt token count changed before inference")

            torch.cuda.reset_peak_memory_stats()
            generation_started = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=int(prompt_config["max_new_tokens"]),
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            torch.cuda.synchronize()
            latency = time.perf_counter() - generation_started
            generated_ids = output[0, actual_input_tokens:]
            generated_tokens = int(generated_ids.shape[-1])
            raw_output = tokenizer.decode(
                generated_ids,
                skip_special_tokens=False,
            )
            extraction = extract_final_answer(raw_output)
            explicit_reasoning = has_explicit_reasoning(raw_output)
            prediction = extraction.answer or ""
            predictions[target.identifier] = prediction
            if extraction.status != "ok":
                invalid_count += 1
            if generated_tokens >= int(prompt_config["max_new_tokens"]):
                budget_exhausted_count += 1
            if explicit_reasoning:
                explicit_reasoning_count += 1
            if built.truncated:
                truncated_count += 1
            latencies.append(latency)
            generated_token_counts.append(generated_tokens)
            peak_allocated = max(peak_allocated, int(torch.cuda.max_memory_allocated()))
            peak_reserved = max(peak_reserved, int(torch.cuda.max_memory_reserved()))
            score = answer_score(prediction, target.answer or "")

            prediction_stream.write(
                json.dumps(
                    {
                        "example_id": target.identifier,
                        "gold_answer": target.answer,
                        "prediction": prediction,
                        "extraction": asdict(extraction),
                        "raw_output": raw_output,
                        "prompt": built.prompt,
                        "prompt_sha256": _sha256_text(built.prompt),
                        "input_tokens": actual_input_tokens,
                        "generated_tokens": generated_tokens,
                        "budget_exhausted": generated_tokens
                        >= int(prompt_config["max_new_tokens"]),
                        "explicit_reasoning_present": explicit_reasoning,
                        "latency_seconds": latency,
                        "answer_score": asdict(score),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            retrieval_stream.write(
                json.dumps(
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
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    elapsed = time.perf_counter() - started
    aggregate = aggregate_answer_metrics(predictions, targets)
    device = torch.cuda.get_device_properties(0)
    metrics = {
        "experiment_id": experiment_id,
        "status": "exploratory_train_only",
        "evaluated_count": len(targets),
        "valid_extraction_count": len(targets) - invalid_count,
        "valid_extraction_rate": (len(targets) - invalid_count) / len(targets),
        "invalid_output_count": invalid_count,
        "budget_exhausted_count": budget_exhausted_count,
        "explicit_reasoning_count": explicit_reasoning_count,
        "explicit_reasoning_rate": explicit_reasoning_count / len(targets),
        "truncated_example_count": truncated_count,
        "answer_exact_match": aggregate.exact_match,
        "answer_f1": aggregate.f1,
        "answer_precision": aggregate.precision,
        "answer_recall": aggregate.recall,
        "runtime": {
            "model_load_seconds": load_seconds,
            "evaluation_seconds": elapsed,
            "throughput_examples_per_second": len(targets) / elapsed,
            "latency_mean_seconds": fmean(latencies),
            "generated_tokens_total": sum(generated_token_counts),
            "generated_tokens_mean": fmean(generated_token_counts),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
        "claim_eligibility": "none; D010 train-only baseline interface smoke",
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
        "model_revision": getattr(model.config, "_commit_hash", None),
        "model_type": model.config.model_type,
        "model_dtype": str(model.dtype),
        "parameter_count": parameter_count,
        "max_position_embeddings": int(model.config.max_position_embeddings),
        "gpu_name": device.name,
        "gpu_total_memory_bytes": int(device.total_memory),
        "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
    }
    return metrics, environment


def _validate_frozen_protocol(config: Mapping[str, Any]) -> None:
    experiment = config["experiment"]
    dataset = config["dataset"]
    model = config["model"]
    prompt = config["prompt"]
    if experiment["stage"] != "train_only_baseline_smoke":
        raise ValueError("this runner is restricted to the train-only baseline smoke")
    if dataset["split"] != "train" or dataset["evidence"] != (
        "gold_supporting_paragraphs_in_original_context_order"
    ):
        raise ValueError("D010 requires the same train-only gold evidence as D009")
    if model["role"] != "primary_size_matched_cot_baseline":
        raise ValueError("D010 requires the preregistered primary baseline")
    if not model["frozen"] or model["dtype"] != "bfloat16":
        raise ValueError("D010 requires a frozen BF16 model")
    valid_styles = {
        "brief_explicit_cot_with_final_answer": "answer_only",
        "few_shot_supporting_fact_cot_final_answer": "supporting_fact_sentences",
    }
    if prompt["style"] not in valid_styles:
        raise ValueError("D010 requires the preregistered CoT prompt style")
    if prompt.get("demonstration_rationale", "answer_only") != valid_styles[
        prompt["style"]
    ]:
        raise ValueError("prompt style and demonstration rationale differ")
    if prompt["do_sample"] is not False:
        raise ValueError("D010 requires greedy decoding")
    if int(prompt["max_input_tokens"]) != 4032 or int(
        prompt["max_new_tokens"]
    ) != 64:
        raise ValueError("D010 requires the shared 4032+64 token budget")
    if int(prompt["max_input_tokens"]) + int(prompt["max_new_tokens"]) > int(
        model["max_position_embeddings"]
    ):
        raise ValueError("input and generation budgets exceed the model context length")


def _source_revision() -> str:
    revision = os.environ.get("HIERARCHICAL_RAG_SOURCE_REVISION", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("HIERARCHICAL_RAG_SOURCE_REVISION must be a full Git SHA")
    return revision


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


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _verify_checksum(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual.casefold() != expected.casefold():
        raise ValueError(f"checksum mismatch for {path}: {actual} != {expected}")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _assert_required_files(run_dir: Path) -> None:
    missing = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError("run is missing required files: " + ", ".join(missing))


def _preserve_failure(
    *,
    run_dir: Path,
    error: Exception,
    log_lines: list[str],
    experiment: Mapping[str, Any],
    command: str,
    revision: str,
    resolved_config_path: Path,
    config_path: Path,
) -> None:
    try:
        for name in ("predictions.jsonl", "retrieval.jsonl"):
            (run_dir / name).touch(exist_ok=True)
        if not (run_dir / "metrics.json").exists():
            write_json_atomic(
                run_dir / "metrics.json",
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                },
            )
        if not (run_dir / "statistics.json").exists():
            write_json_atomic(
                run_dir / "statistics.json",
                {"status": "not_computed", "reason": "run failed"},
            )
        if not (run_dir / "environment.txt").exists():
            write_json_atomic(
                run_dir / "environment.txt",
                {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "traceback": traceback.format_exc(),
                },
            )
        log_lines.extend(
            [f"error={type(error).__name__}: {error}", "status=failed"]
        )
        (run_dir / "run.log").write_text(
            "\n".join(log_lines) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        manifest = build_manifest(
            experiment_id=experiment["id"],
            owner=experiment["owner"],
            command=command,
            git_commit=revision,
            config_path=resolved_config_path,
            extra={
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "source_config_path": str(config_path),
                "source_config_sha256": sha256_file(config_path),
                "file_inventory": file_inventory(
                    run_dir,
                    exclude={"manifest.json"},
                ),
            },
        )
        write_json_atomic(run_dir / "manifest.json", manifest)
    except Exception:
        return


if __name__ == "__main__":
    raise SystemExit(main())
