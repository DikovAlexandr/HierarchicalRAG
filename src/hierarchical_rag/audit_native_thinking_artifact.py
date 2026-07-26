"""Recover verifiable metrics from a post-generation native-thinking failure."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from hierarchical_rag.experiment import (
    load_experiment_config,
    sha256_file,
    write_json_atomic,
)
from hierarchical_rag.hotpotqa import HotpotExample, load_hotpotqa
from hierarchical_rag.metrics import aggregate_answer_metrics, answer_score
from hierarchical_rag.native_thinking import (
    detect_native_reasoning,
    validate_native_thinking_protocol,
)
from hierarchical_rag.qwen_baseline import extract_final_answer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit immutable raw outputs after the known D017 post-generation "
            "metrics-finalization failure; never rerun the model."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--terminal-log", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_artifact(
        run_dir=args.run_dir.resolve(),
        terminal_log=args.terminal_log.resolve(),
        source_config=args.source_config.resolve(),
        archive=args.archive.resolve() if args.archive else None,
    )
    if args.output:
        write_json_atomic(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


def audit_artifact(
    *,
    run_dir: Path,
    terminal_log: Path,
    source_config: Path,
    archive: Path | None,
) -> dict[str, Any]:
    manifest = _load_json(run_dir / "manifest.json")
    failed_metrics = _load_json(run_dir / "metrics.json")
    resolved_config_path = run_dir / "resolved-config.yaml"
    config = load_experiment_config(resolved_config_path)
    validate_native_thinking_protocol(config)

    expected_error = "NameError: name 'experiment_config' is not defined"
    if failed_metrics != {"error": expected_error, "status": "failed"}:
        raise ValueError("run does not contain the known finalization failure")
    if manifest.get("extra", {}).get("error") != expected_error:
        raise ValueError("manifest error differs from the known finalization failure")
    if "D017" not in config["experiment"]["decision_ids"]:
        raise ValueError("artifact recovery is restricted to the D017 final gate")
    if manifest["experiment_id"] != config["experiment"]["id"]:
        raise ValueError("manifest experiment ID differs from the resolved config")
    if manifest["git_commit"] != config["experiment"]["source_revision"]:
        raise ValueError("source revisions differ between manifest and config")
    if manifest["resolved_config_sha256"] != sha256_file(resolved_config_path):
        raise ValueError("resolved config checksum differs from the manifest")
    _verify_inventory(run_dir, manifest["extra"]["file_inventory"])

    if sha256_file(source_config) != manifest["extra"]["source_config_sha256"]:
        raise ValueError("source config checksum differs from the manifest")
    for section, path_field, hash_field in (
        ("runtime", "dependency_lock_file", "dependency_lock_sha256"),
        ("dataset", "source_file", "source_sha256"),
        ("dataset", "manifest_file", "manifest_sha256"),
    ):
        path = Path(config[section][path_field])
        if sha256_file(path) != config[section][hash_field]:
            raise ValueError(f"checksum mismatch for {path}")

    predictions = _load_jsonl(run_dir / "predictions.jsonl")
    retrieval = _load_jsonl(run_dir / "retrieval.jsonl")
    expected_ids = list(config["dataset"]["selection"]["example_ids"])[2:]
    examples = load_hotpotqa(config["dataset"]["source_file"])[2:]
    if [example.identifier for example in examples] != expected_ids:
        raise ValueError("dataset target order differs from the resolved config")
    if [row["example_id"] for row in predictions] != expected_ids:
        raise ValueError("prediction target order differs from the resolved config")
    if [row["example_id"] for row in retrieval] != expected_ids:
        raise ValueError("retrieval target order differs from the resolved config")
    if len(predictions) != int(config["dataset"]["selection"]["evaluated_count"]):
        raise ValueError("prediction count differs from the resolved config")

    recovered = summarize_records(
        config=config,
        predictions=predictions,
        retrieval=retrieval,
        examples=examples,
    )
    terminal_text = terminal_log.read_text(encoding="utf-8")
    completed_ids = re.findall(
        r"^progress=.* examples=\d+/\d+ example_id=(\S+) status=complete ",
        terminal_text,
        flags=re.MULTILINE,
    )
    if completed_ids != expected_ids:
        raise ValueError("terminal completion order differs from the resolved config")
    environment_match = re.search(
        r"^environment_ok torch=(\S+) transformers=(\S+) gpu=(.+)$",
        terminal_text,
        flags=re.MULTILINE,
    )
    model_match = re.search(
        r"^stage=model_ready model=(\S+) load_seconds=([0-9.]+) "
        r"parameters=(\d+)$",
        terminal_text,
        flags=re.MULTILINE,
    )
    if environment_match is None or model_match is None:
        raise ValueError("terminal log is missing the pinned environment/model record")
    if (
        environment_match.group(1) != config["model"]["torch_version"]
        or environment_match.group(2) != config["model"]["transformers_version"]
        or model_match.group(1) != config["model"]["id"]
        or int(model_match.group(3)) != int(config["model"]["parameter_count_expected"])
    ):
        raise ValueError("terminal environment/model record differs from the config")

    provenance: dict[str, Any] = {
        "original_source_revision": manifest["git_commit"],
        "audit_source_revision": _git_revision(),
        "raw_run_dir": _display_path(run_dir),
        "source_config": _display_path(source_config),
        "source_config_sha256": sha256_file(source_config),
        "resolved_config_sha256": sha256_file(resolved_config_path),
        "manifest_sha256": sha256_file(run_dir / "manifest.json"),
        "predictions_sha256": sha256_file(run_dir / "predictions.jsonl"),
        "retrieval_sha256": sha256_file(run_dir / "retrieval.jsonl"),
        "terminal_log_sha256": sha256_file(terminal_log),
    }
    if archive is not None:
        provenance["archive"] = _display_path(archive)
        provenance["archive_sha256"] = sha256_file(archive)

    return {
        "schema_version": 1,
        "experiment_id": config["experiment"]["id"],
        "audit": {
            "integrity_status": "passed",
            "recovery_scope": "deterministic_post_generation_metrics_only",
            "original_status": "failed_during_metrics_finalization",
            "original_error": expected_error,
            "unrecoverable_original_fields": [
                "evaluation_seconds_including_non_generation_overhead",
                "original_throughput_examples_per_second",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
                "full_pip_freeze_from_generation_process",
                "full_nvidia_smi_from_generation_process",
            ],
        },
        "provenance": provenance,
        "environment_from_terminal": {
            "torch": environment_match.group(1),
            "transformers": environment_match.group(2),
            "gpu": environment_match.group(3),
            "model": model_match.group(1),
            "parameters": int(model_match.group(3)),
            "model_load_seconds": float(model_match.group(2)),
        },
        "recovered_metrics": recovered,
        "claim_scope": (
            "Exploratory train-only D017 interface evidence only; no H1, "
            "validation, system-comparison, or article-result claim."
        ),
    }


def summarize_records(
    *,
    config: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    retrieval: Sequence[Mapping[str, Any]],
    examples: Sequence[HotpotExample],
) -> dict[str, Any]:
    if not predictions or len(predictions) != len(retrieval) or len(predictions) != len(examples):
        raise ValueError("predictions, retrieval, and examples must have equal nonzero size")
    prediction_map: dict[str, str] = {}
    reasoning_methods: Counter[str] = Counter()
    max_new_tokens = int(config["prompt"]["max_new_tokens"])

    for prediction, retrieved, example in zip(
        predictions, retrieval, examples, strict=True
    ):
        if not (
            prediction["example_id"]
            == retrieved["example_id"]
            == example.identifier
        ):
            raise ValueError("record identifiers differ")
        prompt_sha256 = hashlib.sha256(
            prediction["prompt"].encode("utf-8")
        ).hexdigest()
        if prompt_sha256 != prediction["prompt_sha256"]:
            raise ValueError(f"{example.identifier}: prompt checksum mismatch")
        extraction = extract_final_answer(prediction["raw_output"])
        reasoning = detect_native_reasoning(prediction["raw_output"])
        score = answer_score(extraction.answer or "", example.answer or "")
        if asdict(extraction) != prediction["extraction"]:
            raise ValueError(f"{example.identifier}: extraction differs")
        if asdict(reasoning) != prediction["reasoning_detection"]:
            raise ValueError(f"{example.identifier}: reasoning detection differs")
        if asdict(score) != prediction["answer_score"]:
            raise ValueError(f"{example.identifier}: answer score differs")
        if prediction["prediction"] != (extraction.answer or ""):
            raise ValueError(f"{example.identifier}: prediction differs")
        exhausted = int(prediction["generated_tokens"]) >= max_new_tokens
        if bool(prediction["budget_exhausted"]) != exhausted:
            raise ValueError(f"{example.identifier}: exhaustion flag differs")
        truncated = bool(
            int(retrieved["dropped_document_count"])
            or int(retrieved["dropped_sentence_count"])
        )
        if bool(retrieved["truncated"]) != truncated:
            raise ValueError(f"{example.identifier}: truncation flag differs")
        prediction_map[example.identifier] = prediction["prediction"]
        reasoning_methods[reasoning.method] += 1

    aggregate = aggregate_answer_metrics(prediction_map, examples)
    valid_count = sum(row["extraction"]["status"] == "ok" for row in predictions)
    reasoning_count = sum(
        bool(row["reasoning_detection"]["present"]) for row in predictions
    )
    exhausted_ids = [
        row["example_id"] for row in predictions if row["budget_exhausted"]
    ]
    truncated_count = sum(bool(row["truncated"]) for row in retrieval)
    latencies = [float(row["latency_seconds"]) for row in predictions]
    generated_tokens = [int(row["generated_tokens"]) for row in predictions]
    count = len(predictions)
    return {
        "interface_gate_passed": all(
            (
                valid_count == count,
                reasoning_count == count,
                not exhausted_ids,
                truncated_count == 0,
            )
        ),
        "evaluated_count": count,
        "valid_extraction_count": valid_count,
        "explicit_reasoning_count": reasoning_count,
        "reasoning_detection_methods": dict(sorted(reasoning_methods.items())),
        "budget_exhausted_count": len(exhausted_ids),
        "budget_exhausted_ids": exhausted_ids,
        "truncated_example_count": truncated_count,
        "answer_exact_match": aggregate.exact_match,
        "answer_f1": aggregate.f1,
        "answer_precision": aggregate.precision,
        "answer_recall": aggregate.recall,
        "generated_tokens_total": sum(generated_tokens),
        "generated_tokens_mean": fmean(generated_tokens),
        "generation_latency_sum_seconds": sum(latencies),
        "generation_latency_mean_seconds": fmean(latencies),
        "generation_only_throughput_examples_per_second": count / sum(latencies),
        "input_tokens_min": min(int(row["input_tokens"]) for row in predictions),
        "input_tokens_max": max(int(row["input_tokens"]) for row in predictions),
    }


def _verify_inventory(run_dir: Path, inventory: Sequence[Mapping[str, Any]]) -> None:
    for item in inventory:
        path = run_dir / str(item["path"])
        if path.stat().st_size != int(item["size_bytes"]):
            raise ValueError(f"size mismatch for {path}")
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"checksum mismatch for {path}")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
