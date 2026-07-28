"""Calibrate the frozen D028 dense encoder without opening benchmark labels."""

from __future__ import annotations

import argparse
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
from typing import Any, Mapping, Sequence

import yaml

from hierarchical_rag.dense_retrieval import (
    project_compute,
    projected_embedding_bytes,
    validate_dense_calibration_protocol,
)
from hierarchical_rag.experiment import (
    REQUIRED_RUN_FILES,
    build_manifest,
    file_inventory,
    load_experiment_config,
    prepare_run_directory,
    sha256_file,
    write_json_atomic,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_calibration(args.config.resolve())


def run_calibration(config_path: Path) -> int:
    config = load_experiment_config(config_path)
    validate_dense_calibration_protocol(config)
    root = Path.cwd().resolve()
    experiment = config["experiment"]
    dataset = config["dataset"]
    runtime = config["runtime"]
    revision = _source_revision()

    sample_path = _resolve(root, dataset["source_file"])
    sample_manifest_path = _resolve(root, dataset["manifest_file"])
    lock_path = _resolve(root, runtime["dependency_lock_file"])
    _verify_checksum(sample_path, dataset["source_sha256"])
    _verify_checksum(sample_manifest_path, dataset["manifest_sha256"])
    _verify_checksum(lock_path, runtime["dependency_lock_sha256"])
    sample_manifest = json.loads(sample_manifest_path.read_text(encoding="utf-8"))
    _validate_sample_manifest(sample_manifest, dataset)

    run_dir = prepare_run_directory(_resolve(root, runtime["output_dir"]))
    command = shlex.join([sys.executable, *sys.argv])
    resolved = json.loads(json.dumps(config))
    resolved["experiment"]["source_revision"] = revision
    resolved["experiment"]["actual_command"] = command
    resolved["dataset"]["source_path_resolved"] = str(sample_path)
    resolved["dataset"]["manifest_path_resolved"] = str(sample_manifest_path)
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
    (run_dir / "retrieval.jsonl").write_text(
        json.dumps(
            {
                "status": "not_applicable",
                "reason": "corpus-side throughput calibration performs no search",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    log_lines = [
        f"experiment_id={experiment['id']}",
        f"source_revision={revision}",
        "status=running",
    ]

    try:
        documents = _load_sample(sample_path, int(dataset["selection"]["size"]))
        metrics, statistics, environment = _measure(
            documents=documents,
            retriever=config["retriever"],
            selection=dataset["selection"],
            runtime=runtime,
            predictions_path=run_dir / "predictions.jsonl",
        )
        metrics["experiment_id"] = experiment["id"]
        write_json_atomic(run_dir / "metrics.json", metrics)
        write_json_atomic(run_dir / "statistics.json", statistics)
        write_json_atomic(run_dir / "environment.txt", environment)
        log_lines.extend(
            (
                f"measured_documents={metrics['measured_documents']}",
                f"documents_per_second={metrics['documents_per_second']}",
                f"projected_full_units={metrics['projection']['projected_units']}",
                "status=complete",
            )
        )
        (run_dir / "run.log").write_text(
            "\n".join(log_lines) + "\n", encoding="utf-8", newline="\n"
        )
        _write_manifest(
            run_dir=run_dir,
            experiment=experiment,
            command=command,
            revision=revision,
            resolved_config_path=resolved_config_path,
            config_path=config_path,
            sample_path=sample_path,
            sample_manifest_path=sample_manifest_path,
            lock_path=lock_path,
            sample_manifest=sample_manifest,
            status="complete",
        )
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
            sample_path=sample_path,
            sample_manifest_path=sample_manifest_path,
            lock_path=lock_path,
            sample_manifest=sample_manifest,
        )
        raise


def _measure(
    *,
    documents: Sequence[Mapping[str, Any]],
    retriever: Mapping[str, Any],
    selection: Mapping[str, Any],
    runtime: Mapping[str, Any],
    predictions_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    import torch
    import torch.nn.functional as functional
    import transformers
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if torch.__version__ != retriever["torch_version"]:
        raise RuntimeError("PyTorch version differs from the config")
    if transformers.__version__ != retriever["transformers_version"]:
        raise RuntimeError("Transformers version differs from the config")
    device = torch.cuda.get_device_properties(0)
    if runtime["expected_gpu_name_contains"] not in device.name:
        raise RuntimeError(
            f"expected {runtime['expected_gpu_name_contains']} GPU, found {device.name}"
        )

    load_started = time.perf_counter()
    print(f"stage=tokenizer_load_start model={retriever['id']}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        retriever["id"],
        revision=retriever["revision"],
        padding_side="left",
    )
    print(f"stage=model_load_start model={retriever['id']}", flush=True)
    model = AutoModel.from_pretrained(
        retriever["id"],
        revision=retriever["revision"],
        dtype=torch.bfloat16,
        attn_implementation=retriever["attention_implementation"],
    ).cuda().eval()
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started
    resolved_revision = getattr(model.config, "_commit_hash", None)
    if resolved_revision != retriever["revision"]:
        raise RuntimeError("resolved model revision differs from the config")
    if model.config.model_type != retriever["architecture"]:
        raise RuntimeError("loaded model architecture differs from the config")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != int(retriever["parameter_count_expected"]):
        raise RuntimeError(
            "model parameter count differs from the config: "
            f"observed={parameter_count} expected={retriever['parameter_count_expected']}"
        )

    max_tokens = int(retriever["max_input_tokens"])
    dimension = int(retriever["output_dimension"])
    original_lengths = [
        len(tokenizer(row["text"], add_special_tokens=True)["input_ids"])
        for row in documents
    ]
    warmup_count = int(selection["warmup_documents"])
    measured_count = int(selection["measured_documents"])
    batch_size = int(runtime["batch_size"])
    warmup = documents[:warmup_count]
    measured = documents[warmup_count : warmup_count + measured_count]
    measured_lengths = original_lengths[warmup_count : warmup_count + measured_count]

    print(
        f"stage=warmup_start documents={len(warmup)} batch_size={batch_size}",
        flush=True,
    )
    _encode_batches(
        rows=warmup,
        tokenizer=tokenizer,
        model=model,
        torch_module=torch,
        functional=functional,
        max_tokens=max_tokens,
        dimension=dimension,
        batch_size=batch_size,
        progress=False,
    )
    torch.cuda.synchronize()
    print("stage=warmup_complete", flush=True)

    torch.cuda.reset_peak_memory_stats()
    measured_started = time.perf_counter()
    batch_records, token_count = _encode_batches(
        rows=measured,
        tokenizer=tokenizer,
        model=model,
        torch_module=torch,
        functional=functional,
        max_tokens=max_tokens,
        dimension=dimension,
        batch_size=batch_size,
        progress=True,
    )
    torch.cuda.synchronize()
    measured_seconds = time.perf_counter() - measured_started
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())

    with predictions_path.open("x", encoding="utf-8", newline="\n") as stream:
        for record in batch_records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    unit_rate = int(runtime["datasphere_units_per_second"])
    projection = project_compute(
        measured_documents=measured_count,
        measured_seconds=measured_seconds,
        full_corpus_documents=int(selection["full_corpus_documents"]),
        units_per_second=max(unit_rate, 1),
        reserve_multiplier=float(runtime["projection_reserve_multiplier"]),
        external_throughput_cap=float(
            runtime["corpus_stream_documents_per_second"]
        ),
    )
    truncated_count = sum(length > max_tokens for length in measured_lengths)
    embedding_bytes = projected_embedding_bytes(
        int(selection["full_corpus_documents"]), dimension
    )
    projection_record = asdict(projection)
    if unit_rate == 0:
        projection_record["projected_units"] = None
    metrics = {
        "status": "complete_non_claim_bearing_resource_calibration",
        "labels_observed": False,
        "warmup_documents": warmup_count,
        "measured_documents": measured_count,
        "batch_size": batch_size,
        "measured_seconds": measured_seconds,
        "documents_per_second": measured_count / measured_seconds,
        "tokens_processed": token_count,
        "tokens_per_second": token_count / measured_seconds,
        "truncated_document_count": truncated_count,
        "truncated_document_rate": truncated_count / measured_count,
        "input_tokens_mean_before_truncation": sum(measured_lengths) / measured_count,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "projected_embedding_bytes_fp16": embedding_bytes,
        "projection": projection_record,
        "authorization": (
            "calibration only; compare projected wall time with the D033 local gate "
            "before a separate full-corpus build"
            if unit_rate == 0
            else "calibration only; compare projected_units with current project "
            "balance before a separate full-corpus build"
        ),
    }
    statistics = {
        "status": "not_applicable_resource_measurement",
        "quality_metrics_observed": False,
        "projection_method": "measured systematic corpus sample times fixed reserve",
        "corpus_stream_throughput_cap": float(
            runtime["corpus_stream_documents_per_second"]
        ),
        "reserve_multiplier": projection.reserve_multiplier,
    }
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "pip_freeze": _pip_freeze(),
        "nvidia_smi": _nvidia_smi(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "model_id": retriever["id"],
        "model_revision": resolved_revision,
        "model_type": model.config.model_type,
        "model_dtype": str(model.dtype),
        "parameter_count": parameter_count,
        "model_load_seconds": model_load_seconds,
        "gpu_name": device.name,
        "gpu_total_memory_bytes": int(device.total_memory),
        "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
    }
    return metrics, statistics, environment


def _encode_batches(
    *,
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    model: Any,
    torch_module: Any,
    functional: Any,
    max_tokens: int,
    dimension: int,
    batch_size: int,
    progress: bool,
) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    token_count = 0
    total = len(rows)
    for start in range(0, total, batch_size):
        batch = rows[start : start + batch_size]
        inputs = tokenizer(
            [row["text"] for row in batch],
            padding=True,
            truncation=True,
            max_length=max_tokens,
            return_tensors="pt",
        ).to(model.device)
        with torch_module.inference_mode():
            outputs = model(**inputs)
            pooled = _last_token_pool(
                outputs.last_hidden_state,
                inputs["attention_mask"],
                torch_module,
            )
            embeddings = functional.normalize(
                pooled[:, :dimension].float(), p=2, dim=1
            ).to(torch_module.float16)
        if tuple(embeddings.shape) != (len(batch), dimension):
            raise RuntimeError("dense encoder returned an unexpected shape")
        norms = embeddings.float().norm(p=2, dim=1)
        max_norm_error = float((norms - 1.0).abs().max().item())
        if max_norm_error > 0.002:
            raise RuntimeError("stored dense embeddings are not L2-normalized")
        batch_tokens = int(inputs["attention_mask"].sum().item())
        token_count += batch_tokens
        completed = min(start + len(batch), total)
        records.append(
            {
                "batch_index": len(records),
                "first_rowid": int(batch[0]["rowid"]),
                "last_rowid": int(batch[-1]["rowid"]),
                "document_count": len(batch),
                "processed_tokens": batch_tokens,
                "embedding_dimension": dimension,
                "max_l2_norm_error_after_fp16_cast": max_norm_error,
            }
        )
        if progress:
            print(
                _progress(completed=completed, total=total, tokens=token_count),
                flush=True,
            )
    return records, token_count


def _last_token_pool(
    hidden_states: Any, attention_mask: Any, torch_module: Any
) -> Any:
    left_padding = bool(attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return hidden_states[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    indices = torch_module.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[indices, lengths]


def _progress(*, completed: int, total: int, tokens: int, width: int = 24) -> str:
    filled = min(width, width * completed // total)
    return (
        f"progress=[{'#' * filled}{'-' * (width - filled)}] "
        f"documents={completed}/{total} processed_tokens={tokens}"
    )


def _load_sample(path: Path, expected_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise ValueError(f"{path}:{line_number}: invalid sample row")
            rows.append(row)
    if len(rows) != expected_count:
        raise ValueError(f"sample count mismatch: {len(rows)} != {expected_count}")
    if [int(row["rowid"]) for row in rows] != sorted(
        int(row["rowid"]) for row in rows
    ):
        raise ValueError("calibration sample rowids are not sorted")
    return rows


def _validate_sample_manifest(
    manifest: Mapping[str, Any], dataset: Mapping[str, Any]
) -> None:
    if manifest.get("status") != "complete":
        raise ValueError("calibration sample manifest is incomplete")
    if manifest.get("selection") != dataset["selection"]["method"]:
        raise ValueError("calibration sample selection differs from config")
    if int(manifest.get("sample_size", -1)) != int(dataset["selection"]["size"]):
        raise ValueError("calibration sample size differs from config")
    if manifest.get("sample_sha256") != dataset["source_sha256"]:
        raise ValueError("calibration sample hash differs from config")
    if manifest.get("corpus_sha256") != dataset["corpus_sha256"]:
        raise ValueError("full corpus hash differs from config")


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
    sample_path: Path,
    sample_manifest_path: Path,
    lock_path: Path,
    sample_manifest: Mapping[str, Any],
) -> None:
    for name in ("predictions.jsonl", "retrieval.jsonl"):
        path = run_dir / name
        if not path.exists():
            path.write_text("", encoding="utf-8")
    write_json_atomic(
        run_dir / "metrics.json",
        {"experiment_id": experiment["id"], "status": "failed"},
    )
    write_json_atomic(
        run_dir / "statistics.json",
        {"status": "not_applicable_failed_calibration"},
    )
    if not (run_dir / "environment.txt").exists():
        write_json_atomic(
            run_dir / "environment.txt",
            {"python": sys.version, "platform": platform.platform()},
        )
    log_lines.extend((f"error={type(error).__name__}: {error}", traceback.format_exc(), "status=failed"))
    (run_dir / "run.log").write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8", newline="\n"
    )
    _write_manifest(
        run_dir=run_dir,
        experiment=experiment,
        command=command,
        revision=revision,
        resolved_config_path=resolved_config_path,
        config_path=config_path,
        sample_path=sample_path,
        sample_manifest_path=sample_manifest_path,
        lock_path=lock_path,
        sample_manifest=sample_manifest,
        status="failed",
    )


def _write_manifest(
    *,
    run_dir: Path,
    experiment: Mapping[str, Any],
    command: str,
    revision: str,
    resolved_config_path: Path,
    config_path: Path,
    sample_path: Path,
    sample_manifest_path: Path,
    lock_path: Path,
    sample_manifest: Mapping[str, Any],
    status: str,
) -> None:
    manifest = build_manifest(
        experiment_id=experiment["id"],
        owner=experiment["owner"],
        command=command,
        git_commit=revision,
        config_path=resolved_config_path,
        extra={
            "status": status,
            "source_config_path": str(config_path),
            "source_config_sha256": sha256_file(config_path),
            "input_files": {
                "sample_sha256": sha256_file(sample_path),
                "sample_manifest_sha256": sha256_file(sample_manifest_path),
                "dependency_lock_sha256": sha256_file(lock_path),
            },
            "sample_manifest": dict(sample_manifest),
            "file_inventory": file_inventory(run_dir, exclude={"manifest.json"}),
        },
    )
    write_json_atomic(run_dir / "manifest.json", manifest)


def _source_revision() -> str:
    revision = os.environ.get("HIERARCHICAL_RAG_SOURCE_REVISION", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("HIERARCHICAL_RAG_SOURCE_REVISION must be a full Git SHA")
    return revision


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _verify_checksum(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual.casefold() != expected.casefold():
        raise ValueError(f"checksum mismatch for {path}: {actual} != {expected}")


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


def _assert_required_files(run_dir: Path) -> None:
    missing = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError("run is missing required files: " + ", ".join(missing))


if __name__ == "__main__":
    raise SystemExit(main())
