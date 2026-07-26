"""Independently audit the non-claim-bearing D028 dense calibration."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from hierarchical_rag.dense_retrieval import (
    project_compute,
    projected_embedding_bytes,
    validate_dense_calibration_protocol,
)
from hierarchical_rag.experiment import (
    load_experiment_config,
    sha256_file,
    write_json_atomic,
)


FULL_ENCODING_UNIT_LIMIT = 180_000
REPORTED_REMAINING_BALANCE = 1_455_036


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_dense_calibration(
        run_dir=args.run_dir.resolve(),
        source_config=args.source_config.resolve(),
        archive=args.archive.resolve(),
    )
    if args.output:
        write_json_atomic(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


def audit_dense_calibration(
    *, run_dir: Path, source_config: Path, archive: Path
) -> dict[str, Any]:
    root = source_config.parents[2]
    manifest = _load_json(run_dir / "manifest.json")
    metrics = _load_json(run_dir / "metrics.json")
    statistics = _load_json(run_dir / "statistics.json")
    environment = _load_json(run_dir / "environment.txt")
    resolved_path = run_dir / "resolved-config.yaml"
    config = load_experiment_config(resolved_path)
    validate_dense_calibration_protocol(config)

    if manifest.get("extra", {}).get("status") != "complete":
        raise ValueError("dense calibration manifest is not complete")
    if metrics.get("status") != "complete_non_claim_bearing_resource_calibration":
        raise ValueError("dense calibration metrics are not complete")
    if manifest["experiment_id"] != config["experiment"]["id"]:
        raise ValueError("manifest experiment ID differs from resolved config")
    if manifest["git_commit"] != config["experiment"]["source_revision"]:
        raise ValueError("manifest revision differs from resolved config")
    if manifest["resolved_config_sha256"] != sha256_file(resolved_path):
        raise ValueError("resolved config checksum differs from manifest")
    if sha256_file(source_config) != manifest["extra"]["source_config_sha256"]:
        raise ValueError("source config checksum differs from manifest")
    _verify_inventory(run_dir, manifest["extra"]["file_inventory"])

    checked_inputs: dict[str, str] = {}
    for section, path_key, checksum_key in (
        ("dataset", "source_file", "source_sha256"),
        ("dataset", "manifest_file", "manifest_sha256"),
        ("runtime", "dependency_lock_file", "dependency_lock_sha256"),
    ):
        path = _resolve(root, config[section][path_key])
        checksum = sha256_file(path)
        if checksum != config[section][checksum_key]:
            raise ValueError(f"input checksum differs: {path_key}")
        checked_inputs[path_key] = checksum

    sample = _load_jsonl(_resolve(root, config["dataset"]["source_file"]))
    sample_manifest = _load_json(
        _resolve(root, config["dataset"]["manifest_file"])
    )
    selection = config["dataset"]["selection"]
    if sample_manifest != manifest["extra"]["sample_manifest"]:
        raise ValueError("sample manifest differs from run manifest")
    if len(sample) != int(selection["size"]):
        raise ValueError("calibration sample size differs from config")
    if any(
        int(right["rowid"]) <= int(left["rowid"])
        for left, right in zip(sample, sample[1:])
    ):
        raise ValueError("calibration sample rowids are not strictly increasing")

    records = _load_jsonl(run_dir / "predictions.jsonl")
    measured = sample[
        int(selection["warmup_documents"]) : int(selection["size"])
    ]
    record_summary = summarize_calibration_records(
        records=records,
        measured_rows=measured,
        batch_size=int(config["runtime"]["batch_size"]),
        dimension=int(config["retriever"]["output_dimension"]),
    )

    measured_seconds = float(metrics["measured_seconds"])
    projection = project_compute(
        measured_documents=record_summary["measured_documents"],
        measured_seconds=measured_seconds,
        full_corpus_documents=int(selection["full_corpus_documents"]),
        units_per_second=int(config["runtime"]["datasphere_units_per_second"]),
        reserve_multiplier=float(config["runtime"]["projection_reserve_multiplier"]),
        external_throughput_cap=float(
            config["runtime"]["corpus_stream_documents_per_second"]
        ),
    )
    verified = {
        "measured_documents": record_summary["measured_documents"],
        "batch_count": record_summary["batch_count"],
        "tokens_processed": record_summary["tokens_processed"],
        "documents_per_second": projection.documents_per_second,
        "tokens_per_second": record_summary["tokens_processed"] / measured_seconds,
        "max_l2_norm_error_after_fp16_cast": record_summary[
            "max_l2_norm_error_after_fp16_cast"
        ],
        "projected_embedding_bytes_fp16": projected_embedding_bytes(
            int(selection["full_corpus_documents"]),
            int(config["retriever"]["output_dimension"]),
        ),
        "projection": {
            "effective_documents_per_second": projection.effective_documents_per_second,
            "projected_seconds": projection.projected_seconds,
            "projected_units": projection.projected_units,
            "reserve_multiplier": projection.reserve_multiplier,
        },
    }
    _verify_reported_metrics(metrics, verified, config)
    _verify_statistics(statistics, config)
    _verify_environment(environment, config)

    projected_units = projection.projected_units
    policy = {
        "decision": "do_not_start_full_dense_corpus_build",
        "d029_encoding_unit_limit": FULL_ENCODING_UNIT_LIMIT,
        "reported_remaining_balance_before_calibration": REPORTED_REMAINING_BALANCE,
        "projected_units": projected_units,
        "exceeds_encoding_limit_by_units": projected_units
        - FULL_ENCODING_UNIT_LIMIT,
        "encoding_limit_multiple": projected_units / FULL_ENCODING_UNIT_LIMIT,
        "exceeds_reported_balance_by_units": projected_units
        - REPORTED_REMAINING_BALANCE,
        "reported_balance_multiple": projected_units / REPORTED_REMAINING_BALANCE,
        "projected_a100_hours_with_reserve": projection.projected_seconds / 3600,
    }

    return {
        "schema_version": 1,
        "experiment_id": config["experiment"]["id"],
        "audit": {
            "integrity_status": "passed_for_resource_projection",
            "original_status": metrics["status"],
            "audit_source_revision": _git_revision(root),
            "limitations": [
                "Per-document pre-truncation token lengths were not retained, so the "
                "reported mean input length and zero-truncation count cannot be "
                "recomputed from the artifact alone. They remain reproducible from "
                "the pinned sample and tokenizer, but are not needed for the D029 "
                "cost decision."
            ],
        },
        "provenance": {
            "archive_sha256": sha256_file(archive),
            "original_source_revision": manifest["git_commit"],
            "manifest_sha256": sha256_file(run_dir / "manifest.json"),
            "source_config_sha256": sha256_file(source_config),
            "resolved_config_sha256": sha256_file(resolved_path),
            "checked_input_sha256": checked_inputs,
            "model_id": environment["model_id"],
            "model_revision": environment["model_revision"],
            "hardware": environment["gpu_name"],
        },
        "verified_metrics": verified,
        "reported_non_recomputed_metrics": {
            "input_tokens_mean_before_truncation": metrics[
                "input_tokens_mean_before_truncation"
            ],
            "truncated_document_count": metrics["truncated_document_count"],
            "truncated_document_rate": metrics["truncated_document_rate"],
            "peak_allocated_bytes": metrics["peak_allocated_bytes"],
            "peak_reserved_bytes": metrics["peak_reserved_bytes"],
        },
        "resource_policy": policy,
        "claim_scope": (
            "Label-free resource calibration only. It supports a resource-bound "
            "decision, not a dense-retrieval quality or reader-quality claim."
        ),
    }


def summarize_calibration_records(
    *,
    records: Sequence[Mapping[str, Any]],
    measured_rows: Sequence[Mapping[str, Any]],
    batch_size: int,
    dimension: int,
) -> dict[str, Any]:
    if not records or not measured_rows:
        raise ValueError("calibration records and measured rows cannot be empty")
    expected_batches = math.ceil(len(measured_rows) / batch_size)
    if len(records) != expected_batches:
        raise ValueError("calibration batch count differs from measured sample")

    tokens = 0
    cursor = 0
    max_norm_error = 0.0
    for batch_index, record in enumerate(records):
        expected_count = min(batch_size, len(measured_rows) - cursor)
        batch = measured_rows[cursor : cursor + expected_count]
        expected = {
            "batch_index": batch_index,
            "document_count": expected_count,
            "embedding_dimension": dimension,
            "first_rowid": int(batch[0]["rowid"]),
            "last_rowid": int(batch[-1]["rowid"]),
        }
        for field, value in expected.items():
            if record.get(field) != value:
                raise ValueError(f"calibration batch field differs: {field}")
        processed_tokens = int(record.get("processed_tokens", 0))
        if processed_tokens < expected_count:
            raise ValueError("calibration batch token count is invalid")
        norm_error = float(record.get("max_l2_norm_error_after_fp16_cast", -1))
        if not 0 <= norm_error <= 0.002:
            raise ValueError("calibration embedding norm error is invalid")
        tokens += processed_tokens
        max_norm_error = max(max_norm_error, norm_error)
        cursor += expected_count

    if cursor != len(measured_rows):
        raise ValueError("calibration records do not cover measured sample")
    return {
        "batch_count": len(records),
        "measured_documents": cursor,
        "tokens_processed": tokens,
        "max_l2_norm_error_after_fp16_cast": max_norm_error,
    }


def _verify_reported_metrics(
    metrics: Mapping[str, Any], verified: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    exact = {
        "labels_observed": False,
        "warmup_documents": int(
            config["dataset"]["selection"]["warmup_documents"]
        ),
        "measured_documents": verified["measured_documents"],
        "batch_size": int(config["runtime"]["batch_size"]),
        "tokens_processed": verified["tokens_processed"],
        "projected_embedding_bytes_fp16": verified[
            "projected_embedding_bytes_fp16"
        ],
    }
    for field, expected in exact.items():
        if metrics.get(field) != expected:
            raise ValueError(f"reported dense metric differs: {field}")
    for field in ("documents_per_second", "tokens_per_second"):
        if not math.isclose(float(metrics[field]), float(verified[field]), rel_tol=1e-12):
            raise ValueError(f"reported dense metric differs: {field}")
    for field in (
        "effective_documents_per_second",
        "projected_seconds",
        "projected_units",
        "reserve_multiplier",
    ):
        reported = metrics["projection"][field]
        expected = verified["projection"][field]
        if isinstance(expected, float):
            matches = math.isclose(float(reported), expected, rel_tol=1e-12)
        else:
            matches = reported == expected
        if not matches:
            raise ValueError(f"reported dense projection differs: {field}")


def _verify_statistics(
    statistics: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    expected = {
        "status": "not_applicable_resource_measurement",
        "quality_metrics_observed": False,
        "projection_method": "measured systematic corpus sample times fixed reserve",
        "corpus_stream_throughput_cap": float(
            config["runtime"]["corpus_stream_documents_per_second"]
        ),
        "reserve_multiplier": float(
            config["runtime"]["projection_reserve_multiplier"]
        ),
    }
    if statistics != expected:
        raise ValueError("dense calibration statistics record differs")


def _verify_environment(
    environment: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    retriever = config["retriever"]
    expected = {
        "model_id": retriever["id"],
        "model_revision": retriever["revision"],
        "model_type": retriever["architecture"],
        "model_dtype": "torch.bfloat16",
        "parameter_count": int(retriever["parameter_count_expected"]),
        "torch": retriever["torch_version"],
        "transformers": retriever["transformers_version"],
    }
    for field, value in expected.items():
        if environment.get(field) != value:
            raise ValueError(f"dense calibration environment differs: {field}")
    if config["runtime"]["expected_gpu_name_contains"] not in environment.get(
        "gpu_name", ""
    ):
        raise ValueError("dense calibration used unexpected hardware")


def _verify_inventory(run_dir: Path, inventory: Sequence[Mapping[str, Any]]) -> None:
    expected = {row["path"] for row in inventory}
    actual = {path.name for path in run_dir.iterdir() if path.name != "manifest.json"}
    if actual != expected:
        raise ValueError("dense calibration file inventory differs")
    for row in inventory:
        path = run_dir / row["path"]
        if path.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"file size differs: {row['path']}")
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"file checksum differs: {row['path']}")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _git_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
