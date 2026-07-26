"""Independently audit a complete E1 retrieval artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from hierarchical_rag.experiment import (
    load_experiment_config,
    sha256_file,
    write_json_atomic,
)
from hierarchical_rag.fullwiki import canonical_title
from hierarchical_rag.hotpotqa import HotpotExample, load_hotpotqa
from hierarchical_rag.statistics import bootstrap_mean


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_e1_artifact(
        run_dir=args.run_dir.resolve(),
        source_config=args.source_config.resolve(),
    )
    if args.output:
        write_json_atomic(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


def audit_e1_artifact(*, run_dir: Path, source_config: Path) -> dict[str, Any]:
    repository_root = source_config.resolve().parents[2]
    manifest = _load_json(run_dir / "manifest.json")
    reported_metrics = _load_json(run_dir / "metrics.json")
    reported_statistics = _load_json(run_dir / "statistics.json")
    resolved_config_path = run_dir / "resolved-config.yaml"
    config = load_experiment_config(resolved_config_path)

    if manifest.get("extra", {}).get("status") != "complete":
        raise ValueError("E1 manifest is not complete")
    if manifest["experiment_id"] != config["experiment"]["id"]:
        raise ValueError("manifest experiment ID differs from config")
    resolved_revision = config["experiment"].get("source_revision")
    if (
        resolved_revision is not None
        and manifest["git_commit"] != resolved_revision
    ):
        raise ValueError("manifest and resolved-config revisions differ")
    if manifest["resolved_config_sha256"] != sha256_file(resolved_config_path):
        raise ValueError("resolved config checksum differs from manifest")
    _verify_inventory(run_dir, manifest["extra"]["file_inventory"])
    if sha256_file(source_config) != manifest["extra"]["source_config_sha256"]:
        raise ValueError("source config checksum differs from manifest")

    checked_inputs: dict[str, str] = {}
    for section, path_key, checksum_key in (
        ("dataset", "source_file", "source_sha256"),
        ("retriever", "index_file", "index_sha256"),
        ("retriever", "index_manifest_file", "index_manifest_sha256"),
        ("runtime", "dependency_lock_file", "dependency_lock_sha256"),
    ):
        path = _resolve(repository_root, config[section][path_key])
        actual = sha256_file(path)
        if actual != config[section][checksum_key]:
            raise ValueError(f"checksum mismatch for {path}")
        checked_inputs[path_key] = actual

    examples = load_hotpotqa(
        _resolve(repository_root, config["dataset"]["source_file"]),
        require_supporting_context=bool(
            config["dataset"].get("require_supporting_context", True)
        ),
    )
    retrieval = _load_jsonl(run_dir / "retrieval.jsonl")
    predictions = _load_jsonl(run_dir / "predictions.jsonl")
    cutoffs = tuple(int(value) for value in config["retriever"]["top_k"])
    verified_metrics, error_analysis, latencies = summarize_e1_records(
        examples=examples,
        retrieval=retrieval,
        predictions=predictions,
        cutoffs=cutoffs,
    )

    expected_count = int(config["dataset"]["selection"]["size"])
    if len(examples) != expected_count:
        raise ValueError("dataset count differs from frozen selection")
    expected_metric_fields = {
        **verified_metrics,
        "experiment_id": config["experiment"]["id"],
        "dataset_count": expected_count,
        "evaluated_count": expected_count,
        "top_k": list(cutoffs),
        "selection": config["dataset"]["selection"],
        "candidate_scope": "official_hotpotqa_fullwiki_corpus",
        "index_schema_version": config["retriever"]["index_schema_version"],
        "index_sha256": config["retriever"]["index_sha256"],
        "reader_metrics": {
            "status": "not_applicable",
            "reason": "E1 retrieval-only run; EM/F1 gap requires a reader.",
        },
    }
    for field, expected in expected_metric_fields.items():
        if reported_metrics.get(field) != expected:
            raise ValueError(f"reported E1 metric differs: {field}")

    runtime = reported_metrics.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("E1 metrics lack runtime")
    for field, expected in (
        ("retrieval_workers", int(config["runtime"]["retrieval_workers"])),
        ("latency_mean_seconds", fmean(latencies)),
        ("latency_p50_seconds", _quantile(latencies, 0.5)),
        ("latency_p95_seconds", _quantile(latencies, 0.95)),
    ):
        if runtime.get(field) != expected:
            raise ValueError(f"reported E1 runtime differs: {field}")
    elapsed = float(runtime["elapsed_seconds"])
    if runtime["throughput_examples_per_second"] != expected_count / elapsed:
        raise ValueError("reported E1 throughput differs from elapsed time")
    if int(runtime["peak_rss_bytes"]) <= 0:
        raise ValueError("reported E1 peak RSS must be positive")

    statistics_config = config["evaluation"]["statistics"]
    interval = bootstrap_mean(
        [float(row["metrics"]["paragraph_recall_at_10"]) for row in retrieval],
        confidence_level=float(statistics_config["confidence_level"]),
        resamples=int(statistics_config["resamples"]),
        seed=int(statistics_config["seed"]),
    )
    expected_statistics = {
        "sample_size": expected_count,
        "primary_metric": "paragraph_recall_at_10",
        "confidence_interval": {
            "estimate": interval.estimate,
            "low": interval.low,
            "high": interval.high,
            "confidence_level": interval.confidence_level,
            "method": interval.method,
            "resamples": interval.resamples,
        },
        "significance_test": {
            "status": "not_applicable",
            "reason": "E1 compares no reader systems.",
        },
    }
    if reported_statistics != expected_statistics:
        raise ValueError("reported E1 statistics differ from raw records")

    environment = _load_json(run_dir / "environment.txt")
    if environment.get("image_revision") != manifest["git_commit"]:
        raise ValueError("environment image revision differs from run revision")
    if int(environment.get("logical_cpu_count", 0)) <= 0:
        raise ValueError("environment lacks a positive CPU count")

    return {
        "schema_version": 1,
        "experiment_id": config["experiment"]["id"],
        "audit": {
            "integrity_status": "passed",
            "original_status": "complete",
            "audit_source_revision": _git_revision(repository_root),
        },
        "provenance": {
            "original_source_revision": manifest["git_commit"],
            "run_dir": _display_path(run_dir, repository_root),
            "source_config": _display_path(source_config, repository_root),
            "source_config_sha256": sha256_file(source_config),
            "resolved_config_sha256": sha256_file(resolved_config_path),
            "manifest_sha256": sha256_file(run_dir / "manifest.json"),
            "metrics_sha256": sha256_file(run_dir / "metrics.json"),
            "statistics_sha256": sha256_file(run_dir / "statistics.json"),
            "retrieval_sha256": sha256_file(run_dir / "retrieval.jsonl"),
            "predictions_sha256": sha256_file(run_dir / "predictions.jsonl"),
            "checked_input_sha256": checked_inputs,
        },
        "verified_metrics": verified_metrics,
        "verified_statistics": expected_statistics,
        "reported_runtime": dict(runtime),
        "error_analysis": error_analysis,
        "claim_scope": (
            "Confirmatory E1 retrieval evidence only; no reader, H1, or "
            "end-to-end answer-quality claim."
        ),
    }


def summarize_e1_records(
    *,
    examples: Sequence[HotpotExample],
    retrieval: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    cutoffs: Sequence[int],
) -> tuple[dict[str, float], dict[str, Any], list[float]]:
    if not examples or not (len(examples) == len(retrieval) == len(predictions)):
        raise ValueError("E1 examples, retrieval, and predictions must align")
    normalized_cutoffs = tuple(sorted(set(int(value) for value in cutoffs)))
    if not normalized_cutoffs or normalized_cutoffs[0] < 1:
        raise ValueError("E1 cutoffs must be positive")
    max_cutoff = normalized_cutoffs[-1]
    recalls = {cutoff: [] for cutoff in normalized_cutoffs}
    fact_recalls = {cutoff: [] for cutoff in normalized_cutoffs}
    all_supporting = {cutoff: [] for cutoff in normalized_cutoffs}
    latencies: list[float] = []
    type_rows: dict[str, list[tuple[float, float]]] = defaultdict(list)
    level_rows: dict[str, list[tuple[float, float]]] = defaultdict(list)
    rank_bins: Counter[str] = Counter()
    missing_titles: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()

    for example, row, prediction in zip(
        examples, retrieval, predictions, strict=True
    ):
        if row.get("example_id") != example.identifier:
            raise ValueError("retrieval example order differs from dataset")
        if prediction != {
            "example_id": example.identifier,
            "status": "not_applicable",
            "reason": "E1 evaluates retrieval without a reader.",
        }:
            raise ValueError(f"{example.identifier}: invalid E1 prediction record")
        if row.get("query") != example.question:
            raise ValueError(f"{example.identifier}: retrieval query differs")
        gold_titles = tuple(
            dict.fromkeys(
                canonical_title(fact.title) for fact in example.supporting_facts
            )
        )
        gold_facts = tuple(
            canonical_title(fact.title) for fact in example.supporting_facts
        )
        if list(gold_titles) != row.get("gold_titles"):
            raise ValueError(f"{example.identifier}: gold titles differ")

        ranked = row.get("retrieved")
        if not isinstance(ranked, list) or len(ranked) != max_cutoff:
            raise ValueError(f"{example.identifier}: invalid ranking length")
        retrieved_ids: list[str] = []
        previous_score = math.inf
        for expected_rank, item in enumerate(ranked, start=1):
            score = float(item["score"])
            document_id = str(item["document_id"])
            if (
                int(item["rank"]) != expected_rank
                or document_id != canonical_title(str(item["title"]))
                or not math.isfinite(score)
                or score > previous_score
            ):
                raise ValueError(f"{example.identifier}: invalid ranking record")
            previous_score = score
            retrieved_ids.append(document_id)
        if len(retrieved_ids) != len(set(retrieved_ids)):
            raise ValueError(f"{example.identifier}: duplicate retrieved document")

        expected_per_example: dict[str, float] = {}
        for cutoff in normalized_cutoffs:
            retrieved_set = set(retrieved_ids[:cutoff])
            recall = len(retrieved_set & set(gold_titles)) / len(gold_titles)
            fact_recall = sum(
                title in retrieved_set for title in gold_facts
            ) / len(gold_facts)
            complete = float(set(gold_titles).issubset(retrieved_set))
            recalls[cutoff].append(recall)
            fact_recalls[cutoff].append(fact_recall)
            all_supporting[cutoff].append(complete)
            expected_per_example[f"paragraph_recall_at_{cutoff}"] = recall
            expected_per_example[f"supporting_fact_recall_at_{cutoff}"] = fact_recall
            expected_per_example[f"all_supporting_paragraphs_at_{cutoff}"] = complete
        if row.get("metrics") != expected_per_example:
            raise ValueError(f"{example.identifier}: per-example metrics differ")

        latency = float(row["latency_seconds"])
        if not math.isfinite(latency) or latency <= 0:
            raise ValueError(f"{example.identifier}: invalid latency")
        latencies.append(latency)
        recall_at_max = recalls[max_cutoff][-1]
        complete_at_max = all_supporting[max_cutoff][-1]
        type_rows[example.question_type or "unknown"].append(
            (recall_at_max, complete_at_max)
        )
        level_rows[example.level or "unknown"].append(
            (recall_at_max, complete_at_max)
        )
        retrieved_gold_count = len(set(gold_titles) & set(retrieved_ids))
        if retrieved_gold_count == 0:
            outcomes["none"] += 1
        elif retrieved_gold_count == len(gold_titles):
            outcomes["complete"] += 1
        else:
            outcomes["partial"] += 1
        rank_by_id = {
            document_id: rank
            for rank, document_id in enumerate(retrieved_ids, start=1)
        }
        for title in gold_titles:
            rank = rank_by_id.get(title)
            if rank is None:
                rank_bins["missing"] += 1
                missing_titles[title] += 1
            elif rank == 1:
                rank_bins["1"] += 1
            elif rank == 2:
                rank_bins["2"] += 1
            elif rank <= 5:
                rank_bins["3-5"] += 1
            else:
                rank_bins["6-10"] += 1

    aggregate: dict[str, float] = {}
    for cutoff in normalized_cutoffs:
        aggregate[f"paragraph_recall_at_{cutoff}"] = fmean(recalls[cutoff])
        aggregate[f"supporting_fact_recall_at_{cutoff}"] = fmean(
            fact_recalls[cutoff]
        )
        aggregate[f"all_supporting_paragraphs_at_{cutoff}"] = fmean(
            all_supporting[cutoff]
        )
    total_gold = sum(rank_bins.values())
    error_analysis = {
        "paragraph_outcomes_at_10": {
            key: {"count": outcomes[key], "rate": outcomes[key] / len(examples)}
            for key in ("complete", "partial", "none")
        },
        "gold_paragraph_rank_bins": {
            key: {"count": rank_bins[key], "rate": rank_bins[key] / total_gold}
            for key in ("1", "2", "3-5", "6-10", "missing")
        },
        "by_question_type": _group_summary(type_rows),
        "by_level": _group_summary(level_rows),
        "most_frequent_missing_gold_titles": [
            {"title": title, "missing_count": count}
            for title, count in sorted(
                missing_titles.items(), key=lambda item: (-item[1], item[0])
            )[:20]
        ],
    }
    return aggregate, error_analysis, latencies


def _group_summary(
    groups: Mapping[str, Sequence[tuple[float, float]]],
) -> dict[str, Any]:
    return {
        group: {
            "count": len(rows),
            "paragraph_recall_at_10": fmean(row[0] for row in rows),
            "all_supporting_paragraphs_at_10": fmean(row[1] for row in rows),
        }
        for group, rows in sorted(groups.items())
    }


def _verify_inventory(
    run_dir: Path, inventory: Sequence[Mapping[str, Any]]
) -> None:
    expected_paths = {str(item["path"]) for item in inventory}
    actual_paths = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if expected_paths != actual_paths:
        raise ValueError("manifest inventory paths differ from run directory")
    for item in inventory:
        path = run_dir / str(item["path"])
        if path.stat().st_size != int(item["size_bytes"]):
            raise ValueError(f"size mismatch for {path}")
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"checksum mismatch for {path}")


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _resolve(repository_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository_root / path


def _git_revision(repository_root: Path) -> str:
    return subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository_root.as_posix()}",
            "rev-parse",
            "HEAD",
        ],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _display_path(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
