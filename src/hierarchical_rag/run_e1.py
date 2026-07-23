"""Run deterministic E1 retrieval diagnostics on HotpotQA fullwiki."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any, Iterator, Mapping, Sequence

import yaml

from hierarchical_rag.experiment import (
    REQUIRED_RUN_FILES,
    build_manifest,
    environment_snapshot,
    file_inventory,
    find_repository_root,
    git_revision,
    git_status,
    load_experiment_config,
    prepare_run_directory,
    sha256_file,
    write_json_atomic,
)
from hierarchical_rag.fullwiki import Fts5BM25Index, canonical_title
from hierarchical_rag.hotpotqa import (
    HotpotExample,
    deterministic_slice,
    load_hotpotqa,
)
from hierarchical_rag.retrieval import ScoredDocument
from hierarchical_rag.statistics import bootstrap_mean


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repository-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    config = load_experiment_config(config_path)
    repository_root = (
        args.repository_root.resolve()
        if args.repository_root is not None
        else find_repository_root(config_path)
    )
    return execute(config_path, config, repository_root)


def execute(
    config_path: Path,
    config: Mapping[str, Any],
    repository_root: Path,
) -> int:
    experiment = config["experiment"]
    dataset = config["dataset"]
    retriever = config.get("retriever")
    evaluation = config["evaluation"]
    runtime = config["runtime"]
    if not isinstance(retriever, Mapping):
        raise ValueError("E1 config requires a retriever mapping")

    status_before = git_status(repository_root)
    if runtime.get("require_clean_worktree", False) and status_before:
        raise RuntimeError("E1 requires a clean Git worktree")
    revision = git_revision(repository_root)
    prerequisite_smoke = runtime.get("prerequisite_smoke")
    prerequisite_smoke_path: Path | None = None
    if prerequisite_smoke is not None:
        if not isinstance(prerequisite_smoke, str) or not prerequisite_smoke:
            raise ValueError("runtime.prerequisite_smoke must be an experiment ID")
        prerequisite_smoke_path = (
            repository_root / "results" / "runs" / prerequisite_smoke
        ).resolve()
        _verify_prerequisite_smoke(
            prerequisite_smoke_path, prerequisite_smoke, revision
        )

    dataset_path = _resolve_path(repository_root, dataset["source_file"])
    index_path = _resolve_path(repository_root, retriever["index_file"])
    index_manifest_path = _resolve_path(
        repository_root, retriever["index_manifest_file"]
    )
    lock_path = _resolve_path(repository_root, runtime["dependency_lock_file"])
    _verify_checksum(dataset_path, dataset["source_sha256"])
    _verify_checksum(index_path, retriever["index_sha256"])
    _verify_checksum(index_manifest_path, retriever["index_manifest_sha256"])
    _verify_checksum(lock_path, runtime["dependency_lock_sha256"])
    ranking_reference = runtime.get("ranking_equivalence_reference")
    ranking_reference_path: Path | None = None
    if ranking_reference is not None:
        if not isinstance(ranking_reference, Mapping):
            raise ValueError("ranking_equivalence_reference must be a mapping")
        ranking_reference_path = _resolve_path(
            repository_root, ranking_reference["retrieval_file"]
        )
        _verify_checksum(ranking_reference_path, ranking_reference["sha256"])

    output_dir = _resolve_path(repository_root, runtime["output_dir"])
    if experiment["stage"] == "confirmatory":
        expected_parent = (repository_root / "results" / "runs").resolve()
        if expected_parent not in output_dir.parents:
            raise ValueError("confirmatory E1 output must be under results/runs")
    run_dir = prepare_run_directory(output_dir)

    arguments = [sys.executable, *sys.argv]
    command = (
        subprocess.list2cmdline(arguments)
        if os.name == "nt"
        else shlex.join(arguments)
    )
    resolved = json.loads(json.dumps(config))
    resolved["experiment"]["git_commit"] = revision
    resolved["experiment"]["actual_command"] = command
    resolved["dataset"]["source_path_resolved"] = str(dataset_path)
    resolved["retriever"]["index_path_resolved"] = str(index_path)
    resolved["retriever"]["index_manifest_path_resolved"] = str(
        index_manifest_path
    )
    resolved["runtime"]["dependency_lock_path_resolved"] = str(lock_path)
    if prerequisite_smoke_path is not None:
        resolved["runtime"]["prerequisite_smoke_path_resolved"] = str(
            prerequisite_smoke_path
        )
    if ranking_reference_path is not None:
        resolved["runtime"]["ranking_equivalence_reference"][
            "retrieval_path_resolved"
        ] = str(ranking_reference_path)
    resolved_config_path = run_dir / "resolved-config.yaml"
    resolved_config_path.write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )
    (run_dir / "command.txt").write_text(
        "Reproduction command:\n"
        f"{experiment['exact_command']}\n\nActual command:\n{command}\n",
        encoding="utf-8",
        newline="\n",
    )
    environment = environment_snapshot()
    environment["image_revision"] = os.environ.get(
        "HIERARCHICAL_RAG_IMAGE_REVISION", "not-containerized"
    )
    write_json_atomic(run_dir / "environment.txt", environment)
    log_lines = [
        f"experiment_id={experiment['id']}",
        f"git_commit={revision}",
        "status=running",
    ]

    try:
        with index_manifest_path.open("r", encoding="utf-8") as stream:
            index_manifest = json.load(stream)
        examples = load_hotpotqa(
            dataset_path,
            require_supporting_context=bool(
                dataset.get("require_supporting_context", True)
            ),
        )
        expected_source_count = int(dataset["source_count"])
        if len(examples) != expected_source_count:
            raise ValueError(
                f"source count mismatch: {len(examples)} != {expected_source_count}"
            )
        selected = _select_examples(examples, dataset["selection"])
        cutoffs = tuple(sorted({int(value) for value in retriever["top_k"]}))
        if not cutoffs or cutoffs[0] < 1:
            raise ValueError("retriever.top_k must contain positive integers")

        with Fts5BM25Index(index_path) as index:
            index_metadata = index.metadata()
            _validate_index_metadata(index_metadata, retriever)
        metrics, statistics, timings = _run_retrieval(
            index_path,
            selected,
            cutoffs,
            run_dir / "retrieval.jsonl",
            run_dir / "predictions.jsonl",
            evaluation["statistics"],
            workers=int(runtime.get("retrieval_workers", 1)),
        )
        if ranking_reference_path is not None:
            _assert_ranking_equivalence(
                ranking_reference_path, run_dir / "retrieval.jsonl"
            )
            metrics["ranking_equivalence"] = {
                "status": "exact_match",
                "reference_sha256": ranking_reference["sha256"],
                "fields": ["example_id", "rank", "document_id", "score"],
            }

        metrics.update(
            {
                "experiment_id": experiment["id"],
                "dataset_count": len(examples),
                "evaluated_count": len(selected),
                "selection": dataset["selection"],
                "candidate_scope": "official_hotpotqa_fullwiki_corpus",
                "index_sha256": retriever["index_sha256"],
                "index_schema_version": index_metadata["schema_version"],
                "runtime": timings,
                "reader_metrics": {
                    "status": "not_applicable",
                    "reason": "E1 retrieval-only run; EM/F1 gap requires a reader.",
                },
            }
        )
        write_json_atomic(run_dir / "metrics.json", metrics)
        write_json_atomic(run_dir / "statistics.json", statistics)
        log_lines.extend(
            [
                f"evaluated_count={len(selected)}",
                f"paragraph_recall_at_10={metrics.get('paragraph_recall_at_10')}",
                "status=complete",
            ]
        )
        if ranking_reference_path is not None:
            log_lines.append("ranking_equivalence=exact_match")
        (run_dir / "run.log").write_text(
            "\n".join(log_lines) + "\n", encoding="utf-8", newline="\n"
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
                    "index_sha256": sha256_file(index_path),
                    "index_manifest_sha256": sha256_file(index_manifest_path),
                    "dependency_lock_sha256": sha256_file(lock_path),
                    **(
                        {
                            "ranking_reference_sha256": sha256_file(
                                ranking_reference_path
                            )
                        }
                        if ranking_reference_path is not None
                        else {}
                    ),
                },
                "index_build": index_manifest,
                "file_inventory": file_inventory(
                    run_dir, exclude={"manifest.json"}
                ),
            },
        )
        write_json_atomic(run_dir / "manifest.json", manifest)
        _assert_required_files(run_dir)
        return 0
    except Exception as error:
        _preserve_failure(
            run_dir,
            error,
            log_lines,
            experiment,
            command,
            revision,
            resolved_config_path,
            config_path,
        )
        raise


def _run_retrieval(
    index_path: Path,
    examples: Sequence[HotpotExample],
    cutoffs: Sequence[int],
    retrieval_path: Path,
    predictions_path: Path,
    statistics_config: Mapping[str, Any],
    *,
    workers: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if workers < 1:
        raise ValueError("runtime.retrieval_workers must be positive")
    recalls = {cutoff: [] for cutoff in cutoffs}
    fact_recalls = {cutoff: [] for cutoff in cutoffs}
    all_supporting = {cutoff: [] for cutoff in cutoffs}
    latencies: list[float] = []
    started = perf_counter()

    with retrieval_path.open("w", encoding="utf-8", newline="\n") as retrieval_stream, predictions_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as prediction_stream:
        for example, ranking, latency in _rank_examples(
            index_path, examples, max(cutoffs), workers
        ):
            gold_titles = tuple(
                dict.fromkeys(
                    canonical_title(fact.title) for fact in example.supporting_facts
                )
            )
            if not gold_titles:
                raise ValueError(f"{example.identifier}: no supporting titles")
            gold_facts = tuple(
                canonical_title(fact.title) for fact in example.supporting_facts
            )
            latencies.append(latency)
            if len(ranking) != max(cutoffs):
                raise RuntimeError(
                    f"{example.identifier}: expected {max(cutoffs)} retrieved "
                    f"documents, received {len(ranking)}"
                )
            retrieved_ids = [item.document.identifier for item in ranking]
            per_example: dict[str, float] = {}
            for cutoff in cutoffs:
                retrieved = set(retrieved_ids[:cutoff])
                recall = len(retrieved & set(gold_titles)) / len(gold_titles)
                fact_recall = sum(title in retrieved for title in gold_facts) / len(
                    gold_facts
                )
                complete = float(set(gold_titles).issubset(retrieved))
                recalls[cutoff].append(recall)
                fact_recalls[cutoff].append(fact_recall)
                all_supporting[cutoff].append(complete)
                per_example[f"paragraph_recall_at_{cutoff}"] = recall
                per_example[f"supporting_fact_recall_at_{cutoff}"] = fact_recall
                per_example[f"all_supporting_paragraphs_at_{cutoff}"] = complete

            retrieval_stream.write(
                json.dumps(
                    {
                        "example_id": example.identifier,
                        "query": example.question,
                        "gold_titles": list(gold_titles),
                        "latency_seconds": latency,
                        "retrieved": [
                            {
                                "rank": rank,
                                "document_id": item.document.identifier,
                                "title": item.document.title,
                                "score": item.score,
                            }
                            for rank, item in enumerate(ranking, start=1)
                        ],
                        "metrics": per_example,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            prediction_stream.write(
                json.dumps(
                    {
                        "example_id": example.identifier,
                        "status": "not_applicable",
                        "reason": "E1 evaluates retrieval without a reader.",
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    elapsed = perf_counter() - started
    aggregate: dict[str, Any] = {"top_k": list(cutoffs)}
    for cutoff in cutoffs:
        aggregate[f"paragraph_recall_at_{cutoff}"] = fmean(recalls[cutoff])
        aggregate[f"supporting_fact_recall_at_{cutoff}"] = fmean(
            fact_recalls[cutoff]
        )
        aggregate[f"all_supporting_paragraphs_at_{cutoff}"] = fmean(
            all_supporting[cutoff]
        )

    primary_name = "paragraph_recall_at_10"
    if 10 not in recalls:
        raise ValueError("E1 primary metric requires top_k=10")
    interval = bootstrap_mean(
        recalls[10],
        confidence_level=float(statistics_config["confidence_level"]),
        resamples=int(statistics_config["resamples"]),
        seed=int(statistics_config["seed"]),
    )
    statistics = {
        "sample_size": len(examples),
        "primary_metric": primary_name,
        "confidence_interval": asdict(interval),
        "significance_test": {
            "status": "not_applicable",
            "reason": "E1 compares no reader systems.",
        },
    }
    timings = {
        "elapsed_seconds": elapsed,
        "retrieval_workers": workers,
        "throughput_examples_per_second": len(examples) / elapsed,
        "latency_mean_seconds": fmean(latencies),
        "latency_p50_seconds": _quantile(latencies, 0.5),
        "latency_p95_seconds": _quantile(latencies, 0.95),
        "peak_rss_bytes": _peak_rss_bytes(),
    }
    return aggregate, statistics, timings


def _rank_examples(
    index_path: Path,
    examples: Sequence[HotpotExample],
    top_k: int,
    workers: int,
) -> Iterator[tuple[HotpotExample, tuple[ScoredDocument, ...], float]]:
    def search(
        example: HotpotExample,
    ) -> tuple[HotpotExample, tuple[ScoredDocument, ...], float]:
        with Fts5BM25Index(index_path) as index:
            query_started = perf_counter()
            ranking = index.search(example.question, top_k=top_k)
            latency = perf_counter() - query_started
        return example, ranking, latency

    if workers == 1:
        with Fts5BM25Index(index_path) as index:
            for example in examples:
                query_started = perf_counter()
                ranking = index.search(example.question, top_k=top_k)
                latency = perf_counter() - query_started
                yield example, ranking, latency
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(search, examples)


def _assert_ranking_equivalence(reference: Path, candidate: Path) -> None:
    if _ranking_signature(candidate) != _ranking_signature(reference):
        raise RuntimeError(
            "parallel ranking differs from the declared sequential reference"
        )


def _ranking_signature(path: Path) -> tuple[tuple[Any, ...], ...]:
    signature: list[tuple[Any, ...]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            retrieved = row.get("retrieved")
            if not isinstance(retrieved, list):
                raise ValueError(f"{path}:{line_number}: missing retrieved list")
            signature.append(
                (
                    row.get("example_id"),
                    tuple(
                        (
                            item.get("rank"),
                            item.get("document_id"),
                            item.get("score"),
                        )
                        for item in retrieved
                    ),
                )
            )
    return tuple(signature)


def _verify_prerequisite_smoke(
    run_dir: Path, experiment_id: str, revision: str
) -> None:
    manifest_path = run_dir / "manifest.json"
    metrics_path = run_dir / "metrics.json"
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    with metrics_path.open("r", encoding="utf-8") as stream:
        metrics = json.load(stream)
    if manifest.get("experiment_id") != experiment_id:
        raise ValueError("prerequisite smoke experiment ID mismatch")
    if manifest.get("git_commit") != revision:
        raise ValueError("prerequisite smoke used a different Git commit")
    extra = manifest.get("extra")
    if not isinstance(extra, Mapping) or extra.get("status") != "complete":
        raise ValueError("prerequisite smoke is not complete")
    inventory = extra.get("file_inventory")
    if not isinstance(inventory, list):
        raise ValueError("prerequisite smoke has no file inventory")
    recorded = {
        item.get("path"): item.get("sha256")
        for item in inventory
        if isinstance(item, Mapping)
    }
    for name in ("metrics.json", "retrieval.jsonl"):
        expected = recorded.get(name)
        if not isinstance(expected, str):
            raise ValueError(f"prerequisite smoke inventory omits {name}")
        _verify_checksum(run_dir / name, expected)
    equivalence = metrics.get("ranking_equivalence")
    if (
        not isinstance(equivalence, Mapping)
        or equivalence.get("status") != "exact_match"
    ):
        raise ValueError("prerequisite smoke did not pass ranking equivalence")


def _select_examples(
    examples: Sequence[HotpotExample], selection: Mapping[str, Any]
) -> tuple[HotpotExample, ...]:
    method = selection["method"]
    size = int(selection["size"])
    if method == "all":
        if size != len(examples):
            raise ValueError(f"all-selection size mismatch: {size} != {len(examples)}")
        return tuple(examples)
    if method == "deterministic_sha256":
        return deterministic_slice(examples, size=size, seed=int(selection["seed"]))
    raise ValueError(f"unsupported selection method: {method}")


def _validate_index_metadata(
    metadata: Mapping[str, Any], retriever: Mapping[str, Any]
) -> None:
    if metadata.get("schema_version") != int(retriever["index_schema_version"]):
        raise ValueError("index schema version mismatch")
    corpus = metadata.get("corpus")
    if not isinstance(corpus, Mapping) or corpus.get("sha256") != retriever[
        "corpus_sha256"
    ]:
        raise ValueError("index corpus checksum mismatch")


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _peak_rss_bytes() -> int | None:
    if platform.system() != "Linux":
        return None
    import resource

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _resolve_path(repository_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repository_root / path).resolve()


def _verify_checksum(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual.casefold() != expected.casefold():
        raise ValueError(f"checksum mismatch for {path}: {actual} != {expected}")


def _assert_required_files(run_dir: Path) -> None:
    missing = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError("run is missing required files: " + ", ".join(missing))


def _preserve_failure(
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
                {"status": "failed", "error": f"{type(error).__name__}: {error}"},
            )
        if not (run_dir / "statistics.json").exists():
            write_json_atomic(
                run_dir / "statistics.json",
                {"status": "not_computed", "reason": "run failed"},
            )
        log_lines.extend([f"error={type(error).__name__}: {error}", "status=failed"])
        (run_dir / "run.log").write_text(
            "\n".join(log_lines) + "\n", encoding="utf-8", newline="\n"
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
                    run_dir, exclude={"manifest.json"}
                ),
            },
        )
        write_json_atomic(run_dir / "manifest.json", manifest)
    except Exception:
        return


if __name__ == "__main__":
    raise SystemExit(main())
