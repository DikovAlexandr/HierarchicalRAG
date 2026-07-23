"""Run the confirmatory E0 HotpotQA metric-validation experiment."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from hierarchical_rag.hotpotqa import HotpotExample, SupportingFact, load_hotpotqa
from hierarchical_rag.metrics import (
    answer_score,
    evaluate_hotpotqa,
    supporting_fact_score,
)


OFFICIAL_METRIC_NAMES = (
    "em",
    "f1",
    "prec",
    "recall",
    "sp_em",
    "sp_f1",
    "sp_prec",
    "sp_recall",
    "joint_em",
    "joint_f1",
    "joint_prec",
    "joint_recall",
)


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
    evaluation = config["evaluation"]
    runtime = config["runtime"]

    if experiment["stage"] == "confirmatory" and not runtime.get(
        "require_clean_worktree", False
    ):
        raise ValueError("confirmatory E0 requires a clean worktree")

    status_before = git_status(repository_root)
    if runtime.get("require_clean_worktree", False) and status_before:
        raise RuntimeError("confirmatory run requires a clean Git worktree")
    revision = git_revision(repository_root)

    gold_path = _resolve_path(repository_root, dataset["gold_file"])
    predictions_path = _resolve_path(repository_root, dataset["predictions_file"])
    _verify_checksum(gold_path, dataset["gold_sha256"])
    _verify_checksum(predictions_path, dataset["predictions_sha256"])
    lock_path: Path | None = None
    if runtime.get("dependency_lock_file") is not None:
        lock_path = _resolve_path(repository_root, runtime["dependency_lock_file"])
        _verify_checksum(lock_path, runtime["dependency_lock_sha256"])
    elif experiment["stage"] == "confirmatory":
        raise ValueError("confirmatory E0 requires a dependency lock file")

    output_dir = _resolve_path(repository_root, runtime["output_dir"])
    if experiment["stage"] == "confirmatory":
        expected_parent = (repository_root / "results" / "runs").resolve()
        if expected_parent not in output_dir.parents:
            raise ValueError("confirmatory output must be under results/runs")
    run_dir = prepare_run_directory(output_dir)

    arguments = [sys.executable, *sys.argv]
    command = (
        subprocess.list2cmdline(arguments)
        if os.name == "nt"
        else shlex.join(arguments)
    )
    reproduction_command = experiment["exact_command"]
    resolved = json.loads(json.dumps(config))
    resolved["experiment"]["git_commit"] = revision
    resolved["experiment"]["actual_command"] = command
    resolved["dataset"]["gold_path_resolved"] = str(gold_path)
    resolved["dataset"]["predictions_path_resolved"] = str(predictions_path)
    if lock_path is not None:
        resolved["runtime"]["dependency_lock_path_resolved"] = str(lock_path)
    resolved_config_path = run_dir / "resolved-config.yaml"
    resolved_config_path.write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )
    (run_dir / "command.txt").write_text(
        f"Reproduction command:\n{reproduction_command}\n\nActual command:\n{command}\n",
        encoding="utf-8",
        newline="\n",
    )
    write_json_atomic(run_dir / "environment.txt", environment_snapshot())

    log_lines = [
        f"experiment_id={experiment['id']}",
        f"git_commit={revision}",
        "status=running",
    ]
    try:
        examples = load_hotpotqa(gold_path)
        if dataset.get("sample_count") is not None and len(examples) != int(
            dataset["sample_count"]
        ):
            raise ValueError(
                f"sample count mismatch: {len(examples)} != {dataset['sample_count']}"
            )
        with predictions_path.open("r", encoding="utf-8") as stream:
            predictions = json.load(stream)
        answers = _prediction_mapping(predictions, "answer")
        supporting = _prediction_mapping(predictions, "sp")

        aggregate = evaluate_hotpotqa(answers, supporting, examples)
        actual_metrics = aggregate.official_dict()
        expected_metrics = evaluation["expected_metrics"]
        tolerance = float(evaluation["absolute_tolerance"])
        errors = {
            name: abs(actual_metrics[name] - float(expected_metrics[name]))
            for name in OFFICIAL_METRIC_NAMES
        }
        passed = all(error <= tolerance for error in errors.values())

        _write_prediction_records(
            run_dir / "predictions.jsonl", examples, answers, supporting
        )
        _write_not_applicable_retrieval(run_dir / "retrieval.jsonl", examples)
        write_json_atomic(
            run_dir / "metrics.json",
            {
                "count": aggregate.count,
                "actual": actual_metrics,
                "expected": {
                    name: float(expected_metrics[name])
                    for name in OFFICIAL_METRIC_NAMES
                },
                "absolute_errors": errors,
                "absolute_tolerance": tolerance,
                "reference_match": passed,
                "missing_answer_ids": list(aggregate.missing_answer_ids),
                "missing_supporting_fact_ids": list(
                    aggregate.missing_supporting_fact_ids
                ),
            },
        )
        write_json_atomic(
            run_dir / "statistics.json",
            {
                "status": "not_applicable",
                "reason": "E0 validates metric implementation and compares no models.",
                "sample_size": aggregate.count,
            },
        )
        log_lines.append(f"reference_match={str(passed).lower()}")
        log_lines.append("status=complete" if passed else "status=failed")
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
                "status": "complete" if passed else "failed",
                "source_config_path": str(config_path),
                "source_config_sha256": sha256_file(config_path),
                "input_files": {
                    "gold_sha256": sha256_file(gold_path),
                    "predictions_sha256": sha256_file(predictions_path),
                    "dependency_lock_sha256": (
                        sha256_file(lock_path) if lock_path is not None else None
                    ),
                },
                "reference_evaluator": evaluation["reference_evaluator"],
                "file_inventory": file_inventory(
                    run_dir, exclude={"manifest.json"}
                ),
            },
        )
        write_json_atomic(run_dir / "manifest.json", manifest)
        _assert_required_files(run_dir)
        if not passed:
            raise RuntimeError("local metrics do not match the pinned official reference")
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


def _write_prediction_records(
    path: Path,
    examples: Sequence[HotpotExample],
    answers: Mapping[str, Any],
    supporting: Mapping[str, Any],
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for example in examples:
            predicted_answer = str(answers.get(example.identifier, ""))
            predicted_supporting = tuple(
                SupportingFact(str(item[0]), int(item[1]))
                for item in supporting.get(example.identifier, ())
            )
            answer = answer_score(predicted_answer, example.answer or "")
            support = supporting_fact_score(
                predicted_supporting, example.supporting_facts
            )
            record = {
                "example_id": example.identifier,
                "prediction": {
                    "answer": predicted_answer,
                    "supporting_facts": [
                        [fact.title, fact.sentence_id]
                        for fact in predicted_supporting
                    ],
                },
                "gold": {
                    "answer": example.answer,
                    "supporting_facts": [
                        [fact.title, fact.sentence_id]
                        for fact in example.supporting_facts
                    ],
                },
                "answer_metrics": _score_dict(answer),
                "supporting_fact_metrics": _score_dict(support),
            }
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _write_not_applicable_retrieval(
    path: Path, examples: Sequence[HotpotExample]
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for example in examples:
            stream.write(
                json.dumps(
                    {
                        "example_id": example.identifier,
                        "status": "not_applicable",
                        "reason": "E0 validates the scorer without a retriever.",
                        "retrieved": [],
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def _score_dict(score: Any) -> dict[str, float]:
    return {
        "exact_match": score.exact_match,
        "f1": score.f1,
        "precision": score.precision,
        "recall": score.recall,
    }


def _prediction_mapping(payload: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping) or not isinstance(payload.get(field), Mapping):
        raise ValueError(f"predictions require mapping field {field!r}")
    return payload[field]


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
    """Best-effort completion of the required record for a failed run."""

    try:
        if not (run_dir / "predictions.jsonl").exists():
            (run_dir / "predictions.jsonl").write_text("", encoding="utf-8")
        if not (run_dir / "retrieval.jsonl").exists():
            (run_dir / "retrieval.jsonl").write_text("", encoding="utf-8")
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
        if not (run_dir / "run.log").exists():
            log_lines.extend(
                [f"error={type(error).__name__}: {error}", "status=failed"]
            )
            (run_dir / "run.log").write_text(
                "\n".join(log_lines) + "\n", encoding="utf-8", newline="\n"
            )
        if not (run_dir / "manifest.json").exists():
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
        # Never replace the original experimental failure with a bookkeeping error.
        return


if __name__ == "__main__":
    raise SystemExit(main())
