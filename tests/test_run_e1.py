from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from hierarchical_rag.experiment import (
    REQUIRED_RUN_FILES,
    load_experiment_config,
    sha256_file,
)
from hierarchical_rag.fullwiki import (
    Fts5BM25Index,
    WikiDocument,
    build_fts5_index,
    fts5_available,
)
from hierarchical_rag.run_e1 import _verify_prerequisite_smoke, execute


@pytest.mark.skipif(not fts5_available(), reason="SQLite lacks FTS5")
def test_e1_writes_complete_retrieval_record(tmp_path):
    config_path, output = _write_fixture(tmp_path, document_count=10)

    exit_code = execute(
        config_path,
        load_experiment_config(config_path),
        Path.cwd(),
    )

    assert exit_code == 0
    assert all((output / name).is_file() for name in REQUIRED_RUN_FILES)
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    retrieval = json.loads(
        (output / "retrieval.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert metrics["paragraph_recall_at_1"] == 1.0
    assert metrics["evaluated_count"] == 1
    assert metrics["reader_metrics"]["status"] == "not_applicable"
    assert len(retrieval["retrieved"]) == 10
    assert manifest["extra"]["status"] == "complete"


@pytest.mark.skipif(not fts5_available(), reason="SQLite lacks FTS5")
def test_e1_preserves_failure_when_top_k_cannot_be_filled(tmp_path):
    config_path, output = _write_fixture(tmp_path, document_count=9)

    with pytest.raises(RuntimeError, match="expected 10 retrieved documents"):
        execute(
            config_path,
            load_experiment_config(config_path),
            Path.cwd(),
        )

    assert all((output / name).is_file() for name in REQUIRED_RUN_FILES)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["status"] == "failed"


@pytest.mark.skipif(not fts5_available(), reason="SQLite lacks FTS5")
def test_e1_parallel_ranking_exactly_matches_declared_reference(tmp_path):
    sequential_config, sequential_output = _write_fixture(
        tmp_path / "sequential", document_count=10
    )
    execute(
        sequential_config,
        load_experiment_config(sequential_config),
        Path.cwd(),
    )
    reference = sequential_output / "retrieval.jsonl"
    parallel_config, parallel_output = _write_fixture(
        tmp_path / "parallel",
        document_count=10,
        workers=2,
        ranking_reference=reference,
    )

    execute(
        parallel_config,
        load_experiment_config(parallel_config),
        Path.cwd(),
    )

    metrics = json.loads(
        (parallel_output / "metrics.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (parallel_output / "manifest.json").read_text(encoding="utf-8")
    )
    assert metrics["ranking_equivalence"]["status"] == "exact_match"
    assert metrics["runtime"]["retrieval_workers"] == 2
    _verify_prerequisite_smoke(
        parallel_output, "e1-test", manifest["git_commit"]
    )


def _write_fixture(
    tmp_path: Path,
    document_count: int,
    *,
    workers: int = 1,
    ranking_reference: Path | None = None,
) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    dataset_path = tmp_path / "fullwiki.json"
    index_path = tmp_path / "index.sqlite3"
    index_manifest_path = tmp_path / "index.manifest.json"
    lock_path = tmp_path / "eval.lock"
    config_path = tmp_path / "config.yaml"
    output = tmp_path / "run"

    dataset_path.write_text(
        json.dumps(
            [
                {
                    "_id": "example-a",
                    "question": "What is the capital of France?",
                    "answer": "Paris",
                    "type": "bridge",
                    "level": "easy",
                    "supporting_facts": [["France", 0]],
                    "context": [["Unrelated", ["Not the gold paragraph."]]],
                }
            ]
        ),
        encoding="utf-8",
    )
    documents = [
        WikiDocument(
            wiki_id="1",
            document_id="France",
            title="France",
            text="France France France capital.",
        )
    ]
    documents.extend(
        WikiDocument(
            wiki_id=str(index + 2),
            document_id=f"Other {index:02d}",
            title=f"Other {index:02d}",
            text="A capital is a city.",
        )
        for index in range(document_count - 1)
    )
    build_fts5_index(
        documents,
        index_path,
        corpus_metadata={"sha256": "fixture-corpus-sha256"},
    )
    index_manifest_path.write_text(
        json.dumps({"status": "complete", "document_count": document_count}),
        encoding="utf-8",
    )
    lock_path.write_text("fixture==1.0 --hash=sha256:fixture\n", encoding="utf-8")

    config = {
        "schema_version": 1,
        "experiment": {
            "id": "e1-test",
            "hypothesis_id": "E1",
            "owner": "test",
            "stage": "test",
            "exact_command": "pytest",
        },
        "dataset": {
            "source_file": str(dataset_path),
            "source_sha256": sha256_file(dataset_path),
            "source_count": 1,
            "require_supporting_context": False,
            "selection": {"method": "all", "size": 1, "seed": None},
        },
        "retriever": {
            "index_file": str(index_path),
            "index_sha256": sha256_file(index_path),
            "index_manifest_file": str(index_manifest_path),
            "index_manifest_sha256": sha256_file(index_manifest_path),
            "index_schema_version": 2,
            "corpus_sha256": "fixture-corpus-sha256",
            "top_k": [1, 2, 5, 10],
        },
        "evaluation": {
            "primary_metric": "paragraph_recall_at_10",
            "statistics": {
                "confidence_level": 0.95,
                "resamples": 10,
                "seed": 42,
            },
        },
        "runtime": {
            "require_clean_worktree": False,
            "retrieval_workers": workers,
            "dependency_lock_file": str(lock_path),
            "dependency_lock_sha256": sha256_file(lock_path),
            "output_dir": str(output),
        },
    }
    if ranking_reference is not None:
        config["runtime"]["ranking_equivalence_reference"] = {
            "retrieval_file": str(ranking_reference),
            "sha256": sha256_file(ranking_reference),
        }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, output
