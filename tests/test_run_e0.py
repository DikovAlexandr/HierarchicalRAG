from __future__ import annotations

import json
from pathlib import Path

import yaml
import pytest

from hierarchical_rag.experiment import (
    REQUIRED_RUN_FILES,
    load_experiment_config,
    sha256_file,
)
from hierarchical_rag.run_e0 import execute


def test_e0_writes_complete_reproducibility_record(tmp_path, hotpot_records):
    config_path, output = _write_config(tmp_path, hotpot_records)

    exit_code = execute(
        config_path,
        load_experiment_config(config_path),
        Path.cwd(),
    )

    assert exit_code == 0
    assert all((output / name).is_file() for name in REQUIRED_RUN_FILES)
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert metrics["reference_match"] is True
    assert manifest["extra"]["status"] == "complete"


def test_e0_preserves_complete_record_when_reference_check_fails(
    tmp_path, hotpot_records
):
    config_path, output = _write_config(
        tmp_path, hotpot_records, expected_value=0.0
    )

    with pytest.raises(RuntimeError, match="do not match"):
        execute(config_path, load_experiment_config(config_path), Path.cwd())

    assert all((output / name).is_file() for name in REQUIRED_RUN_FILES)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["status"] == "failed"


def _write_config(tmp_path, hotpot_records, expected_value=1.0):
    gold = tmp_path / "gold.json"
    predictions = tmp_path / "predictions.json"
    output = tmp_path / "run"
    config_path = tmp_path / "config.yaml"
    gold.write_text(json.dumps(hotpot_records[:1]), encoding="utf-8")
    predictions.write_text(
        json.dumps(
            {
                "answer": {"example-a": "Paris"},
                "sp": {"example-a": [["France", 0], ["Paris", 0]]},
            }
        ),
        encoding="utf-8",
    )

    config = {
        "schema_version": 1,
        "experiment": {
            "id": "e0-test",
            "owner": "test",
            "stage": "test",
            "exact_command": "pytest",
        },
        "dataset": {
            "gold_file": str(gold),
            "gold_sha256": sha256_file(gold),
            "predictions_file": str(predictions),
            "predictions_sha256": sha256_file(predictions),
        },
        "evaluation": {
            "absolute_tolerance": 1.0e-12,
            "reference_evaluator": {"revision": "test"},
            "expected_metrics": {
                name: expected_value
                for name in (
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
            },
        },
        "runtime": {
            "require_clean_worktree": False,
            "output_dir": str(output),
        },
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, output
