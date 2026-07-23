from __future__ import annotations

import json

import pytest
import yaml

from hierarchical_rag.experiment import (
    REQUIRED_RUN_FILES,
    build_manifest,
    load_experiment_config,
    prepare_run_directory,
    write_json_atomic,
)


def test_run_directory_never_overwrites_existing_evidence(tmp_path):
    run = prepare_run_directory(tmp_path / "run")
    (run / "run.log").write_text("evidence", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        prepare_run_directory(run)


def test_manifest_records_config_checksum_and_required_files(tmp_path):
    config = tmp_path / "resolved-config.yaml"
    config.write_text("experiment:\n  id: E0\n", encoding="utf-8")

    manifest = build_manifest(
        experiment_id="E0",
        owner="tester",
        command="python -m runner",
        git_commit="abc123",
        config_path=config,
    )

    assert manifest["experiment_id"] == "E0"
    assert len(manifest["resolved_config_sha256"]) == 64
    assert manifest["required_run_files"] == list(REQUIRED_RUN_FILES)


def test_json_write_is_utf8_and_complete(tmp_path):
    destination = tmp_path / "manifest.json"

    write_json_atomic(destination, {"status": "готово"})

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "status": "готово"
    }
    assert not destination.with_suffix(".json.tmp").exists()


def test_config_loader_rejects_unresolved_placeholders(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "experiment": {"id": "TO_SET"},
                "dataset": {},
                "evaluation": {},
                "runtime": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="placeholders"):
        load_experiment_config(config)
