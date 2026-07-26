from __future__ import annotations

from copy import deepcopy
import json

import pytest

from hierarchical_rag.dense_retrieval import (
    QUERY_INSTRUCTION,
    QWEN3_EMBEDDING_ID,
    QWEN3_EMBEDDING_PARAMETERS,
    QWEN3_EMBEDDING_REVISION,
    instructed_query,
    project_compute,
    projected_embedding_bytes,
    systematic_rowids,
    validate_dense_calibration_protocol,
)
from hierarchical_rag.experiment import load_experiment_config, sha256_file
from hierarchical_rag.fullwiki import WikiDocument, build_fts5_index
from hierarchical_rag.prepare_dense_calibration import prepare_sample
from hierarchical_rag.run_dense_calibration import (
    _progress,
    _validate_sample_manifest,
)


def _config() -> dict:
    return {
        "experiment": {
            "stage": "corpus_side_resource_calibration",
            "decision_ids": ["D027", "D028"],
        },
        "dataset": {
            "split": "corpus_only",
            "labels_observed": False,
            "selection": {
                "method": "systematic_rowid_v1",
                "size": 8448,
                "warmup_documents": 256,
                "measured_documents": 8192,
            },
        },
        "retriever": {
            "id": QWEN3_EMBEDDING_ID,
            "revision": QWEN3_EMBEDDING_REVISION,
            "parameter_count_expected": QWEN3_EMBEDDING_PARAMETERS,
            "frozen": True,
            "dtype": "bfloat16",
            "serialization": "text_only",
            "pooling": "last_non_padding_token",
            "normalization": "truncate_mrl_then_l2",
            "output_dimension": 512,
            "max_input_tokens": 512,
            "attention_implementation": "sdpa",
            "query_instruction": QUERY_INSTRUCTION,
        },
        "runtime": {
            "batch_size": 128,
            "datasphere_units_per_second": 116,
            "corpus_stream_documents_per_second": 6539.4019962511575,
            "projection_reserve_multiplier": 1.25,
        },
    }


def test_systematic_sample_is_stable_unique_and_spans_corpus():
    rowids = systematic_rowids(100, 8)

    assert rowids == (1, 13, 26, 38, 51, 63, 76, 88)
    assert len(set(rowids)) == 8


def test_query_instruction_matches_official_format():
    assert instructed_query("Where is Paris?") == (
        f"Instruct: {QUERY_INSTRUCTION}\nQuery:Where is Paris?"
    )
    with pytest.raises(ValueError, match="cannot be empty"):
        instructed_query("  ")


def test_storage_and_compute_projections_are_explicit():
    assert projected_embedding_bytes(5_233_235, 512) == 5_358_832_640
    projection = project_compute(
        measured_documents=8192,
        measured_seconds=64.0,
        full_corpus_documents=5_233_235,
        units_per_second=116,
        reserve_multiplier=1.25,
    )

    assert projection.documents_per_second == 128.0
    assert projection.effective_documents_per_second == 128.0
    assert projection.projected_seconds == pytest.approx(51_105.810546875)
    assert projection.projected_units == 5_928_275

    capped = project_compute(
        measured_documents=8192,
        measured_seconds=1.0,
        full_corpus_documents=5_233_235,
        units_per_second=116,
        reserve_multiplier=1.25,
        external_throughput_cap=6539.4019962511575,
    )
    assert capped.effective_documents_per_second == pytest.approx(
        6539.4019962511575
    )
    assert capped.projected_seconds == pytest.approx(1000.3275152299966)


def test_dense_calibration_protocol_accepts_frozen_choice():
    validate_dense_calibration_protocol(_config())


def test_dense_calibration_protocol_rejects_quality_relevant_drift():
    config = deepcopy(_config())
    config["retriever"]["serialization"] = "title_plus_text"

    with pytest.raises(ValueError, match="serialization"):
        validate_dense_calibration_protocol(config)


def test_dense_calibration_protocol_rejects_benchmark_labels():
    config = deepcopy(_config())
    config["dataset"]["labels_observed"] = True

    with pytest.raises(ValueError, match="cannot open benchmark labels"):
        validate_dense_calibration_protocol(config)


def test_prepare_calibration_sample_records_provenance(tmp_path):
    index_path = tmp_path / "corpus.sqlite3"
    build_fts5_index(
        tuple(
            WikiDocument(str(index), f"D{index}", f"D{index}", f"Text {index}")
            for index in range(1, 11)
        ),
        index_path,
        corpus_metadata={"sha256": "a" * 64},
    )
    sample_path = tmp_path / "sample.jsonl"
    manifest_path = tmp_path / "sample.manifest.json"
    index_sha256 = sha256_file(index_path)

    prepare_sample(
        index_path=index_path,
        output_path=sample_path,
        manifest_path=manifest_path,
        sample_size=4,
        corpus_sha256="a" * 64,
        index_sha256=index_sha256,
    )

    rows = [json.loads(line) for line in sample_path.read_text().splitlines()]
    manifest = json.loads(manifest_path.read_text())
    assert [row["rowid"] for row in rows] == [1, 3, 6, 8]
    assert manifest["sample_sha256"] == sha256_file(sample_path)
    assert manifest["source_index_sha256"] == index_sha256


def test_versioned_dense_calibration_config_is_valid():
    config = load_experiment_config(
        "experiments/configs/p2-qwen3-embedding-fullwiki-calibration-v1.yaml"
    )

    validate_dense_calibration_protocol(config)


def test_sample_manifest_validation_links_full_corpus_and_sample():
    dataset = {
        "source_sha256": "b" * 64,
        "corpus_sha256": "a" * 64,
        "selection": {"method": "systematic_rowid_v1", "size": 8},
    }
    manifest = {
        "status": "complete",
        "selection": "systematic_rowid_v1",
        "sample_size": 8,
        "sample_sha256": "b" * 64,
        "corpus_sha256": "a" * 64,
    }

    _validate_sample_manifest(manifest, dataset)
    assert "documents=8/8" in _progress(completed=8, total=8, tokens=400)
