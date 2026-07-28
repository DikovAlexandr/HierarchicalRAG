from __future__ import annotations

import pytest

from hierarchical_rag.audit_dense_calibration import (
    _resource_policy,
    summarize_calibration_records,
)
from hierarchical_rag.dense_retrieval import ComputeProjection


def _rows(count: int) -> list[dict[str, int]]:
    return [{"rowid": 10 + index * 3} for index in range(count)]


def _records(rows: list[dict[str, int]]) -> list[dict[str, int | float]]:
    records: list[dict[str, int | float]] = []
    for index, start in enumerate(range(0, len(rows), 2)):
        batch = rows[start : start + 2]
        records.append(
            {
                "batch_index": index,
                "document_count": len(batch),
                "embedding_dimension": 512,
                "first_rowid": batch[0]["rowid"],
                "last_rowid": batch[-1]["rowid"],
                "processed_tokens": 20,
                "max_l2_norm_error_after_fp16_cast": 0.0001,
            }
        )
    return records


def test_dense_calibration_summary_recomputes_batch_aggregates():
    rows = _rows(5)
    summary = summarize_calibration_records(
        records=_records(rows), measured_rows=rows, batch_size=2, dimension=512
    )

    assert summary == {
        "batch_count": 3,
        "measured_documents": 5,
        "tokens_processed": 60,
        "max_l2_norm_error_after_fp16_cast": 0.0001,
    }


def test_dense_calibration_summary_rejects_tampered_rowids():
    rows = _rows(4)
    records = _records(rows)
    records[1]["first_rowid"] = 999

    with pytest.raises(ValueError, match="first_rowid"):
        summarize_calibration_records(
            records=records, measured_rows=rows, batch_size=2, dimension=512
        )


def test_local_resource_policy_authorizes_only_when_both_gates_pass():
    projection = ComputeProjection(
        measured_documents=8192,
        measured_seconds=136.0,
        documents_per_second=60.0,
        effective_documents_per_second=60.0,
        full_corpus_documents=5_233_235,
        projected_seconds=100_000.0,
        projected_units=None,
        reserve_multiplier=1.25,
    )
    runtime = {
        "full_build_wall_time_limit_seconds": 172_800,
        "full_build_peak_reserved_limit_bytes": 7 * 1024**3,
    }

    accepted = _resource_policy(
        backend="local_docker",
        projection=projection,
        peak_reserved_bytes=5 * 1024**3,
        peak_host_rss_bytes=None,
        runtime=runtime,
    )
    rejected = _resource_policy(
        backend="local_docker",
        projection=projection,
        peak_reserved_bytes=8 * 1024**3,
        peak_host_rss_bytes=None,
        runtime=runtime,
    )

    assert accepted["decision"] == "authorize_separate_local_full_dense_corpus_build"
    assert accepted["wall_time_gate_passed"] is True
    assert accepted["memory_gate_passed"] is True
    assert rejected["decision"] == "do_not_start_full_dense_corpus_build"


def test_ssh_resource_policy_requires_host_memory_gate():
    projection = ComputeProjection(
        measured_documents=8192,
        measured_seconds=136.0,
        documents_per_second=60.0,
        effective_documents_per_second=60.0,
        full_corpus_documents=5_233_235,
        projected_seconds=100_000.0,
        projected_units=None,
        reserve_multiplier=1.25,
    )
    runtime = {
        "full_build_wall_time_limit_seconds": 172_800,
        "full_build_peak_reserved_limit_bytes": 7 * 1024**3,
        "full_build_peak_host_rss_limit_bytes": 6 * 1024**3,
    }

    accepted = _resource_policy(
        backend="ssh_docker",
        projection=projection,
        peak_reserved_bytes=5 * 1024**3,
        peak_host_rss_bytes=4 * 1024**3,
        runtime=runtime,
    )
    rejected = _resource_policy(
        backend="ssh_docker",
        projection=projection,
        peak_reserved_bytes=5 * 1024**3,
        peak_host_rss_bytes=7 * 1024**3,
        runtime=runtime,
    )

    assert accepted["host_memory_gate_passed"] is True
    assert accepted["decision"] == "authorize_separate_local_full_dense_corpus_build"
    assert rejected["host_memory_gate_passed"] is False
    assert rejected["decision"] == "do_not_start_full_dense_corpus_build"
