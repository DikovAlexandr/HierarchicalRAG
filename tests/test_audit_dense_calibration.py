from __future__ import annotations

import pytest

from hierarchical_rag.audit_dense_calibration import summarize_calibration_records


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
