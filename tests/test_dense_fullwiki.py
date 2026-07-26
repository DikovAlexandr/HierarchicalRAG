from __future__ import annotations

import json

import pytest

from hierarchical_rag.dense_fullwiki import DenseBuildStore
from hierarchical_rag.dense_retrieval import validate_dense_build_protocol
from hierarchical_rag.experiment import load_experiment_config
from hierarchical_rag.fullwiki import WikiDocument
from hierarchical_rag.run_dense_fullwiki_build import (
    _select_first_overflow_chunks,
    _source_revision,
)


def _document(index: int) -> WikiDocument:
    return WikiDocument(
        wiki_id=str(index),
        document_id=f"Document {index}",
        title=f"Document {index}",
        text=f"Body {index}",
    )


def test_dense_store_resumes_only_committed_contiguous_shards(tmp_path):
    final_dir = tmp_path / "dense-index"
    identity = {"config": "abc", "model": "qwen"}
    store = DenseBuildStore(
        final_dir=final_dir,
        document_count=3,
        dimension=2,
        identity=identity,
    )
    assert store.open() == 0
    first = store.commit_shard(
        documents=(_document(1), _document(2)),
        vector_bytes=b"\x00" * 8,
        tokens_processed=10,
        truncated_documents=0,
        encode_seconds=1.5,
        max_norm_error=0.0001,
    )
    assert first["end_rowid"] == 2
    store.close()

    resumed = DenseBuildStore(
        final_dir=final_dir,
        document_count=3,
        dimension=2,
        identity=identity,
    )
    assert resumed.open() == 2
    resumed.commit_shard(
        documents=(_document(3),),
        vector_bytes=b"\x01" * 4,
        tokens_processed=4,
        truncated_documents=1,
        encode_seconds=0.5,
        max_norm_error=0.0002,
    )
    manifest_path, manifest = resumed.finalize(
        corpus_audit={
            "complete": True,
            "source_record_count": 3,
            "indexed_document_count": 3,
            "skipped_empty_text_count": 0,
            "skipped_documents": [],
        },
        environment={"gpu_name": "fixture"},
    )

    assert manifest_path == final_dir / "manifest.json"
    assert manifest["index"]["document_count"] == 3
    assert manifest["index"]["shard_count"] == 2
    assert manifest["index"]["tokens_processed"] == 14
    assert manifest["index"]["truncated_documents"] == 1
    assert manifest["index"]["vector_size_bytes"] == 12
    progress = json.loads((final_dir / "progress.json").read_text())
    assert progress["completed_documents"] == 3


def test_dense_store_rejects_resume_with_changed_identity(tmp_path):
    final_dir = tmp_path / "dense-index"
    store = DenseBuildStore(
        final_dir=final_dir,
        document_count=2,
        dimension=2,
        identity={"revision": "one"},
    )
    store.open()
    store.close()

    changed = DenseBuildStore(
        final_dir=final_dir,
        document_count=2,
        dimension=2,
        identity={"revision": "two"},
    )
    with pytest.raises(ValueError, match="identity differs"):
        changed.open()


def test_versioned_dense_fullwiki_build_config_is_frozen():
    config = load_experiment_config(
        "experiments/configs/p2-qwen3-embedding-fullwiki-build-v1.yaml"
    )
    validate_dense_build_protocol(config)


def test_overflow_selection_keeps_first_chunk_and_counts_truncation():
    selected, truncated = _select_first_overflow_chunks(
        {
            "input_ids": [[10, 11], [12], [20]],
            "attention_mask": [[1, 1], [1], [1]],
            "overflow_to_sample_mapping": [0, 0, 1],
        },
        batch_size=2,
    )

    assert selected["input_ids"] == [[10, 11], [20]]
    assert selected["attention_mask"] == [[1, 1], [1]]
    assert truncated == 1


def test_bundle_revision_precedes_stale_notebook_git_metadata(tmp_path, monkeypatch):
    revision = "a" * 40
    (tmp_path / ".git").mkdir()
    (tmp_path / "SOURCE_REVISION.txt").write_text(revision + "\n")
    monkeypatch.setenv("HIERARCHICAL_RAG_SOURCE_REVISION", revision)

    assert _source_revision(tmp_path, require_clean=True) == revision
