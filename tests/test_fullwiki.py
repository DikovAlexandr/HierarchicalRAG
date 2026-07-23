from __future__ import annotations

import bz2
import io
import json
import tarfile

import pytest

from hierarchical_rag.fullwiki import (
    FTS5_B,
    FTS5_K1,
    Fts5BM25Index,
    WikiDocument,
    build_fts5_index,
    canonical_title,
    fts5_available,
    iter_wiki_documents,
)


def test_jsonl_parser_preserves_sentence_and_paragraph_boundaries(tmp_path):
    source = tmp_path / "wiki.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": 1,
                "title": "Mixed  Case",
                "text": [["First sentence. ", "Second sentence."], ["Next paragraph."]],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    document = next(iter_wiki_documents(source))

    assert document.document_id == "mixed case"
    assert document.text == "First sentence. Second sentence.\nNext paragraph."


def test_stream_official_style_tar_with_bz2_jsonl_member(tmp_path):
    archive_path = tmp_path / "wiki.tar.bz2"
    line = json.dumps({"id": 7, "title": "Paris", "text": [["Paris text."]]})
    compressed = bz2.compress((line + "\n").encode())
    with tarfile.open(archive_path, "w:bz2") as archive:
        info = tarfile.TarInfo("AA/wiki_00.bz2")
        info.size = len(compressed)
        archive.addfile(info, io.BytesIO(compressed))

    documents = tuple(iter_wiki_documents(archive_path))

    assert documents == (
        WikiDocument("7", "paris", "Paris", "Paris text."),
    )


@pytest.mark.skipif(not fts5_available(), reason="SQLite lacks FTS5")
def test_build_and_search_fts5_index(tmp_path):
    documents = (
        WikiDocument("1", "france", "France", "France is in Europe."),
        WikiDocument("2", "paris", "Paris", "Paris is the capital of France."),
        WikiDocument("3", "berlin", "Berlin", "Berlin is the capital of Germany."),
    )
    index_path = tmp_path / "fullwiki.sqlite3"

    built = build_fts5_index(
        documents,
        index_path,
        corpus_metadata={"revision": "fixture-v1"},
    )
    with Fts5BM25Index(index_path) as index:
        ranking = index.search("Paris capital France", top_k=2)
        metadata = index.metadata()

    assert ranking[0].document.identifier == "paris"
    assert ranking[0].score >= ranking[1].score
    assert built["document_count"] == 3
    assert metadata["bm25_k1"] == FTS5_K1
    assert metadata["bm25_b"] == FTS5_B
    assert metadata["corpus"]["revision"] == "fixture-v1"
    assert metadata["indexed_field"] == "text"
    assert metadata["title_indexed_separately"] is False


def test_index_refuses_duplicate_canonical_titles_and_preserves_failure(tmp_path):
    destination = tmp_path / "duplicate.sqlite3"
    documents = (
        WikiDocument("1", canonical_title("Same"), "Same", "First."),
        WikiDocument("2", canonical_title("SAME"), "SAME", "Second."),
    )

    with pytest.raises(ValueError, match="duplicate canonical title"):
        build_fts5_index(documents, destination, corpus_metadata={})

    assert not destination.exists()
    assert destination.with_suffix(".sqlite3.building").exists()
