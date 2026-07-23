from __future__ import annotations

import hashlib
import json

from hierarchical_rag.build_fullwiki_index import main


def test_build_command_records_corpus_and_runtime(tmp_path):
    corpus = tmp_path / "intro.jsonl"
    index = tmp_path / "index.sqlite3"
    manifest = tmp_path / "manifest.json"
    corpus.write_text(
        '{"id": 1, "title": "Paris", "text": [["Paris is in France."]]}\n',
        encoding="utf-8",
    )
    payload = corpus.read_bytes()

    exit_code = main(
        [
            "--corpus",
            str(corpus),
            "--index",
            str(index),
            "--manifest",
            str(manifest),
            "--source-url",
            "https://example.invalid/fixture",
            "--source-revision",
            "fixture-v1",
            "--license",
            "CC-BY-SA-4.0",
            "--expected-size",
            str(len(payload)),
            "--expected-md5",
            hashlib.md5(payload, usedforsecurity=False).hexdigest(),
        ]
    )

    record = json.loads(manifest.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert record["status"] == "complete"
    assert record["index"]["document_count"] == 1
    assert record["corpus"]["source_revision"] == "fixture-v1"
