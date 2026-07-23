"""Build a deterministic SQLite FTS5 index from the HotpotQA fullwiki corpus."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import sqlite3
import sys
from pathlib import Path
from time import perf_counter
from typing import Sequence

from hierarchical_rag.experiment import write_json_atomic
from hierarchical_rag.fullwiki import (
    CorpusReadReport,
    build_fts5_index,
    iter_wiki_documents,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--expected-size", type=int)
    parser.add_argument("--expected-md5")
    parser.add_argument("--max-documents", type=int)
    parser.add_argument(
        "--empty-text-policy", choices=("error", "skip"), default="error"
    )
    parser.add_argument("--expected-record-count", type=int)
    parser.add_argument("--expected-skipped-empty-text", type=int)
    parser.add_argument("--expected-indexed-document-count", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_documents is not None and args.max_documents < 1:
        raise ValueError("max-documents must be positive")

    actual_size = args.corpus.stat().st_size
    if args.expected_size is not None and actual_size != args.expected_size:
        raise ValueError(
            f"corpus size mismatch: {actual_size} != {args.expected_size}"
        )
    checksums = _checksums(args.corpus)
    if args.expected_md5 is not None and checksums["md5"] != args.expected_md5:
        raise ValueError(
            f"corpus MD5 mismatch: {checksums['md5']} != {args.expected_md5}"
        )

    read_report = CorpusReadReport()
    documents = iter_wiki_documents(
        args.corpus,
        empty_text_policy=args.empty_text_policy,
        report=read_report,
    )
    if args.max_documents is not None:
        documents = itertools.islice(documents, args.max_documents)
    audit: dict[str, object] = {}

    def validated_documents():
        yield from documents
        audit.update(read_report.as_dict(complete=args.max_documents is None))
        _require_count(
            "source record",
            read_report.records_seen,
            args.expected_record_count,
        )
        _require_count(
            "skipped empty-text",
            len(read_report.skipped_empty_text),
            args.expected_skipped_empty_text,
        )
        _require_count(
            "indexed document",
            read_report.documents_yielded,
            args.expected_indexed_document_count,
        )

    corpus_metadata = {
        "path": str(args.corpus),
        "source_url": args.source_url,
        "source_revision": args.source_revision,
        "license": args.license,
        "size_bytes": actual_size,
        "md5": checksums["md5"],
        "sha256": checksums["sha256"],
        "max_documents": args.max_documents,
        "empty_text_policy": args.empty_text_policy,
        "audit": audit,
    }

    started = perf_counter()
    index_metadata = build_fts5_index(
        validated_documents(),
        args.index,
        corpus_metadata=corpus_metadata,
    )
    elapsed = perf_counter() - started
    manifest = {
        "status": "complete",
        "corpus": corpus_metadata,
        "index": index_metadata,
        "runtime": {
            "elapsed_seconds": elapsed,
            "documents_per_second": index_metadata["document_count"] / elapsed,
            "python": sys.version,
            "platform": platform.platform(),
            "sqlite_version": sqlite3.sqlite_version,
            "logical_cpu_count": os.cpu_count(),
            "image_revision": os.environ.get(
                "HIERARCHICAL_RAG_IMAGE_REVISION", "not-containerized"
            ),
        },
    }
    write_json_atomic(args.manifest, manifest)
    return 0


def _checksums(path: Path) -> dict[str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return {"md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def _require_count(label: str, actual: int, expected: int | None) -> None:
    if expected is not None and actual != expected:
        raise ValueError(f"{label} count mismatch: {actual} != {expected}")


if __name__ == "__main__":
    raise SystemExit(main())
