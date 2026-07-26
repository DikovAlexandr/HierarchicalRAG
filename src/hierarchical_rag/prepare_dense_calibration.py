"""Prepare a label-free systematic corpus sample for dense GPU calibration."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Sequence

from hierarchical_rag.dense_retrieval import systematic_rowids
from hierarchical_rag.experiment import sha256_file, write_json_atomic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=8448)
    parser.add_argument("--corpus-sha256", required=True)
    parser.add_argument("--index-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepare_sample(
        index_path=args.index.resolve(),
        output_path=args.output.resolve(),
        manifest_path=args.manifest.resolve(),
        sample_size=args.sample_size,
        corpus_sha256=args.corpus_sha256,
        index_sha256=args.index_sha256,
    )
    return 0


def prepare_sample(
    *,
    index_path: Path,
    output_path: Path,
    manifest_path: Path,
    sample_size: int,
    corpus_sha256: str,
    index_sha256: str,
) -> None:
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite a calibration sample")
    if sha256_file(index_path) != index_sha256:
        raise ValueError("source index checksum differs from the frozen value")

    connection = sqlite3.connect(index_path)
    connection.execute("PRAGMA query_only = ON")
    try:
        document_count = int(
            connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        )
        rowids = systematic_rowids(document_count, sample_size)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("x", encoding="utf-8", newline="\n") as stream:
            for rowid in rowids:
                row = connection.execute(
                    "SELECT document_id, title, body FROM documents WHERE rowid = ?",
                    (rowid,),
                ).fetchone()
                if row is None:
                    raise RuntimeError(f"missing corpus rowid {rowid}")
                document_id, title, text = row
                stream.write(
                    json.dumps(
                        {
                            "rowid": rowid,
                            "document_id": document_id,
                            "title": title,
                            "text": text,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    finally:
        connection.close()

    write_json_atomic(
        manifest_path,
        {
            "schema_version": 1,
            "status": "complete",
            "selection": "systematic_rowid_v1",
            "document_count": document_count,
            "sample_size": sample_size,
            "first_rowid": rowids[0],
            "last_rowid": rowids[-1],
            "corpus_sha256": corpus_sha256,
            "source_index_sha256": index_sha256,
            "sample_sha256": sha256_file(output_path),
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
