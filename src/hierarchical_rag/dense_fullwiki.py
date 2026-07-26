"""Resumable storage for the frozen fullwiki dense corpus index."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Mapping, Sequence

from hierarchical_rag.experiment import sha256_file, write_json_atomic
from hierarchical_rag.fullwiki import WikiDocument


VECTOR_BYTES = 2


class DenseBuildStore:
    """Commit vectors and metadata together at deterministic shard boundaries."""

    def __init__(
        self,
        *,
        final_dir: Path,
        document_count: int,
        dimension: int,
        identity: Mapping[str, Any],
    ) -> None:
        self.final_dir = final_dir
        self.building_dir = final_dir.with_name(final_dir.name + ".building")
        self.document_count = document_count
        self.dimension = dimension
        self.identity = dict(identity)
        self.vector_path = self.building_dir / "vectors.fp16"
        self.metadata_path = self.building_dir / "documents.sqlite3"
        self.info_path = self.building_dir / "build-info.json"
        self.progress_path = self.building_dir / "progress.json"
        self.connection: sqlite3.Connection | None = None

    def open(self) -> int:
        if self.final_dir.exists():
            raise FileExistsError(f"completed dense index already exists: {self.final_dir}")
        if self.building_dir.exists():
            self._open_existing()
        else:
            self._create()
        assert self.connection is not None
        completed = self._completed_documents()
        self._validate_contiguous_shards(completed)
        return completed

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def commit_shard(
        self,
        *,
        documents: Sequence[WikiDocument],
        vector_bytes: bytes,
        tokens_processed: int,
        truncated_documents: int,
        encode_seconds: float,
        max_norm_error: float,
    ) -> dict[str, Any]:
        if self.connection is None:
            raise RuntimeError("dense build store is not open")
        if not documents:
            raise ValueError("cannot commit an empty dense shard")
        start_rowid = self._completed_documents() + 1
        end_rowid = start_rowid + len(documents) - 1
        if end_rowid > self.document_count:
            raise ValueError("dense shard exceeds declared corpus size")
        expected_bytes = len(documents) * self.dimension * VECTOR_BYTES
        if len(vector_bytes) != expected_bytes:
            raise ValueError("dense shard byte count differs from its shape")
        if tokens_processed < len(documents):
            raise ValueError("dense shard token count is invalid")
        if not 0 <= truncated_documents <= len(documents):
            raise ValueError("dense shard truncation count is invalid")
        if not 0 <= max_norm_error <= 0.002:
            raise ValueError("dense shard norm error is invalid")

        offset = (start_rowid - 1) * self.dimension * VECTOR_BYTES
        with self.vector_path.open("r+b") as stream:
            stream.seek(offset)
            stream.write(vector_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        vector_sha256 = hashlib.sha256(vector_bytes).hexdigest()
        shard_index = self._shard_count()
        with self.connection:
            self.connection.executemany(
                """
                INSERT INTO documents(rowid, document_id, wiki_id, title, body)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    (
                        rowid,
                        document.document_id,
                        document.wiki_id,
                        document.title,
                        document.text,
                    )
                    for rowid, document in enumerate(documents, start=start_rowid)
                ),
            )
            self.connection.execute(
                """
                INSERT INTO shards(
                    shard_index, start_rowid, end_rowid, document_count,
                    vector_sha256, tokens_processed, truncated_documents,
                    encode_seconds, max_norm_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    shard_index,
                    start_rowid,
                    end_rowid,
                    len(documents),
                    vector_sha256,
                    tokens_processed,
                    truncated_documents,
                    encode_seconds,
                    max_norm_error,
                ),
            )
        write_json_atomic(
            self.progress_path,
            {
                "status": "building",
                "completed_documents": end_rowid,
                "total_documents": self.document_count,
                "last_committed_shard": shard_index,
                "last_vector_sha256": vector_sha256,
                "aggregate": self.aggregate(),
            },
        )
        return {
            "shard_index": shard_index,
            "start_rowid": start_rowid,
            "end_rowid": end_rowid,
            "document_count": len(documents),
            "vector_sha256": vector_sha256,
        }

    def aggregate(self) -> dict[str, Any]:
        if self.connection is None:
            raise RuntimeError("dense build store is not open")
        row = self.connection.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(document_count), 0),
                   COALESCE(SUM(tokens_processed), 0),
                   COALESCE(SUM(truncated_documents), 0),
                   COALESCE(SUM(encode_seconds), 0.0),
                   COALESCE(MAX(max_norm_error), 0.0)
            FROM shards
            """
        ).fetchone()
        assert row is not None
        return {
            "shard_count": int(row[0]),
            "document_count": int(row[1]),
            "tokens_processed": int(row[2]),
            "truncated_documents": int(row[3]),
            "encode_seconds": float(row[4]),
            "max_norm_error": float(row[5]),
        }

    def finalize(
        self,
        *,
        corpus_audit: Mapping[str, Any],
        environment: Mapping[str, Any],
    ) -> tuple[Path, dict[str, Any]]:
        if self.connection is None:
            raise RuntimeError("dense build store is not open")
        aggregate = self.aggregate()
        if aggregate["document_count"] != self.document_count:
            raise ValueError("dense index cannot finalize before all documents exist")
        if int(corpus_audit["indexed_document_count"]) != self.document_count:
            raise ValueError("corpus audit and dense index counts differ")
        self._validate_contiguous_shards(self.document_count)
        self.connection.execute("PRAGMA optimize")
        self.connection.commit()
        self.close()

        manifest = {
            "schema_version": 1,
            "status": "complete",
            "identity": self.identity,
            "corpus_audit": dict(corpus_audit),
            "index": {
                "document_count": self.document_count,
                "dimension": self.dimension,
                "vector_dtype": "float16_little_endian",
                "vector_size_bytes": self.vector_path.stat().st_size,
                "vector_sha256": sha256_file(self.vector_path),
                "metadata_size_bytes": self.metadata_path.stat().st_size,
                "metadata_sha256": sha256_file(self.metadata_path),
                **aggregate,
            },
            "environment": dict(environment),
        }
        write_json_atomic(
            self.building_dir / "build-info.json",
            {
                "schema_version": 1,
                "status": "complete",
                "document_count": self.document_count,
                "dimension": self.dimension,
                "identity": self.identity,
            },
        )
        write_json_atomic(
            self.building_dir / "progress.json",
            {
                "status": "complete",
                "completed_documents": self.document_count,
                "total_documents": self.document_count,
                "last_committed_shard": aggregate["shard_count"] - 1,
                "aggregate": aggregate,
            },
        )
        write_json_atomic(self.building_dir / "manifest.json", manifest)
        self.building_dir.replace(self.final_dir)
        return self.final_dir / "manifest.json", manifest

    def _create(self) -> None:
        self.building_dir.parent.mkdir(parents=True, exist_ok=True)
        self.building_dir.mkdir()
        write_json_atomic(
            self.info_path,
            {
                "schema_version": 1,
                "status": "building",
                "document_count": self.document_count,
                "dimension": self.dimension,
                "identity": self.identity,
            },
        )
        write_json_atomic(
            self.progress_path,
            {
                "status": "building",
                "completed_documents": 0,
                "total_documents": self.document_count,
                "last_committed_shard": None,
                "last_vector_sha256": None,
                "aggregate": {
                    "shard_count": 0,
                    "document_count": 0,
                    "tokens_processed": 0,
                    "truncated_documents": 0,
                    "encode_seconds": 0.0,
                    "max_norm_error": 0.0,
                },
            },
        )
        vector_size = self.document_count * self.dimension * VECTOR_BYTES
        with self.vector_path.open("xb") as stream:
            stream.truncate(vector_size)
        self.connection = sqlite3.connect(self.metadata_path)
        self.connection.executescript(
            """
            PRAGMA journal_mode = DELETE;
            PRAGMA synchronous = FULL;
            CREATE TABLE documents (
                rowid INTEGER PRIMARY KEY,
                document_id TEXT NOT NULL UNIQUE,
                wiki_id TEXT NOT NULL,
                title TEXT NOT NULL,
                body TEXT NOT NULL
            );
            CREATE TABLE shards (
                shard_index INTEGER PRIMARY KEY,
                start_rowid INTEGER NOT NULL UNIQUE,
                end_rowid INTEGER NOT NULL UNIQUE,
                document_count INTEGER NOT NULL,
                vector_sha256 TEXT NOT NULL,
                tokens_processed INTEGER NOT NULL,
                truncated_documents INTEGER NOT NULL,
                encode_seconds REAL NOT NULL,
                max_norm_error REAL NOT NULL
            );
            """
        )
        self.connection.commit()

    def _open_existing(self) -> None:
        info = json.loads(self.info_path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": 1,
            "status": "building",
            "document_count": self.document_count,
            "dimension": self.dimension,
            "identity": self.identity,
        }
        if info != expected:
            raise ValueError("partial dense index identity differs from this run")
        expected_size = self.document_count * self.dimension * VECTOR_BYTES
        if self.vector_path.stat().st_size != expected_size:
            raise ValueError("partial dense vector file has an unexpected size")
        self.connection = sqlite3.connect(self.metadata_path)

    def _completed_documents(self) -> int:
        assert self.connection is not None
        row = self.connection.execute(
            "SELECT COALESCE(MAX(end_rowid), 0) FROM shards"
        ).fetchone()
        return int(row[0])

    def _shard_count(self) -> int:
        assert self.connection is not None
        row = self.connection.execute("SELECT COUNT(*) FROM shards").fetchone()
        return int(row[0])

    def _validate_contiguous_shards(self, completed: int) -> None:
        assert self.connection is not None
        next_rowid = 1
        for shard_index, start, end, count in self.connection.execute(
            """
            SELECT shard_index, start_rowid, end_rowid, document_count
            FROM shards ORDER BY shard_index
            """
        ):
            if int(shard_index) < 0 or int(start) != next_rowid:
                raise ValueError("dense shard sequence is not contiguous")
            if int(end) - int(start) + 1 != int(count):
                raise ValueError("dense shard row range differs from its count")
            next_rowid = int(end) + 1
        if next_rowid - 1 != completed:
            raise ValueError("dense shard completion cursor is inconsistent")
