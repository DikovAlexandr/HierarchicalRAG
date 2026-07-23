"""Streaming HotpotQA fullwiki corpus parsing and SQLite FTS5 retrieval."""

from __future__ import annotations

import bz2
import hashlib
import io
import json
import sqlite3
import tarfile
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator, Mapping, Sequence, TextIO

from hierarchical_rag.retrieval import Document, ScoredDocument, tokenize


INDEX_SCHEMA_VERSION = 2
FTS5_TOKENIZER = "unicode61 remove_diacritics 2"
FTS5_K1 = 1.2
FTS5_B = 0.75


@dataclass(frozen=True, slots=True)
class WikiDocument:
    wiki_id: str
    document_id: str
    title: str
    text: str

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], *, allow_empty_text: bool = False
    ) -> "WikiDocument":
        title = raw.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("Wikipedia document requires a non-empty title")
        wiki_id = raw.get("id")
        if not isinstance(wiki_id, (str, int)):
            raise ValueError(f"{title}: Wikipedia id must be a string or integer")
        text = _join_text(raw.get("text"), title)
        if not text and not allow_empty_text:
            raise ValueError(f"{title}: Wikipedia text cannot be empty")
        normalized_title = title.strip()
        return cls(
            wiki_id=str(wiki_id),
            document_id=canonical_title(normalized_title),
            title=normalized_title,
            text=text,
        )


def canonical_title(title: str) -> str:
    """Normalize a HotpotQA title without collapsing case-distinct pages."""

    normalized = unicodedata.normalize("NFKC", title)
    return " ".join(normalized.split())


@dataclass(slots=True)
class CorpusReadReport:
    records_seen: int = 0
    documents_yielded: int = 0
    skipped_empty_text: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self, *, complete: bool) -> dict[str, Any]:
        return {
            "complete": complete,
            "source_record_count": self.records_seen,
            "indexed_document_count": self.documents_yielded,
            "skipped_empty_text_count": len(self.skipped_empty_text),
            "skipped_documents": list(self.skipped_empty_text),
        }


def iter_wiki_documents(
    path: str | Path,
    *,
    empty_text_policy: str = "error",
    report: CorpusReadReport | None = None,
) -> Iterator[WikiDocument]:
    """Stream documents from official tar.bz2, bz2 JSONL, or plain JSONL."""

    if empty_text_policy not in {"error", "skip"}:
        raise ValueError("empty_text_policy must be 'error' or 'skip'")
    read_report = report if report is not None else CorpusReadReport()
    source = Path(path)
    suffixes = [suffix.casefold() for suffix in source.suffixes]
    if suffixes[-2:] == [".tar", ".bz2"]:
        yield from _iter_tar_bz2(source, empty_text_policy, read_report)
    elif suffixes[-1:] == [".bz2"]:
        with bz2.open(source, "rt", encoding="utf-8") as stream:
            yield from _iter_jsonl(
                stream, str(source), empty_text_policy, read_report
            )
    else:
        with source.open("r", encoding="utf-8") as stream:
            yield from _iter_jsonl(
                stream, str(source), empty_text_policy, read_report
            )


def build_fts5_index(
    documents: Iterable[WikiDocument],
    destination: str | Path,
    *,
    corpus_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an atomic immutable FTS5 index in deterministic input order."""

    output = Path(destination)
    if output.exists():
        raise FileExistsError(f"index already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".building")
    if temporary.exists():
        raise FileExistsError(f"incomplete index already exists: {temporary}")

    connection = sqlite3.connect(temporary)
    count = 0
    try:
        _create_schema(connection)
        seen_ids: set[str] = set()
        with connection:
            for rowid, document in enumerate(documents, start=1):
                if document.document_id in seen_ids:
                    raise ValueError(
                        f"duplicate normalized title: {document.document_id!r}"
                    )
                seen_ids.add(document.document_id)
                connection.execute(
                    """
                    INSERT INTO documents(rowid, document_id, wiki_id, title, body)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        rowid,
                        document.document_id,
                        document.wiki_id,
                        document.title,
                        document.text,
                    ),
                )
                count += 1
            if count == 0:
                raise ValueError("cannot build an index without documents")
            connection.execute(
                "INSERT INTO documents_fts(documents_fts) VALUES('rebuild')"
            )
            connection.execute(
                "INSERT INTO documents_fts(documents_fts) VALUES('optimize')"
            )
            metadata = {
                "schema_version": INDEX_SCHEMA_VERSION,
                "document_count": count,
                "sqlite_version": sqlite3.sqlite_version,
                "fts5_tokenizer": FTS5_TOKENIZER,
                "ranking": "sqlite_fts5_bm25",
                "bm25_k1": FTS5_K1,
                "bm25_b": FTS5_B,
                "body_weight": 1.0,
                "indexed_field": "text",
                "title_indexed_separately": False,
                "query_operator": "OR over unique tokenized terms",
                "corpus": dict(corpus_metadata),
            }
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                (
                    (key, json.dumps(value, ensure_ascii=False, sort_keys=True))
                    for key, value in metadata.items()
                ),
            )
        connection.close()
        temporary.replace(output)
    except Exception:
        connection.close()
        raise

    return {
        **metadata,
        "index_path": str(output),
        "index_size_bytes": output.stat().st_size,
        "index_sha256": sha256_file(output),
    }


class Fts5BM25Index:
    """Read-only deterministic search over a completed fullwiki index."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA query_only = ON")
        metadata = self.metadata()
        if metadata.get("schema_version") != INDEX_SCHEMA_VERSION:
            raise ValueError("unsupported fullwiki index schema")

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Fts5BM25Index":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def metadata(self) -> dict[str, Any]:
        rows = self.connection.execute("SELECT key, value FROM metadata ORDER BY key")
        return {key: json.loads(value) for key, value in rows}

    def search(self, query: str, top_k: int) -> tuple[ScoredDocument, ...]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        terms = tuple(dict.fromkeys(tokenize(query)))
        if not terms:
            return ()
        match_query = " OR ".join(f'"{term}"' for term in terms)
        rows = self.connection.execute(
            """
            SELECT d.document_id, d.title, d.body,
                   bm25(documents_fts) AS raw_score
            FROM documents_fts
            JOIN documents AS d ON d.rowid = documents_fts.rowid
            WHERE documents_fts MATCH ?
            ORDER BY raw_score ASC, d.document_id ASC
            LIMIT ?
            """,
            (match_query, top_k),
        )
        return tuple(
            ScoredDocument(
                document=Document(
                    identifier=document_id,
                    title=title,
                    text=body,
                ),
                score=-float(raw_score),
            )
            for document_id, title, body, raw_score in rows
        )


def fts5_available() -> bool:
    try:
        connection = sqlite3.connect(":memory:")
        connection.execute("CREATE VIRTUAL TABLE probe USING fts5(text)")
        connection.close()
        return True
    except sqlite3.OperationalError:
        return False


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _create_schema(connection: sqlite3.Connection) -> None:
    if not fts5_available():
        raise RuntimeError("this Python SQLite build does not include FTS5")
    connection.executescript(
        f"""
        PRAGMA journal_mode = DELETE;
        PRAGMA synchronous = NORMAL;
        PRAGMA temp_store = FILE;
        PRAGMA user_version = {INDEX_SCHEMA_VERSION};

        CREATE TABLE documents (
            rowid INTEGER PRIMARY KEY,
            document_id TEXT NOT NULL UNIQUE,
            wiki_id TEXT NOT NULL,
            title TEXT NOT NULL,
            body TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE documents_fts USING fts5(
            body,
            content='documents',
            content_rowid='rowid',
            tokenize='{FTS5_TOKENIZER}'
        );

        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )


def _iter_tar_bz2(
    path: Path, empty_text_policy: str, report: CorpusReadReport
) -> Iterator[WikiDocument]:
    with tarfile.open(path, mode="r:bz2") as archive:
        for member in archive:
            if not member.isfile():
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            yield from _iter_member(
                extracted, member.name, empty_text_policy, report
            )


def _iter_member(
    stream: BinaryIO,
    name: str,
    empty_text_policy: str,
    report: CorpusReadReport,
) -> Iterator[WikiDocument]:
    binary: BinaryIO
    if name.casefold().endswith(".bz2"):
        binary = bz2.BZ2File(stream)
    else:
        binary = stream
    with io.TextIOWrapper(binary, encoding="utf-8") as text_stream:
        yield from _iter_jsonl(text_stream, name, empty_text_policy, report)


def _iter_jsonl(
    stream: TextIO,
    source_name: str,
    empty_text_policy: str,
    report: CorpusReadReport,
) -> Iterator[WikiDocument]:
    for line_number, line in enumerate(stream, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            if not isinstance(raw, Mapping):
                raise ValueError("record must be a mapping")
            report.records_seen += 1
            document = WikiDocument.from_mapping(raw, allow_empty_text=True)
            if not document.text:
                if empty_text_policy == "error":
                    raise ValueError(
                        f"{document.title}: Wikipedia text cannot be empty"
                    )
                report.skipped_empty_text.append(
                    {
                        "wiki_id": document.wiki_id,
                        "title": document.title,
                        "source": source_name,
                        "line": line_number,
                        "reason": "empty_text",
                    }
                )
                continue
            report.documents_yielded += 1
            yield document
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"{source_name}:{line_number}: {error}") from error


def _join_text(raw: Any, title: str) -> str:
    if isinstance(raw, str):
        return raw.strip()
    if not isinstance(raw, Sequence) or isinstance(raw, (bytes, bytearray)):
        raise ValueError(f"{title}: text must be a string or nested sequence")
    if all(isinstance(item, str) for item in raw):
        return "".join(raw).strip()
    parts: list[str] = []
    for item in raw:
        nested = _join_text(item, title)
        if nested:
            parts.append(nested)
    return "\n".join(part.strip() for part in parts if part.strip())
