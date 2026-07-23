"""HotpotQA records and deterministic dataset preparation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class SupportingFact:
    """A sentence identified by its document title and zero-based index."""

    title: str
    sentence_id: int


@dataclass(frozen=True, slots=True)
class Paragraph:
    """A titled paragraph represented as benchmark-provided sentences."""

    title: str
    sentences: tuple[str, ...]

    @property
    def text(self) -> str:
        return "".join(self.sentences).strip()


@dataclass(frozen=True, slots=True)
class HotpotExample:
    """The model-independent fields required for HotpotQA evaluation."""

    identifier: str
    question: str
    answer: str | None
    question_type: str | None
    level: str | None
    supporting_facts: tuple[SupportingFact, ...]
    context: tuple[Paragraph, ...]

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        require_supporting_context: bool = True,
    ) -> "HotpotExample":
        identifier = raw.get("_id", raw.get("id"))
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("HotpotQA example requires a non-empty _id or id")

        question = raw.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{identifier}: question must be a non-empty string")

        answer = raw.get("answer")
        if answer is not None and not isinstance(answer, str):
            raise ValueError(f"{identifier}: answer must be a string or null")

        context = _parse_context(raw.get("context"), identifier)
        supporting_facts = _parse_supporting_facts(
            raw.get("supporting_facts", ()), identifier
        )
        if require_supporting_context:
            _validate_supporting_facts(identifier, context, supporting_facts)

        return cls(
            identifier=identifier,
            question=question.strip(),
            answer=answer,
            question_type=_optional_string(raw.get("type"), "type", identifier),
            level=_optional_string(raw.get("level"), "level", identifier),
            supporting_facts=supporting_facts,
            context=context,
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return the official JSON representation without changing content order."""

        record: dict[str, Any] = {
            "_id": self.identifier,
            "question": self.question,
            "answer": self.answer,
            "supporting_facts": [
                [fact.title, fact.sentence_id] for fact in self.supporting_facts
            ],
            "context": [
                [paragraph.title, list(paragraph.sentences)]
                for paragraph in self.context
            ],
        }
        if self.question_type is not None:
            record["type"] = self.question_type
        if self.level is not None:
            record["level"] = self.level
        return record


def load_hotpotqa(
    path: str | Path, *, require_supporting_context: bool = True
) -> tuple[HotpotExample, ...]:
    """Load official JSON or Parquet with explicit context validation."""

    source = Path(path)
    if source.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise RuntimeError(
                "Parquet input requires the 'data' extra: "
                "pip install -e '.[data]'"
            ) from error
        raw = parquet.read_table(source).to_pylist()
    else:
        with source.open("r", encoding="utf-8") as stream:
            raw = json.load(stream)
        if not isinstance(raw, list):
            raise ValueError("HotpotQA JSON must contain a top-level list")

    examples = tuple(
        HotpotExample.from_mapping(
            item, require_supporting_context=require_supporting_context
        )
        for item in raw
    )
    identifiers = [example.identifier for example in examples]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("HotpotQA example identifiers must be unique")
    return examples


def write_hotpotqa(
    path: str | Path,
    examples: Sequence[HotpotExample],
) -> None:
    """Write examples in the official JSON-list representation."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            [example.to_mapping() for example in examples],
            stream,
            ensure_ascii=False,
            indent=2,
        )
        stream.write("\n")


def deterministic_slice(
    examples: Iterable[HotpotExample], size: int, seed: int
) -> tuple[HotpotExample, ...]:
    """Select examples by a stable hash, independent of input order."""

    materialized = tuple(examples)
    if size < 1:
        raise ValueError("size must be positive")
    if size > len(materialized):
        raise ValueError("size cannot exceed the number of examples")
    if len({example.identifier for example in materialized}) != len(materialized):
        raise ValueError("example identifiers must be unique")

    def rank(example: HotpotExample) -> tuple[bytes, str]:
        payload = f"{seed}\0{example.identifier}".encode()
        return hashlib.sha256(payload).digest(), example.identifier

    return tuple(sorted(materialized, key=rank)[:size])


def gold_paragraphs(example: HotpotExample) -> tuple[Paragraph, ...]:
    """Return supporting paragraphs in their original context order."""

    gold_titles = {fact.title for fact in example.supporting_facts}
    return tuple(paragraph for paragraph in example.context if paragraph.title in gold_titles)


def serialize_context(
    question: str, paragraphs: Sequence[Paragraph]
) -> str:
    """Serialize shared evidence deterministically for every reader."""

    lines = [f"Question: {question.strip()}", "", "Evidence:"]
    for document_index, paragraph in enumerate(paragraphs, start=1):
        lines.append(f"[Document {document_index}] {paragraph.title}")
        for sentence_index, sentence in enumerate(paragraph.sentences):
            lines.append(f"({sentence_index}) {sentence.strip()}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slice_manifest(
    source_path: str | Path,
    examples: Sequence[HotpotExample],
    seed: int,
) -> dict[str, Any]:
    """Create lightweight evidence for reproducing a selected slice."""

    return {
        "source_path": str(Path(source_path)),
        "source_sha256": sha256_file(source_path),
        "selection": "sha256(seed + NUL + example_id)",
        "seed": seed,
        "size": len(examples),
        "example_ids": [example.identifier for example in examples],
    }


def supporting_fact_reference_issues(
    examples: Iterable[HotpotExample],
) -> tuple[dict[str, Any], ...]:
    """Report benchmark annotations whose sentence index is outside its paragraph."""

    issues: list[dict[str, Any]] = []
    for example in examples:
        by_title = {paragraph.title: paragraph for paragraph in example.context}
        for fact in example.supporting_facts:
            paragraph = by_title[fact.title]
            if fact.sentence_id >= len(paragraph.sentences):
                issues.append(
                    {
                        "example_id": example.identifier,
                        "title": fact.title,
                        "sentence_id": fact.sentence_id,
                        "paragraph_sentence_count": len(paragraph.sentences),
                    }
                )
    return tuple(issues)


def _parse_context(raw: Any, identifier: str) -> tuple[Paragraph, ...]:
    if isinstance(raw, Mapping):
        titles = raw.get("title")
        sentence_groups = raw.get("sentences")
        if not isinstance(titles, Sequence) or isinstance(titles, str):
            raise ValueError(f"{identifier}: context.title must be a sequence")
        if not isinstance(sentence_groups, Sequence) or isinstance(sentence_groups, str):
            raise ValueError(f"{identifier}: context.sentences must be a sequence")
        raw = list(zip(titles, sentence_groups, strict=True))

    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{identifier}: context must be a sequence")

    paragraphs: list[Paragraph] = []
    for item in raw:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)):
            raise ValueError(f"{identifier}: each context item must be [title, sentences]")
        if len(item) != 2:
            raise ValueError(f"{identifier}: each context item must have two fields")
        title, sentences = item
        if not isinstance(title, str) or not title:
            raise ValueError(f"{identifier}: paragraph title must be non-empty")
        if not isinstance(sentences, Sequence) or isinstance(sentences, (str, bytes)):
            raise ValueError(f"{identifier}: paragraph sentences must be a sequence")
        if not all(isinstance(sentence, str) for sentence in sentences):
            raise ValueError(f"{identifier}: every sentence must be a string")
        paragraphs.append(Paragraph(title=title, sentences=tuple(sentences)))

    if len({paragraph.title for paragraph in paragraphs}) != len(paragraphs):
        raise ValueError(f"{identifier}: context titles must be unique")
    return tuple(paragraphs)


def _parse_supporting_facts(raw: Any, identifier: str) -> tuple[SupportingFact, ...]:
    if isinstance(raw, Mapping):
        titles = raw.get("title")
        sentence_ids = raw.get("sent_id")
        if not isinstance(titles, Sequence) or isinstance(titles, str):
            raise ValueError(f"{identifier}: supporting_facts.title must be a sequence")
        if not isinstance(sentence_ids, Sequence) or isinstance(sentence_ids, str):
            raise ValueError(f"{identifier}: supporting_facts.sent_id must be a sequence")
        raw = list(zip(titles, sentence_ids, strict=True))

    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{identifier}: supporting_facts must be a sequence")

    facts: list[SupportingFact] = []
    for item in raw:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)):
            raise ValueError(f"{identifier}: each supporting fact must be [title, sent_id]")
        if len(item) != 2:
            raise ValueError(f"{identifier}: each supporting fact must have two fields")
        title, sentence_id = item
        if not isinstance(title, str) or not title:
            raise ValueError(f"{identifier}: supporting title must be non-empty")
        if not isinstance(sentence_id, int) or sentence_id < 0:
            raise ValueError(f"{identifier}: sentence ID must be a non-negative integer")
        facts.append(SupportingFact(title=title, sentence_id=sentence_id))
    return tuple(facts)


def _validate_supporting_facts(
    identifier: str,
    context: Sequence[Paragraph],
    supporting_facts: Sequence[SupportingFact],
) -> None:
    by_title = {paragraph.title: paragraph for paragraph in context}
    for fact in supporting_facts:
        paragraph = by_title.get(fact.title)
        if paragraph is None:
            raise ValueError(f"{identifier}: supporting title {fact.title!r} is absent")


def _optional_string(value: Any, field: str, identifier: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{identifier}: {field} must be a string or null")
    return value
