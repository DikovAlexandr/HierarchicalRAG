"""Deterministic irrelevant-context construction for H2."""

from __future__ import annotations

import hashlib
from typing import Iterable, Sequence

from hierarchical_rag.retrieval import Document


def select_distractors(
    base_documents: Sequence[Document],
    candidates: Iterable[Document],
    *,
    count: int,
    seed: int,
    example_id: str,
) -> tuple[Document, ...]:
    """Select non-overlapping distractors by a stable per-example hash."""

    if count < 0:
        raise ValueError("count cannot be negative")
    base_ids = {document.identifier for document in base_documents}
    available: dict[str, Document] = {}
    for document in candidates:
        if document.identifier not in base_ids:
            available.setdefault(document.identifier, document)
    if count > len(available):
        raise ValueError("not enough unique distractor candidates")

    def rank(item: tuple[str, Document]) -> tuple[bytes, str]:
        identifier, _ = item
        payload = f"{seed}\0{example_id}\0{identifier}".encode()
        return hashlib.sha256(payload).digest(), identifier

    selected = sorted(available.items(), key=rank)[:count]
    return tuple(document for _, document in selected)


def append_distractors(
    base_documents: Sequence[Document],
    candidates: Iterable[Document],
    *,
    count: int,
    seed: int,
    example_id: str,
) -> tuple[Document, ...]:
    """Append controlled noise without changing the original evidence order."""

    distractors = select_distractors(
        base_documents,
        candidates,
        count=count,
        seed=seed,
        example_id=example_id,
    )
    return tuple(base_documents) + distractors
