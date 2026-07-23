"""Model-independent lexical retrieval and retrieval metrics."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Iterable, Sequence

from hierarchical_rag.hotpotqa import HotpotExample, Paragraph


@dataclass(frozen=True, slots=True)
class Document:
    identifier: str
    title: str
    text: str


@dataclass(frozen=True, slots=True)
class ScoredDocument:
    document: Document
    score: float


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


class BM25Index:
    """Small, deterministic Okapi BM25 implementation for the first baseline."""

    def __init__(
        self,
        documents: Iterable[Document],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be in [0, 1]")
        self.documents = tuple(documents)
        if not self.documents:
            raise ValueError("at least one document is required")
        if len({document.identifier for document in self.documents}) != len(
            self.documents
        ):
            raise ValueError("document identifiers must be unique")

        self.k1 = k1
        self.b = b
        self._tokens = tuple(tokenize(document.text) for document in self.documents)
        self._term_frequencies = tuple(Counter(tokens) for tokens in self._tokens)
        self._lengths = tuple(len(tokens) for tokens in self._tokens)
        self._average_length = sum(self._lengths) / len(self._lengths)

        document_frequency: dict[str, int] = defaultdict(int)
        for tokens in self._tokens:
            for term in set(tokens):
                document_frequency[term] += 1
        count = len(self.documents)
        self._idf = {
            term: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, top_k: int) -> tuple[ScoredDocument, ...]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query_terms = set(tokenize(query))
        scored: list[ScoredDocument] = []
        for index, document in enumerate(self.documents):
            score = 0.0
            length = self._lengths[index]
            normalization = 1 - self.b
            if self._average_length:
                normalization += self.b * length / self._average_length
            for term in query_terms:
                term_frequency = self._term_frequencies[index].get(term, 0)
                if not term_frequency:
                    continue
                denominator = term_frequency + self.k1 * normalization
                score += (
                    self._idf[term]
                    * term_frequency
                    * (self.k1 + 1)
                    / denominator
                )
            scored.append(ScoredDocument(document=document, score=score))

        scored.sort(key=lambda item: (-item.score, item.document.identifier))
        return tuple(scored[: min(top_k, len(scored))])


def paragraph_documents(paragraphs: Iterable[Paragraph]) -> tuple[Document, ...]:
    return tuple(
        Document(identifier=paragraph.title, title=paragraph.title, text=paragraph.text)
        for paragraph in paragraphs
    )


def recall_at_k(
    ranking: Sequence[ScoredDocument],
    relevant_document_ids: Iterable[str],
    top_k: int,
) -> float:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    relevant = set(relevant_document_ids)
    if not relevant:
        raise ValueError("at least one relevant document is required")
    retrieved = {
        item.document.identifier for item in ranking[: min(top_k, len(ranking))]
    }
    return len(relevant & retrieved) / len(relevant)


def evaluate_bm25_candidate_reranking(
    examples: Sequence[HotpotExample],
    *,
    top_ks: Sequence[int],
    k1: float = 1.5,
    b: float = 0.75,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Evaluate BM25 within each benchmark-provided distractor candidate set."""

    if not examples:
        raise ValueError("at least one example is required")
    cutoffs = tuple(sorted(set(top_ks)))
    if not cutoffs or cutoffs[0] < 1:
        raise ValueError("top_ks must contain positive integers")

    rows: list[dict[str, Any]] = []
    paragraph_recalls: dict[int, list[float]] = {cutoff: [] for cutoff in cutoffs}
    supporting_fact_recalls: dict[int, list[float]] = {
        cutoff: [] for cutoff in cutoffs
    }

    for example in examples:
        index = BM25Index(paragraph_documents(example.context), k1=k1, b=b)
        ranking = index.search(example.question, top_k=max(cutoffs))
        gold_titles = {fact.title for fact in example.supporting_facts}
        if not gold_titles:
            raise ValueError(f"{example.identifier}: no supporting paragraphs")

        per_cutoff: dict[str, float] = {}
        for cutoff in cutoffs:
            retrieved_titles = {
                item.document.identifier for item in ranking[:cutoff]
            }
            paragraph_recall = recall_at_k(ranking, gold_titles, cutoff)
            supporting_fact_recall = sum(
                fact.title in retrieved_titles for fact in example.supporting_facts
            ) / len(example.supporting_facts)
            paragraph_recalls[cutoff].append(paragraph_recall)
            supporting_fact_recalls[cutoff].append(supporting_fact_recall)
            per_cutoff[f"paragraph_recall_at_{cutoff}"] = paragraph_recall
            per_cutoff[f"supporting_fact_recall_at_{cutoff}"] = (
                supporting_fact_recall
            )

        rows.append(
            {
                "example_id": example.identifier,
                "query": example.question,
                "candidate_count": len(example.context),
                "gold_titles": sorted(gold_titles),
                "retrieved": [
                    {
                        "rank": rank,
                        "document_id": item.document.identifier,
                        "title": item.document.title,
                        "score": item.score,
                    }
                    for rank, item in enumerate(ranking, start=1)
                ],
                "metrics": per_cutoff,
            }
        )

    aggregate: dict[str, Any] = {
        "count": len(examples),
        "candidate_scope": "benchmark_provided_distractor_context",
        "retriever": "okapi_bm25",
        "parameters": {"k1": k1, "b": b},
        "top_ks": list(cutoffs),
    }
    for cutoff in cutoffs:
        aggregate[f"paragraph_recall_at_{cutoff}"] = fmean(
            paragraph_recalls[cutoff]
        )
        aggregate[f"supporting_fact_recall_at_{cutoff}"] = fmean(
            supporting_fact_recalls[cutoff]
        )
    return aggregate, tuple(rows)
