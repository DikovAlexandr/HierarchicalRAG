"""Frozen dense-retrieval protocol and resource projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


QWEN3_EMBEDDING_ID = "Qwen/Qwen3-Embedding-0.6B"
QWEN3_EMBEDDING_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
QWEN3_EMBEDDING_PARAMETERS = 595_776_512
QUERY_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


@dataclass(frozen=True, slots=True)
class ComputeProjection:
    measured_documents: int
    measured_seconds: float
    documents_per_second: float
    effective_documents_per_second: float
    full_corpus_documents: int
    projected_seconds: float
    projected_units: int
    reserve_multiplier: float


def instructed_query(question: str) -> str:
    """Apply the checkpoint's documented retrieval-query instruction."""

    if not question.strip():
        raise ValueError("dense retrieval query cannot be empty")
    return f"Instruct: {QUERY_INSTRUCTION}\nQuery:{question}"


def systematic_rowids(total: int, sample_size: int) -> tuple[int, ...]:
    """Return stable one-based positions spread over the complete corpus."""

    if total < 1:
        raise ValueError("total must be positive")
    if sample_size < 1 or sample_size > total:
        raise ValueError("sample_size must be between one and total")
    return tuple((index * total) // sample_size + 1 for index in range(sample_size))


def projected_embedding_bytes(
    document_count: int, dimension: int, bytes_per_value: int = 2
) -> int:
    if min(document_count, dimension, bytes_per_value) < 1:
        raise ValueError("storage projection inputs must be positive")
    return document_count * dimension * bytes_per_value


def project_compute(
    *,
    measured_documents: int,
    measured_seconds: float,
    full_corpus_documents: int,
    units_per_second: int,
    reserve_multiplier: float,
    external_throughput_cap: float | None = None,
) -> ComputeProjection:
    """Extrapolate a calibration while retaining a conservative reserve."""

    if min(
        measured_documents,
        measured_seconds,
        full_corpus_documents,
        units_per_second,
    ) <= 0:
        raise ValueError("compute projection inputs must be positive")
    if reserve_multiplier < 1.0:
        raise ValueError("reserve_multiplier cannot be below one")
    throughput = measured_documents / measured_seconds
    if external_throughput_cap is not None and external_throughput_cap <= 0:
        raise ValueError("external_throughput_cap must be positive")
    effective_throughput = (
        min(throughput, external_throughput_cap)
        if external_throughput_cap is not None
        else throughput
    )
    projected_seconds = (
        full_corpus_documents / effective_throughput * reserve_multiplier
    )
    return ComputeProjection(
        measured_documents=measured_documents,
        measured_seconds=measured_seconds,
        documents_per_second=throughput,
        effective_documents_per_second=effective_throughput,
        full_corpus_documents=full_corpus_documents,
        projected_seconds=projected_seconds,
        projected_units=int(projected_seconds * units_per_second + 0.999999999),
        reserve_multiplier=reserve_multiplier,
    )


def validate_dense_calibration_protocol(config: Mapping[str, Any]) -> None:
    experiment = config["experiment"]
    dataset = config["dataset"]
    retriever = config["retriever"]
    runtime = config["runtime"]

    if experiment["stage"] != "corpus_side_resource_calibration":
        raise ValueError("dense calibration is corpus-side and non-claim-bearing")
    if not {"D027", "D028"}.issubset(experiment["decision_ids"]):
        raise ValueError("dense calibration must cite D027 and D028")
    if dataset["split"] != "corpus_only" or dataset["labels_observed"]:
        raise ValueError("calibration cannot open benchmark labels")
    if dataset["selection"]["method"] != "systematic_rowid_v1":
        raise ValueError("calibration requires the frozen systematic sample")

    expected = {
        "id": QWEN3_EMBEDDING_ID,
        "revision": QWEN3_EMBEDDING_REVISION,
        "parameter_count_expected": QWEN3_EMBEDDING_PARAMETERS,
        "frozen": True,
        "dtype": "bfloat16",
        "serialization": "text_only",
        "pooling": "last_non_padding_token",
        "normalization": "truncate_mrl_then_l2",
        "output_dimension": 512,
        "max_input_tokens": 512,
        "attention_implementation": "sdpa",
    }
    for key, value in expected.items():
        if retriever.get(key) != value:
            raise ValueError(f"retriever.{key} differs from frozen D028 protocol")
    if retriever.get("query_instruction") != QUERY_INSTRUCTION:
        raise ValueError("query instruction differs from the official frozen text")

    measured = int(dataset["selection"]["measured_documents"])
    warmup = int(dataset["selection"]["warmup_documents"])
    if measured < 1024 or warmup < 1:
        raise ValueError("calibration sample is too small")
    if int(dataset["selection"]["size"]) != measured + warmup:
        raise ValueError("sample size must equal warmup plus measured documents")
    if int(runtime["batch_size"]) < 1:
        raise ValueError("runtime.batch_size must be positive")
    if int(runtime["datasphere_units_per_second"]) < 1:
        raise ValueError("DataSphere unit rate must be positive")
    if float(runtime["projection_reserve_multiplier"]) < 1.0:
        raise ValueError("resource projection reserve cannot be below one")
    if float(runtime["corpus_stream_documents_per_second"]) <= 0:
        raise ValueError("corpus stream throughput must be positive")
