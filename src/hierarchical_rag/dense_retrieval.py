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
    projected_units: int | None
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
    execution_backend = runtime.get("execution_backend", "datasphere")
    unit_rate = int(runtime["datasphere_units_per_second"])
    if execution_backend == "datasphere":
        if unit_rate < 1:
            raise ValueError("DataSphere unit rate must be positive")
    elif execution_backend in {"local_docker", "ssh_docker"}:
        if "D033" not in experiment["decision_ids"]:
            raise ValueError("unmetered dense calibration must cite D033")
        if execution_backend == "ssh_docker" and "D035" not in experiment[
            "decision_ids"
        ]:
            raise ValueError("SSH dense calibration must cite D035")
        if unit_rate != 0:
            raise ValueError("unmetered calibration must use a zero unit rate")
    else:
        raise ValueError("unsupported dense calibration execution backend")
    if float(runtime["projection_reserve_multiplier"]) < 1.0:
        raise ValueError("resource projection reserve cannot be below one")
    if float(runtime["corpus_stream_documents_per_second"]) <= 0:
        raise ValueError("corpus stream throughput must be positive")


def validate_dense_build_protocol(config: Mapping[str, Any]) -> None:
    """Reject changes to the quality-relevant D028 full-corpus protocol."""

    experiment = config["experiment"]
    dataset = config["dataset"]
    retriever = config["retriever"]
    index = config["index"]
    runtime = config["runtime"]
    if experiment["stage"] != "corpus_index_build":
        raise ValueError("dense build must be a corpus-only infrastructure stage")
    if not {"D027", "D028", "D031"}.issubset(experiment["decision_ids"]):
        raise ValueError("dense build must cite D027, D028, and D031")
    if dataset["split"] != "corpus_only" or dataset["labels_observed"]:
        raise ValueError("dense corpus build cannot observe benchmark labels")

    expected_retriever = {
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
    for key, value in expected_retriever.items():
        if retriever.get(key) != value:
            raise ValueError(f"retriever.{key} differs from frozen D028 protocol")
    if retriever.get("query_instruction") != QUERY_INSTRUCTION:
        raise ValueError("query instruction differs from frozen D028 text")

    expected_index = {
        "schema_version": 1,
        "vector_dtype": "float16_little_endian",
        "dimension": 512,
        "document_order": "official_corpus_order_after_audited_empty_skip",
        "similarity": "exact_inner_product",
        "tie_break": "document_id_ascending",
    }
    for key, value in expected_index.items():
        if index.get(key) != value:
            raise ValueError(f"index.{key} differs from frozen D028 protocol")
    if int(dataset["expected_indexed_document_count"]) != 5_233_235:
        raise ValueError("dense build requires the complete audited corpus")
    if int(dataset["expected_source_record_count"]) != 5_233_329:
        raise ValueError("dense build source count differs from corpus audit")
    if int(dataset["expected_skipped_empty_text"]) != 94:
        raise ValueError("dense build empty-text exclusions differ from corpus audit")
    backend = runtime.get("execution_backend", "datasphere_notebook")
    if backend == "local_docker":
        if "D034" not in experiment["decision_ids"]:
            raise ValueError("local dense build must cite D034")
        if int(runtime["batch_size"]) != 32:
            raise ValueError("local dense build batch size differs from D033 calibration")
        if runtime.get("expected_gpu_name_contains") != "RTX 4060":
            raise ValueError("local dense build must use the calibrated RTX 4060")
        if int(runtime.get("datasphere_units_per_second", -1)) != 0:
            raise ValueError("local dense build must record zero DataSphere unit rate")
        if int(runtime["max_attempt_seconds"]) > 172_800:
            raise ValueError("local dense build exceeds the D033 wall-time gate")
        if not runtime.get("calibration_audit_file"):
            raise ValueError("local dense build must pin its calibration audit")
        if len(str(runtime.get("calibration_audit_sha256", ""))) != 64:
            raise ValueError("local dense build calibration audit checksum is invalid")
        batch_size = 32
    elif backend == "datasphere_notebook":
        if int(runtime["batch_size"]) != 128:
            raise ValueError("dense build batch size differs from successful calibration")
        batch_size = 128
    else:
        raise ValueError("unsupported dense build execution backend")
    shard_documents = int(runtime["shard_documents"])
    if shard_documents < batch_size or shard_documents % batch_size:
        raise ValueError("dense shard size must be a positive batch-size multiple")
    if int(runtime["max_attempt_seconds"]) < 1:
        raise ValueError("dense build attempt limit must be positive")
    if int(runtime["minimum_initial_free_bytes"]) < 10 * 1024**3:
        raise ValueError("dense build initial free-space guard is too small")
    if int(runtime["minimum_resume_free_bytes"]) < 2 * 1024**3:
        raise ValueError("dense build resume free-space guard is too small")
