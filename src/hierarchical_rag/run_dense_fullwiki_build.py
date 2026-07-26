"""Build the frozen D028 fullwiki dense corpus index with shard checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from hierarchical_rag.dense_fullwiki import DenseBuildStore
from hierarchical_rag.dense_retrieval import validate_dense_build_protocol
from hierarchical_rag.experiment import (
    REQUIRED_RUN_FILES,
    build_manifest,
    file_inventory,
    git_revision,
    git_status,
    load_experiment_config,
    prepare_run_directory,
    sha256_file,
    write_json_atomic,
)
from hierarchical_rag.fullwiki import CorpusReadReport, WikiDocument, iter_wiki_documents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    config = load_experiment_config(config_path)
    validate_dense_build_protocol(config)
    root = (
        args.repository_root.resolve()
        if args.repository_root
        else config_path.parents[2]
    )
    return execute(config_path=config_path, config=config, root=root)


def execute(
    *, config_path: Path, config: Mapping[str, Any], root: Path
) -> int:
    experiment = config["experiment"]
    dataset = config["dataset"]
    retriever = config["retriever"]
    index = config["index"]
    runtime = config["runtime"]
    revision = _source_revision(root, require_clean=bool(runtime.get("require_clean_worktree")))

    corpus_path = _resolve(root, dataset["source_file"])
    lock_path = _resolve(root, runtime["dependency_lock_file"])
    final_dir = _resolve(root, index["output_dir"])
    run_dir = _resolve(root, runtime["output_dir"])
    _verify_file(corpus_path, int(dataset["source_size_bytes"]), dataset["source_sha256"])
    if _md5_file(corpus_path) != dataset["source_md5"]:
        raise ValueError("fullwiki corpus MD5 differs from the official checksum")
    _verify_file(lock_path, None, runtime["dependency_lock_sha256"])

    identity = {
        "source_revision": revision,
        "source_config_sha256": sha256_file(config_path),
        "corpus_sha256": dataset["source_sha256"],
        "model_id": retriever["id"],
        "model_revision": retriever["revision"],
        "dimension": int(retriever["output_dimension"]),
        "max_input_tokens": int(retriever["max_input_tokens"]),
        "batch_size": int(runtime["batch_size"]),
        "shard_documents": int(runtime["shard_documents"]),
    }
    final_manifest_path = final_dir / "manifest.json"
    if final_manifest_path.is_file():
        final_manifest = _load_json(final_manifest_path)
        if final_manifest.get("identity") != identity:
            raise ValueError("completed dense index identity differs from this config")
        return _write_run_records(
            config_path=config_path,
            config=config,
            root=root,
            run_dir=run_dir,
            index_manifest_path=final_manifest_path,
            index_manifest=final_manifest,
            corpus_path=corpus_path,
            lock_path=lock_path,
            revision=revision,
        )

    attempt_started = time.perf_counter()
    store = DenseBuildStore(
        final_dir=final_dir,
        document_count=int(dataset["expected_indexed_document_count"]),
        dimension=int(index["dimension"]),
        identity=identity,
    )
    completed = store.open()
    resumed_from = completed
    print(
        f"stage=dense_build_store_ready completed_documents={completed} "
        f"total_documents={store.document_count}",
        flush=True,
    )
    try:
        model_runtime = _load_model(retriever, runtime)
        documents = iter_wiki_documents(
            corpus_path,
            empty_text_policy=dataset["empty_text_policy"],
            report=(read_report := CorpusReadReport()),
        )
        for _ in itertools.islice(documents, completed):
            pass

        progress = _progress(total=store.document_count, initial=completed)
        try:
            while completed < store.document_count:
                elapsed = time.perf_counter() - attempt_started
                if elapsed > int(runtime["max_attempt_seconds"]):
                    raise TimeoutError(
                        "dense build reached the preregistered per-attempt wall-clock cap"
                    )
                shard = tuple(
                    itertools.islice(documents, int(runtime["shard_documents"]))
                )
                if not shard:
                    raise ValueError("fullwiki corpus ended before the declared count")
                encoded = _encode_documents(
                    documents=shard,
                    tokenizer=model_runtime["tokenizer"],
                    model=model_runtime["model"],
                    torch_module=model_runtime["torch"],
                    functional=model_runtime["functional"],
                    batch_size=int(runtime["batch_size"]),
                    max_tokens=int(retriever["max_input_tokens"]),
                    dimension=int(retriever["output_dimension"]),
                    progress=progress,
                )
                record = store.commit_shard(
                    documents=shard,
                    vector_bytes=encoded["vector_bytes"],
                    tokens_processed=encoded["tokens_processed"],
                    truncated_documents=encoded["truncated_documents"],
                    encode_seconds=encoded["encode_seconds"],
                    max_norm_error=encoded["max_norm_error"],
                )
                completed = int(record["end_rowid"])
                print(
                    f"stage=shard_committed shard={record['shard_index']} "
                    f"completed_documents={completed} "
                    f"vector_sha256={record['vector_sha256']}",
                    flush=True,
                )
        finally:
            progress.close()

        try:
            extra = next(documents)
        except StopIteration:
            extra = None
        if extra is not None:
            raise ValueError("fullwiki corpus contains more documents than declared")
        corpus_audit = read_report.as_dict(complete=True)
        _validate_corpus_audit(corpus_audit, dataset)
        environment = {
            **model_runtime["environment"],
            "attempt_started_from_document": resumed_from,
            "attempt_wall_seconds": time.perf_counter() - attempt_started,
            "peak_allocated_bytes": int(
                model_runtime["torch"].cuda.max_memory_allocated()
            ),
            "peak_reserved_bytes": int(
                model_runtime["torch"].cuda.max_memory_reserved()
            ),
        }
        final_manifest_path, final_manifest = store.finalize(
            corpus_audit=corpus_audit,
            environment=environment,
        )
    finally:
        store.close()

    print(
        f"stage=dense_index_complete manifest={final_manifest_path}", flush=True
    )
    return _write_run_records(
        config_path=config_path,
        config=config,
        root=root,
        run_dir=run_dir,
        index_manifest_path=final_manifest_path,
        index_manifest=final_manifest,
        corpus_path=corpus_path,
        lock_path=lock_path,
        revision=revision,
    )


def _load_model(
    retriever: Mapping[str, Any], runtime: Mapping[str, Any]
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional
    import transformers
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.__version__ != retriever["torch_version"]:
        raise RuntimeError("PyTorch version differs from dense config")
    if transformers.__version__ != retriever["transformers_version"]:
        raise RuntimeError("Transformers version differs from dense config")
    device = torch.cuda.get_device_properties(0)
    if runtime["expected_gpu_name_contains"] not in device.name:
        raise RuntimeError(f"unexpected GPU: {device.name}")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(
        retriever["id"],
        revision=retriever["revision"],
        padding_side="left",
    )
    model = AutoModel.from_pretrained(
        retriever["id"],
        revision=retriever["revision"],
        dtype=torch.bfloat16,
        attn_implementation=retriever["attention_implementation"],
    ).cuda().eval()
    resolved_revision = getattr(model.config, "_commit_hash", None)
    if resolved_revision != retriever["revision"]:
        raise RuntimeError("resolved dense model revision differs from config")
    if model.config.model_type != retriever["architecture"]:
        raise RuntimeError("resolved dense architecture differs from config")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != int(retriever["parameter_count_expected"]):
        raise RuntimeError("resolved dense parameter count differs from config")
    torch.cuda.synchronize()
    return {
        "tokenizer": tokenizer,
        "model": model,
        "torch": torch,
        "functional": functional,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "model_id": retriever["id"],
            "model_revision": resolved_revision,
            "model_type": model.config.model_type,
            "model_dtype": str(model.dtype),
            "parameter_count": parameter_count,
            "gpu_name": device.name,
            "gpu_total_memory_bytes": int(device.total_memory),
            "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
            "nvidia_smi": _nvidia_smi(),
            "pip_freeze": _pip_freeze(),
        },
    }


def _encode_documents(
    *,
    documents: Sequence[WikiDocument],
    tokenizer: Any,
    model: Any,
    torch_module: Any,
    functional: Any,
    batch_size: int,
    max_tokens: int,
    dimension: int,
    progress: Any,
) -> dict[str, Any]:
    payload = bytearray()
    tokens_processed = 0
    truncated_documents = 0
    max_norm_error = 0.0
    started = time.perf_counter()
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        texts = [row.text for row in batch]
        encoded = tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=max_tokens,
            return_attention_mask=True,
            return_overflowing_tokens=True,
        )
        selected, truncated_count = _select_first_overflow_chunks(
            encoded, batch_size=len(batch)
        )
        truncated_documents += truncated_count
        inputs = tokenizer.pad(
            selected,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        with torch_module.inference_mode():
            output = model(**inputs).last_hidden_state
            pooled = _last_token_pool(output, inputs["attention_mask"], torch_module)
            embeddings = functional.normalize(
                pooled[:, :dimension].float(), p=2, dim=1
            ).to(torch_module.float16)
        if tuple(embeddings.shape) != (len(batch), dimension):
            raise RuntimeError("dense encoder returned an unexpected shape")
        norms = embeddings.float().norm(p=2, dim=1)
        norm_error = float((norms - 1.0).abs().max().item())
        if norm_error > 0.002:
            raise RuntimeError("dense embedding norm exceeds frozen tolerance")
        max_norm_error = max(max_norm_error, norm_error)
        tokens_processed += int(inputs["attention_mask"].sum().item())
        array = embeddings.detach().cpu().contiguous().numpy().astype("<f2", copy=False)
        payload.extend(array.tobytes(order="C"))
        progress.update(len(batch))
    torch_module.cuda.synchronize()
    return {
        "vector_bytes": bytes(payload),
        "tokens_processed": tokens_processed,
        "truncated_documents": truncated_documents,
        "encode_seconds": time.perf_counter() - started,
        "max_norm_error": max_norm_error,
    }


def _select_first_overflow_chunks(
    encoded: Mapping[str, Any], *, batch_size: int
) -> tuple[dict[str, list[Any]], int]:
    mutable = dict(encoded)
    mapping = [
        int(value) for value in mutable.pop("overflow_to_sample_mapping")
    ]
    if batch_size < 1:
        raise ValueError("tokenized batch size must be positive")
    positions: list[int] = []
    chunk_counts = [0] * batch_size
    for position, sample_index in enumerate(mapping):
        if not 0 <= sample_index < batch_size:
            raise RuntimeError("tokenizer returned an invalid overflow mapping")
        if chunk_counts[sample_index] == 0:
            positions.append(position)
        chunk_counts[sample_index] += 1
    if len(positions) != batch_size or any(count < 1 for count in chunk_counts):
        raise RuntimeError("tokenizer overflow mapping does not cover the batch")
    selected = {
        key: [values[position] for position in positions]
        for key, values in mutable.items()
        if key in {"input_ids", "attention_mask", "token_type_ids"}
    }
    if "input_ids" not in selected or "attention_mask" not in selected:
        raise RuntimeError("tokenizer omitted required dense input fields")
    return selected, sum(count > 1 for count in chunk_counts)


def _last_token_pool(hidden: Any, attention_mask: Any, torch_module: Any) -> Any:
    if bool((attention_mask[:, -1] == 1).all().item()):
        return hidden[:, -1]
    indices = attention_mask.sum(dim=1) - 1
    return hidden[
        torch_module.arange(hidden.shape[0], device=hidden.device), indices
    ]


def _write_run_records(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    root: Path,
    run_dir: Path,
    index_manifest_path: Path,
    index_manifest: Mapping[str, Any],
    corpus_path: Path,
    lock_path: Path,
    revision: str,
) -> int:
    experiment = config["experiment"]
    runtime = config["runtime"]
    run = prepare_run_directory(run_dir)
    command = shlex.join([sys.executable, *sys.argv])
    resolved = json.loads(json.dumps(config))
    resolved["experiment"]["source_revision"] = revision
    resolved["experiment"]["actual_command"] = command
    resolved["dataset"]["source_path_resolved"] = str(corpus_path)
    resolved["index"]["manifest_path_resolved"] = str(index_manifest_path)
    resolved["index"]["manifest_sha256"] = sha256_file(index_manifest_path)
    resolved["runtime"]["dependency_lock_path_resolved"] = str(lock_path)
    resolved_path = run / "resolved-config.yaml"
    resolved_path.write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )
    (run / "command.txt").write_text(
        f"Source revision: {revision}\nCommand: {command}\n",
        encoding="utf-8",
        newline="\n",
    )
    not_applicable = {
        "status": "not_applicable",
        "reason": "corpus-only index build performs no query retrieval or prediction",
    }
    for name in ("predictions.jsonl", "retrieval.jsonl"):
        (run / name).write_text(
            json.dumps(not_applicable, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    aggregate = index_manifest["index"]
    units_per_second = int(runtime["datasphere_units_per_second"])
    metrics = {
        "status": "complete_non_claim_bearing_corpus_index_build",
        "labels_observed": False,
        "document_count": aggregate["document_count"],
        "shard_count": aggregate["shard_count"],
        "tokens_processed": aggregate["tokens_processed"],
        "truncated_document_count": aggregate["truncated_documents"],
        "truncated_document_rate": aggregate["truncated_documents"]
        / aggregate["document_count"],
        "encode_seconds": aggregate["encode_seconds"],
        "documents_per_second": aggregate["document_count"]
        / aggregate["encode_seconds"],
        "encoded_unit_estimate": int(
            aggregate["encode_seconds"] * units_per_second + 0.999999999
        ),
        "vector_size_bytes": aggregate["vector_size_bytes"],
        "vector_sha256": aggregate["vector_sha256"],
        "metadata_size_bytes": aggregate["metadata_size_bytes"],
        "metadata_sha256": aggregate["metadata_sha256"],
        "max_l2_norm_error_after_fp16_cast": aggregate["max_norm_error"],
    }
    write_json_atomic(run / "metrics.json", metrics)
    write_json_atomic(
        run / "statistics.json",
        {
            "status": "not_applicable_corpus_index_build",
            "quality_metrics_observed": False,
        },
    )
    write_json_atomic(run / "environment.txt", index_manifest["environment"])
    (run / "run.log").write_text(
        "\n".join(
            (
                f"experiment_id={experiment['id']}",
                f"source_revision={revision}",
                f"document_count={aggregate['document_count']}",
                f"vector_sha256={aggregate['vector_sha256']}",
                "status=complete",
            )
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest = build_manifest(
        experiment_id=experiment["id"],
        owner=experiment["owner"],
        command=command,
        git_commit=revision,
        config_path=resolved_path,
        extra={
            "status": "complete",
            "source_config_path": str(config_path),
            "source_config_sha256": sha256_file(config_path),
            "input_files": {
                "corpus_sha256": sha256_file(corpus_path),
                "dependency_lock_sha256": sha256_file(lock_path),
                "index_manifest_sha256": sha256_file(index_manifest_path),
            },
            "index_manifest": dict(index_manifest),
            "file_inventory": file_inventory(run, exclude={"manifest.json"}),
        },
    )
    write_json_atomic(run / "manifest.json", manifest)
    missing = [name for name in REQUIRED_RUN_FILES if not (run / name).is_file()]
    if missing:
        raise RuntimeError("dense build run record is incomplete: " + ", ".join(missing))
    return 0


def _validate_corpus_audit(
    audit: Mapping[str, Any], dataset: Mapping[str, Any]
) -> None:
    expected = {
        "complete": True,
        "source_record_count": int(dataset["expected_source_record_count"]),
        "indexed_document_count": int(dataset["expected_indexed_document_count"]),
        "skipped_empty_text_count": int(dataset["expected_skipped_empty_text"]),
    }
    for field, value in expected.items():
        if audit.get(field) != value:
            raise ValueError(f"fullwiki corpus audit differs: {field}")


def _verify_file(path: Path, size: int | None, checksum: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if size is not None and path.stat().st_size != size:
        raise ValueError(f"file size differs: {path}")
    if sha256_file(path) != checksum:
        raise ValueError(f"file checksum differs: {path}")


def _md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _progress(*, total: int, initial: int) -> Any:
    from tqdm.auto import tqdm

    return tqdm(
        total=total,
        initial=initial,
        unit="doc",
        desc="fullwiki dense encoding",
        dynamic_ncols=True,
        mininterval=1.0,
    )


def _pip_freeze() -> str:
    return subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _nvidia_smi() -> str:
    return subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _source_revision(root: Path, *, require_clean: bool) -> str:
    marker_path = root / "SOURCE_REVISION.txt"
    marker = (
        marker_path.read_text(encoding="ascii").strip()
        if marker_path.is_file()
        else None
    )
    environment = os.environ.get("HIERARCHICAL_RAG_SOURCE_REVISION")
    if marker and environment and marker != environment:
        raise RuntimeError("bundle marker and source-revision environment differ")
    if (root / ".git").exists():
        revision = git_revision(root)
        if marker and marker != revision:
            raise RuntimeError("bundle marker differs from checked-out Git revision")
        if environment and environment != revision:
            raise RuntimeError("source-revision environment differs from Git revision")
        if require_clean:
            dirty = [
                line
                for line in git_status(root).splitlines()
                if line.strip() != "?? SOURCE_REVISION.txt"
            ]
            if dirty:
                raise RuntimeError("dense fullwiki build requires a clean Git worktree")
        return revision
    revision = environment or marker
    if revision is None or len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise RuntimeError("bundle does not declare a valid source revision")
    return revision


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
