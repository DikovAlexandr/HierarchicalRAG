"""Evaluate BM25 reranking over HotpotQA's provided distractor candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Sequence

from hierarchical_rag.hotpotqa import load_hotpotqa, sha256_file
from hierarchical_rag.retrieval import evaluate_bm25_candidate_reranking


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--retrieval-output", required=True, type=Path)
    parser.add_argument("--metrics-output", required=True, type=Path)
    parser.add_argument("--top-k", nargs="+", type=int, default=[1, 2, 5, 10])
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    examples = load_hotpotqa(args.dataset)
    started = perf_counter()
    metrics, rows = evaluate_bm25_candidate_reranking(
        examples,
        top_ks=args.top_k,
        k1=args.k1,
        b=args.b,
    )
    elapsed = perf_counter() - started
    metrics["dataset_path"] = str(args.dataset)
    metrics["dataset_sha256"] = sha256_file(args.dataset)
    metrics["elapsed_seconds"] = elapsed
    metrics["throughput_examples_per_second"] = len(examples) / elapsed

    args.retrieval_output.parent.mkdir(parents=True, exist_ok=True)
    with args.retrieval_output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_output.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
