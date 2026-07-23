"""Evaluate HotpotQA predictions with official-compatible metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from hierarchical_rag.hotpotqa import load_hotpotqa
from hierarchical_rag.metrics import evaluate_hotpotqa


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with args.predictions.open("r", encoding="utf-8") as stream:
        predictions: Mapping[str, Any] = json.load(stream)

    answers = predictions.get("answer")
    supporting_facts = predictions.get("sp")
    if not isinstance(answers, Mapping) or not isinstance(supporting_facts, Mapping):
        raise ValueError("prediction JSON requires mapping fields 'answer' and 'sp'")

    metrics = evaluate_hotpotqa(answers, supporting_facts, load_hotpotqa(args.gold))
    payload = {
        **metrics.official_dict(),
        "count": metrics.count,
        "missing_answer_ids": list(metrics.missing_answer_ids),
        "missing_supporting_fact_ids": list(metrics.missing_supporting_fact_ids),
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
