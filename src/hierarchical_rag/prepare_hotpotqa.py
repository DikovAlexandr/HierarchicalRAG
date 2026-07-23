"""Create a deterministic HotpotQA subset and its provenance manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from hierarchical_rag.hotpotqa import (
    deterministic_slice,
    load_hotpotqa,
    sha256_file,
    slice_manifest,
    supporting_fact_reference_issues,
    write_hotpotqa,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a deterministic HotpotQA subset."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--source-url")
    parser.add_argument("--source-revision")
    parser.add_argument("--license")
    parser.add_argument("--retrieved-at")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    examples = load_hotpotqa(args.input)
    selected = deterministic_slice(examples, size=args.size, seed=args.seed)
    write_hotpotqa(args.output, selected)

    manifest = slice_manifest(args.input, selected, seed=args.seed)
    manifest["output_path"] = str(args.output)
    manifest["output_sha256"] = sha256_file(args.output)
    manifest["provenance"] = {
        key: value
        for key, value in {
            "source_url": args.source_url,
            "source_revision": args.source_revision,
            "license": args.license,
            "retrieved_at": args.retrieved_at,
        }.items()
        if value is not None
    }
    source_issues = supporting_fact_reference_issues(examples)
    selected_issues = supporting_fact_reference_issues(selected)
    manifest["annotation_validation"] = {
        "policy": (
            "Preserve official annotations; report out-of-range sentence references "
            "without silently editing or excluding examples."
        ),
        "source_issue_count": len(source_issues),
        "selected_issue_count": len(selected_issues),
        "source_issues": list(source_issues),
        "selected_issues": list(selected_issues),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    print(
        f"Prepared {len(selected)} examples: {args.output} "
        f"(sha256={manifest['output_sha256']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
