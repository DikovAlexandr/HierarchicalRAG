"""Validate a local or cluster execution environment before experiments."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sqlite3
import sys
from pathlib import Path
from typing import Any, Sequence

from hierarchical_rag.experiment import find_repository_root, git_revision
from hierarchical_rag.fullwiki import fts5_available


EVAL_CPU_PACKAGES = {
    "hierarchical-rag": "0.1.0",
    "PyYAML": "6.0.3",
    "pyarrow": "22.0.0",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("host", "eval-cpu"), required=True)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = inspect_environment(args.profile, args.repository_root)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8", newline="\n")
    return 0 if report["passed"] else 1


def inspect_environment(
    profile: str, repository_root: Path | None = None
) -> dict[str, Any]:
    packages = {
        package: _package_version(package)
        for package in sorted(EVAL_CPU_PACKAGES)
    }
    image_revision = os.environ.get("HIERARCHICAL_RAG_IMAGE_REVISION")
    checks: dict[str, bool] = {"fts5_available": fts5_available()}
    repository_revision: str | None = None
    if repository_root is not None:
        root = repository_root.resolve()
        repository_revision = git_revision(root)
    else:
        try:
            root = find_repository_root(Path.cwd())
            repository_revision = git_revision(root)
        except ValueError:
            repository_revision = None

    if profile == "eval-cpu":
        checks.update(
            {
                "linux": platform.system() == "Linux",
                "python_3_11": sys.version_info[:2] == (3, 11),
                "packages_pinned": packages == EVAL_CPU_PACKAGES,
                "image_revision_present": bool(
                    image_revision and image_revision != "unversioned"
                ),
                "image_matches_repository": bool(
                    repository_revision
                    and image_revision
                    and image_revision == repository_revision
                ),
            }
        )

    return {
        "profile": profile,
        "passed": all(checks.values()),
        "checks": checks,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "sqlite_version": sqlite3.sqlite_version,
        "packages": packages,
        "image_revision": image_revision,
        "repository_revision": repository_revision,
    }


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


if __name__ == "__main__":
    raise SystemExit(main())
