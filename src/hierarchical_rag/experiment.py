"""Immutable run-directory and manifest helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


REQUIRED_RUN_FILES = (
    "manifest.json",
    "resolved-config.yaml",
    "command.txt",
    "environment.txt",
    "predictions.jsonl",
    "retrieval.jsonl",
    "metrics.json",
    "statistics.json",
    "run.log",
)


def prepare_run_directory(path: str | Path) -> Path:
    """Create an empty run directory without overwriting evidence."""

    destination = Path(path)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"run directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def build_manifest(
    *,
    experiment_id: str,
    owner: str,
    command: str,
    git_commit: str,
    config_path: str | Path,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = Path(config_path)
    manifest: dict[str, Any] = {
        "experiment_id": experiment_id,
        "owner": owner,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "git_commit": git_commit,
        "config_path": str(config),
        "resolved_config_sha256": sha256_file(config),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "process_id": os.getpid(),
        "required_run_files": list(REQUIRED_RUN_FILES),
    }
    if extra:
        manifest["extra"] = dict(extra)
    return manifest


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    temporary.replace(destination)


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load a versioned YAML config and reject unresolved placeholders."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("experiment config must be a mapping")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported or missing schema_version")
    for section in ("experiment", "dataset", "evaluation", "runtime"):
        if not isinstance(payload.get(section), dict):
            raise ValueError(f"config section {section!r} must be a mapping")
    placeholders = tuple(_find_placeholders(payload))
    if placeholders:
        raise ValueError(
            "unresolved config placeholders: " + ", ".join(placeholders)
        )
    return payload


def find_repository_root(start: str | Path) -> Path:
    candidate = Path(start).resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if (directory / ".git").exists():
            return directory
    raise ValueError(f"cannot find repository root from {start}")


def git_revision(repository_root: str | Path) -> str:
    return _git(repository_root, "rev-parse", "HEAD").strip()


def git_status(repository_root: str | Path) -> str:
    return _git(
        repository_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).strip()


def environment_snapshot() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for package in ("hierarchical-rag", "PyYAML"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_identifier": os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "logical_cpu_count": os.cpu_count(),
        "hostname": platform.node(),
        "packages": packages,
    }


def file_inventory(
    directory: str | Path, *, exclude: Iterable[str] = ()
) -> list[dict[str, Any]]:
    root = Path(directory)
    excluded = set(exclude)
    return [
        {
            "path": str(path.relative_to(root)).replace("\\", "/"),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and str(path.relative_to(root)).replace("\\", "/") not in excluded
    ]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_placeholders(value: Any, path: str = "") -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            yield from _find_placeholders(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _find_placeholders(item, f"{path}[{index}]")
    elif isinstance(value, str) and value.startswith("TO_"):
        yield path


def _git(repository_root: str | Path, *arguments: str) -> str:
    root = Path(repository_root).resolve()
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={root.as_posix()}", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout
