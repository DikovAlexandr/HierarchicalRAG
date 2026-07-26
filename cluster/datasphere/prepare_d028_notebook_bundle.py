"""Build and validate the clean-commit DataSphere bundle for D028."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tempfile


CONFIG_PATH = "experiments/configs/p2-qwen3-embedding-fullwiki-calibration-v1.yaml"
LOCK_PATH = "environments/hrm-text-gpu-py310.lock"
SAMPLE_PATH = "data/interim/hotpotqa/qwen3-embedding-calibration-s8448.jsonl"
MANIFEST_PATH = "data/interim/hotpotqa/qwen3-embedding-calibration-s8448.manifest.json"
RUNTIME_INPUTS = (SAMPLE_PATH, MANIFEST_PATH)
TEXT_SUFFIXES = {"", ".json", ".jsonl", ".lock", ".md", ".py", ".sh", ".toml", ".yaml", ".yml"}


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def _require_clean_commit(repo: Path) -> str:
    dirty = _git(repo, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise RuntimeError("tracked worktree changes must be committed before bundling")
    return _git(repo, "rev-parse", "HEAD").decode("ascii").strip()


def _append_bytes(
    archive: tarfile.TarFile, relative_path: str, payload: bytes, mtime: int
) -> None:
    info = tarfile.TarInfo(f"HierarchicalRAG/{relative_path}")
    info.size = len(payload)
    info.mode = 0o644
    info.mtime = mtime
    archive.addfile(info, io.BytesIO(payload))


def _member_bytes(archive: tarfile.TarFile, relative_path: str) -> bytes:
    member = archive.getmember(f"HierarchicalRAG/{relative_path}")
    extracted = archive.extractfile(member)
    if extracted is None:
        raise RuntimeError(f"bundle member is not a file: {member.name}")
    return extracted.read()


def _config_scalar(config: bytes, key: str) -> str:
    match = re.search(
        rb"(?m)^\s*" + re.escape(key.encode("ascii")) + rb":\s*([^\s#]+)\s*$",
        config,
    )
    if match is None:
        raise RuntimeError(f"config does not define {key}")
    return match.group(1).decode("utf-8")


def validate_bundle(bundle_path: Path, revision: str) -> None:
    with tarfile.open(bundle_path, "r:gz") as archive:
        if _member_bytes(archive, "SOURCE_REVISION.txt") != f"{revision}\n".encode():
            raise RuntimeError("SOURCE_REVISION.txt differs from bundled commit")
        config = _member_bytes(archive, CONFIG_PATH)
        for path_key, checksum_key, expected_path in (
            ("source_file", "source_sha256", SAMPLE_PATH),
            ("manifest_file", "manifest_sha256", MANIFEST_PATH),
            ("dependency_lock_file", "dependency_lock_sha256", LOCK_PATH),
        ):
            if _config_scalar(config, path_key) != expected_path:
                raise RuntimeError(f"config references unexpected {path_key}")
            actual = hashlib.sha256(_member_bytes(archive, expected_path)).hexdigest()
            if actual != _config_scalar(config, checksum_key):
                raise RuntimeError(f"bundled {expected_path} checksum differs from config")
        crlf = []
        for member in archive.getmembers():
            if not member.isfile():
                continue
            path = PurePosixPath(member.name)
            if path.suffix.lower() not in TEXT_SUFFIXES and not path.name.startswith("."):
                continue
            extracted = archive.extractfile(member)
            if extracted is not None and b"\r\n" in extracted.read():
                crlf.append(member.name)
        if crlf:
            raise RuntimeError("bundle contains CRLF text members: " + ", ".join(crlf))


def build_bundle(repo: Path, output_path: Path | None = None) -> Path:
    revision = _require_clean_commit(repo)
    commit_time = int(_git(repo, "show", "-s", "--format=%ct", revision))
    if output_path is None:
        output_path = repo / f"d028-notebook-bundle-{revision[:7]}.tar.gz"
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle: {output_path}")

    with tempfile.TemporaryDirectory(prefix="d028-bundle-") as temporary_dir:
        tar_path = Path(temporary_dir) / "bundle.tar"
        subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repo.as_posix()}",
                "-C",
                str(repo),
                "archive",
                "--format=tar",
                "--prefix=HierarchicalRAG/",
                "--output",
                str(tar_path),
                revision,
            ],
            check=True,
        )
        with tarfile.open(tar_path, "a") as archive:
            _append_bytes(archive, "SOURCE_REVISION.txt", f"{revision}\n".encode(), commit_time)
            for relative_path in RUNTIME_INPUTS:
                source = repo / relative_path
                if not source.is_file():
                    raise FileNotFoundError(f"required D028 input is missing: {source}")
                _append_bytes(archive, relative_path, source.read_bytes(), commit_time)
        with tar_path.open("rb") as source, output_path.open("wb") as destination:
            with gzip.GzipFile(filename="", mode="wb", fileobj=destination, mtime=commit_time) as compressed:
                shutil.copyfileobj(source, compressed)
    try:
        validate_bundle(output_path, revision)
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    print(f"source_revision={revision}")
    print(f"bundle={output_path}")
    print(f"bundle_sha256={hashlib.sha256(output_path.read_bytes()).hexdigest()}")
    print(f"size_bytes={output_path.stat().st_size}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    build_bundle(Path(__file__).resolve().parents[2], args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
