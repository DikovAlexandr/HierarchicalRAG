"""Build and validate the single-upload D031 fullwiki Notebook bundle."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


CONFIG_PATH = "experiments/configs/p2-qwen3-embedding-fullwiki-build-v1.yaml"
LOCK_PATH = "environments/hrm-text-gpu-py310.lock"
CORPUS_PATH = "data/raw/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2"
RUNTIME_INPUTS = (CORPUS_PATH,)
TEXT_SUFFIXES = {
    "",
    ".json",
    ".jsonl",
    ".lock",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
}


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


def _append_file(
    archive: tarfile.TarFile, relative_path: str, source: Path, mtime: int
) -> None:
    info = tarfile.TarInfo(f"HierarchicalRAG/{relative_path}")
    info.size = source.stat().st_size
    info.mode = 0o644
    info.mtime = mtime
    with source.open("rb") as stream:
        archive.addfile(info, stream)


def _member_bytes(archive: tarfile.TarFile, relative_path: str) -> bytes:
    member = archive.getmember(f"HierarchicalRAG/{relative_path}")
    extracted = archive.extractfile(member)
    if extracted is None:
        raise RuntimeError(f"bundle member is not a file: {member.name}")
    return extracted.read()


def _member_sha256(archive: tarfile.TarFile, relative_path: str) -> str:
    member = archive.getmember(f"HierarchicalRAG/{relative_path}")
    extracted = archive.extractfile(member)
    if extracted is None:
        raise RuntimeError(f"bundle member is not a file: {member.name}")
    digest = hashlib.sha256()
    for chunk in iter(lambda: extracted.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


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
        if _config_scalar(config, "source_file") != CORPUS_PATH:
            raise RuntimeError("config references an unexpected fullwiki corpus path")
        if _member_sha256(archive, CORPUS_PATH) != _config_scalar(
            config, "source_sha256"
        ):
            raise RuntimeError("bundled fullwiki corpus checksum differs from config")
        if _config_scalar(config, "dependency_lock_file") != LOCK_PATH:
            raise RuntimeError("config references an unexpected dependency lock")
        if _member_sha256(archive, LOCK_PATH) != _config_scalar(
            config, "dependency_lock_sha256"
        ):
            raise RuntimeError("bundled dependency lock checksum differs from config")
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
        output_path = repo / f"d031-notebook-bundle-{revision[:7]}.tar.gz"
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle: {output_path}")

    with tempfile.TemporaryDirectory(prefix="d031-bundle-") as temporary_dir:
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
                    raise FileNotFoundError(f"required D031 input is missing: {source}")
                _append_file(archive, relative_path, source, commit_time)
        with tar_path.open("rb") as source, output_path.open("wb") as destination:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=destination,
                compresslevel=1,
                mtime=commit_time,
            ) as compressed:
                shutil.copyfileobj(source, compressed, length=8 * 1024 * 1024)
    try:
        validate_bundle(output_path, revision)
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    digest = hashlib.sha256()
    with output_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    print(f"source_revision={revision}")
    print(f"bundle={output_path}")
    print(f"bundle_sha256={digest.hexdigest()}")
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
