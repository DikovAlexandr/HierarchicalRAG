"""Build and validate the clean-commit DataSphere bundle for D023."""

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


CONFIG_PATHS = (
    "experiments/configs/p1-lfm2.5-thinking-gold-train-budget-4096-v1.yaml",
    "experiments/configs/p1-qwen3.5-2b-thinking-gold-train-budget-4096-v1.yaml",
    "experiments/configs/p1-lfm2.5-thinking-gold-train-budget-8192-v1.yaml",
    "experiments/configs/p1-qwen3.5-2b-thinking-gold-train-budget-8192-v1.yaml",
)
LOCK_PATH = "environments/hrm-text-gpu-py310.lock"
DATA_PATHS = (
    "data/processed/hotpotqa/train-distractor-s42-n18.json",
    "data/processed/hotpotqa/train-distractor-s42-n18.manifest.json",
)
TEXT_SUFFIXES = {
    "",
    ".bib",
    ".csv",
    ".dockerignore",
    ".gitattributes",
    ".gitignore",
    ".json",
    ".jsonl",
    ".lock",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
CHECKSUM_RE = re.compile(rb"dependency_lock_sha256:\s*([0-9a-f]{64})")


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def _require_clean_commit(repo: Path) -> str:
    dirty = _git(repo, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise RuntimeError("tracked worktree changes must be committed before bundling")
    return _git(repo, "rev-parse", "HEAD").decode("ascii").strip()


def _append_bytes(
    archive: tarfile.TarFile,
    relative_path: str,
    payload: bytes,
    mtime: int,
) -> None:
    info = tarfile.TarInfo(f"HierarchicalRAG/{relative_path}")
    info.size = len(payload)
    info.mode = 0o644
    info.mtime = mtime
    archive.addfile(info, io.BytesIO(payload))


def _append_runtime_inputs(
    tar_path: Path,
    repo: Path,
    revision: str,
    mtime: int,
) -> None:
    with tarfile.open(tar_path, "a") as archive:
        _append_bytes(
            archive,
            "SOURCE_REVISION.txt",
            f"{revision}\n".encode("ascii"),
            mtime,
        )
        for relative_path in DATA_PATHS:
            source_path = repo / relative_path
            if not source_path.is_file():
                raise FileNotFoundError(f"required D023 input is missing: {source_path}")
            _append_bytes(archive, relative_path, source_path.read_bytes(), mtime)


def _gzip_tar(tar_path: Path, output_path: Path, mtime: int) -> None:
    with tar_path.open("rb") as source, output_path.open("wb") as destination:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=destination,
            mtime=mtime,
        ) as compressed:
            shutil.copyfileobj(source, compressed)


def _member_bytes(archive: tarfile.TarFile, relative_path: str) -> bytes:
    member = archive.getmember(f"HierarchicalRAG/{relative_path}")
    extracted = archive.extractfile(member)
    if extracted is None:
        raise RuntimeError(f"bundle member is not a file: {member.name}")
    return extracted.read()


def _config_scalar(config_bytes: bytes, key: str) -> str:
    match = re.search(
        rb"(?m)^\s*" + re.escape(key.encode("ascii")) + rb":\s*([^\s#]+)\s*$",
        config_bytes,
    )
    if match is None:
        raise RuntimeError(f"config does not define {key}")
    return match.group(1).decode("utf-8")


def validate_bundle(bundle_path: Path, revision: str) -> str:
    with tarfile.open(bundle_path, "r:gz") as archive:
        source_revision = _member_bytes(archive, "SOURCE_REVISION.txt")
        if source_revision != f"{revision}\n".encode("ascii"):
            raise RuntimeError("SOURCE_REVISION.txt does not match the bundled commit")

        lock_bytes = _member_bytes(archive, LOCK_PATH)
        lock_sha256 = hashlib.sha256(lock_bytes).hexdigest()
        for config_path in CONFIG_PATHS:
            config_bytes = _member_bytes(archive, config_path)
            match = CHECKSUM_RE.search(config_bytes)
            if match is None or match.group(1).decode("ascii") != lock_sha256:
                raise RuntimeError(
                    f"{config_path} does not pin bundled lock SHA256 {lock_sha256}"
                )
            for path_key, checksum_key in (
                ("source_file", "source_sha256"),
                ("manifest_file", "manifest_sha256"),
            ):
                data_path = _config_scalar(config_bytes, path_key)
                if data_path not in DATA_PATHS:
                    raise RuntimeError(
                        f"{config_path} references unexpected D023 input {data_path}"
                    )
                data_sha256 = hashlib.sha256(
                    _member_bytes(archive, data_path)
                ).hexdigest()
                expected_sha256 = _config_scalar(config_bytes, checksum_key)
                if data_sha256 != expected_sha256:
                    raise RuntimeError(
                        f"{config_path} expects {data_path} SHA256 "
                        f"{expected_sha256}, bundled file has {data_sha256}"
                    )

        crlf_members = []
        for member in archive.getmembers():
            if not member.isfile():
                continue
            suffix = PurePosixPath(member.name).suffix.lower()
            name = PurePosixPath(member.name).name
            if suffix not in TEXT_SUFFIXES and not name.startswith("."):
                continue
            extracted = archive.extractfile(member)
            if extracted is not None and b"\r\n" in extracted.read():
                crlf_members.append(member.name)
        if crlf_members:
            raise RuntimeError(
                "bundle contains CRLF text members: " + ", ".join(crlf_members)
            )
    return lock_sha256


def build_bundle(repo: Path, output_path: Path | None = None) -> Path:
    revision = _require_clean_commit(repo)
    commit_time = int(_git(repo, "show", "-s", "--format=%ct", revision))
    if output_path is None:
        output_path = repo / f"d023-notebook-bundle-{revision[:7]}.tar.gz"
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle: {output_path}")

    with tempfile.TemporaryDirectory(prefix="d023-bundle-") as temporary_dir:
        tar_path = Path(temporary_dir) / "bundle.tar"
        subprocess.run(
            [
                "git",
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
        _append_runtime_inputs(tar_path, repo, revision, commit_time)
        _gzip_tar(tar_path, output_path, commit_time)

    try:
        lock_sha256 = validate_bundle(output_path, revision)
    except Exception:
        output_path.unlink(missing_ok=True)
        raise

    bundle_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    print(f"source_revision={revision}")
    print(f"dependency_lock_sha256={lock_sha256}")
    print(f"bundle={output_path}")
    print(f"bundle_sha256={bundle_sha256}")
    print(f"size_bytes={output_path.stat().st_size}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="output path; defaults to d023-notebook-bundle-<revision>.tar.gz",
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    build_bundle(repo, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
