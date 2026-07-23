from __future__ import annotations

from pathlib import Path

from hierarchical_rag.check_environment import inspect_environment


def test_host_environment_report_is_traceable():
    report = inspect_environment("host", Path.cwd())

    assert report["passed"] is True
    assert report["checks"]["fts5_available"] is True
    assert report["repository_revision"]
    assert report["sqlite_version"]
