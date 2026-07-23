from __future__ import annotations

import json

from hierarchical_rag.hotpotqa import load_hotpotqa
from hierarchical_rag.prepare_hotpotqa import main


def test_prepare_command_writes_reproducible_subset(tmp_path, hotpot_records):
    source = tmp_path / "source.json"
    output = tmp_path / "subset.json"
    manifest = tmp_path / "subset.manifest.json"
    source.write_text(json.dumps(hotpot_records), encoding="utf-8")

    exit_code = main(
        [
            "--input",
            str(source),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
            "--size",
            "2",
            "--seed",
            "42",
        ]
    )

    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert len(load_hotpotqa(output)) == 2
    assert metadata["size"] == 2
    assert metadata["output_sha256"]
