from __future__ import annotations

import json

from hierarchical_rag.hotpotqa import HotpotExample, deterministic_slice, load_hotpotqa
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


def test_prepare_command_selects_globally_across_multiple_inputs(
    tmp_path, hotpot_records
):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    output = tmp_path / "subset.json"
    manifest = tmp_path / "subset.manifest.json"
    first.write_text(json.dumps(hotpot_records[:2]), encoding="utf-8")
    second.write_text(json.dumps(hotpot_records[2:]), encoding="utf-8")

    exit_code = main(
        [
            "--input",
            str(first),
            str(second),
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
    expected_ids = [
        example.identifier
        for example in deterministic_slice(
            tuple(HotpotExample.from_mapping(record) for record in hotpot_records),
            size=2,
            seed=42,
        )
    ]
    assert exit_code == 0
    assert [example.identifier for example in load_hotpotqa(output)] == expected_ids
    assert metadata["source_count"] == 3
    assert [item["example_count"] for item in metadata["source_files"]] == [2, 1]
