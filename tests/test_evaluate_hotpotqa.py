from __future__ import annotations

import json

from hierarchical_rag.evaluate_hotpotqa import main


def test_evaluate_command_writes_official_fields(tmp_path, hotpot_records):
    gold = tmp_path / "gold.json"
    predictions = tmp_path / "predictions.json"
    output = tmp_path / "metrics.json"
    gold.write_text(json.dumps(hotpot_records[:1]), encoding="utf-8")
    predictions.write_text(
        json.dumps(
            {
                "answer": {"example-a": "Paris"},
                "sp": {"example-a": [["France", 0], ["Paris", 0]]},
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--predictions",
            str(predictions),
            "--gold",
            str(gold),
            "--output",
            str(output),
        ]
    )

    metrics = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert metrics["joint_em"] == 1.0
    assert metrics["count"] == 1
