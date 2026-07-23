from __future__ import annotations

import json

from hierarchical_rag.evaluate_bm25_candidates import main


def test_bm25_command_writes_metrics_and_rankings(tmp_path, hotpot_records):
    dataset = tmp_path / "subset.json"
    retrieval = tmp_path / "retrieval.jsonl"
    metrics = tmp_path / "metrics.json"
    dataset.write_text(json.dumps(hotpot_records[:1]), encoding="utf-8")

    exit_code = main(
        [
            "--dataset",
            str(dataset),
            "--retrieval-output",
            str(retrieval),
            "--metrics-output",
            str(metrics),
            "--top-k",
            "1",
            "2",
        ]
    )

    aggregate = json.loads(metrics.read_text(encoding="utf-8"))
    first_row = json.loads(retrieval.read_text(encoding="utf-8").splitlines()[0])
    assert exit_code == 0
    assert aggregate["paragraph_recall_at_2"] == 0.5
    assert first_row["retrieved"][0]["rank"] == 1
