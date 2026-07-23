from __future__ import annotations

import json

import pytest

from hierarchical_rag.hotpotqa import (
    HotpotExample,
    deterministic_slice,
    gold_paragraphs,
    load_hotpotqa,
    serialize_context,
    slice_manifest,
    supporting_fact_reference_issues,
    write_hotpotqa,
)


def test_parse_official_record_and_gold_context(hotpot_records):
    example = HotpotExample.from_mapping(hotpot_records[0])

    assert example.identifier == "example-a"
    assert example.question_type == "bridge"
    assert [paragraph.title for paragraph in gold_paragraphs(example)] == [
        "France",
        "Paris",
    ]


def test_parse_huggingface_column_representation(hotpot_records):
    raw = hotpot_records[0]
    raw["id"] = raw.pop("_id")
    raw["context"] = {
        "title": [item[0] for item in raw["context"]],
        "sentences": [item[1] for item in raw["context"]],
    }
    raw["supporting_facts"] = {
        "title": [item[0] for item in raw["supporting_facts"]],
        "sent_id": [item[1] for item in raw["supporting_facts"]],
    }

    example = HotpotExample.from_mapping(raw)

    assert example.identifier == "example-a"
    assert example.supporting_facts[1].title == "Paris"


def test_reject_supporting_fact_outside_context(hotpot_records):
    hotpot_records[0]["supporting_facts"] = [["Missing", 0]]

    with pytest.raises(ValueError, match="supporting title"):
        HotpotExample.from_mapping(hotpot_records[0])


def test_fullwiki_mode_allows_gold_title_outside_provided_context(hotpot_records):
    hotpot_records[0]["supporting_facts"] = [["Missing", 0]]

    example = HotpotExample.from_mapping(
        hotpot_records[0], require_supporting_context=False
    )

    assert example.supporting_facts[0].title == "Missing"


def test_report_out_of_range_sentence_without_mutating_record(hotpot_records):
    hotpot_records[0]["supporting_facts"] = [["France", 99]]

    example = HotpotExample.from_mapping(hotpot_records[0])
    issues = supporting_fact_reference_issues([example])

    assert example.supporting_facts[0].sentence_id == 99
    assert issues[0]["example_id"] == "example-a"
    assert issues[0]["paragraph_sentence_count"] == 1


def test_deterministic_slice_is_independent_of_input_order(hotpot_records):
    examples = tuple(HotpotExample.from_mapping(record) for record in hotpot_records)

    first = deterministic_slice(examples, size=2, seed=42)
    second = deterministic_slice(reversed(examples), size=2, seed=42)

    assert [item.identifier for item in first] == [
        item.identifier for item in second
    ]


def test_serialization_is_stable_and_sentence_numbered(hotpot_records):
    example = HotpotExample.from_mapping(hotpot_records[0])

    serialized = serialize_context(example.question, gold_paragraphs(example))

    assert serialized.startswith(
        "Question: What city is the capital of France?\n\nEvidence:\n"
    )
    assert "[Document 1] France\n(0) France is a country in Europe." in serialized
    assert serialized.endswith("(0) Paris is the capital of France.\n")


def test_load_and_manifest_record_source_checksum(tmp_path, hotpot_records):
    source = tmp_path / "hotpot.json"
    source.write_text(json.dumps(hotpot_records), encoding="utf-8")

    examples = load_hotpotqa(source)
    selected = deterministic_slice(examples, size=2, seed=7)
    manifest = slice_manifest(source, selected, seed=7)

    assert len(examples) == 3
    assert manifest["size"] == 2
    assert len(manifest["source_sha256"]) == 64
    assert manifest["example_ids"] == [item.identifier for item in selected]


def test_write_round_trip_preserves_benchmark_fields(tmp_path, hotpot_records):
    examples = tuple(HotpotExample.from_mapping(record) for record in hotpot_records)
    destination = tmp_path / "subset.json"

    write_hotpotqa(destination, examples)
    loaded = load_hotpotqa(destination)

    assert loaded == examples
