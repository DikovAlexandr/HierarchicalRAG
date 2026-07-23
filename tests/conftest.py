from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture
def hotpot_records() -> list[dict[str, Any]]:
    return [
        {
            "_id": "example-a",
            "question": "What city is the capital of France?",
            "answer": "Paris",
            "type": "bridge",
            "level": "hard",
            "supporting_facts": [["France", 0], ["Paris", 0]],
            "context": [
                ["France", ["France is a country in Europe."]],
                ["Paris", ["Paris is the capital of France."]],
                ["Berlin", ["Berlin is the capital of Germany."]],
            ],
        },
        {
            "_id": "example-b",
            "question": "Is the Nile longer than the Thames?",
            "answer": "yes",
            "type": "comparison",
            "level": "hard",
            "supporting_facts": [["Nile", 0], ["Thames", 0]],
            "context": [
                ["Nile", ["The Nile is approximately 6,650 km long."]],
                ["Thames", ["The Thames is approximately 346 km long."]],
                ["Danube", ["The Danube flows through Europe."]],
            ],
        },
        {
            "_id": "example-c",
            "question": "Which planet is known as the Red Planet?",
            "answer": "Mars",
            "type": "bridge",
            "level": "medium",
            "supporting_facts": [["Mars", 0]],
            "context": [
                ["Mars", ["Mars is often called the Red Planet."]],
                ["Venus", ["Venus has a dense atmosphere."]],
            ],
        },
    ]
