from __future__ import annotations

import math

import pytest

from hierarchical_rag.statistics import (
    paired_bootstrap_difference,
    paired_cohens_dz,
    paired_permutation_test,
)


def test_paired_bootstrap_is_deterministic():
    baseline = [0.0, 0.0, 1.0, 1.0]
    candidate = [0.0, 1.0, 1.0, 1.0]

    first = paired_bootstrap_difference(
        baseline, candidate, resamples=500, seed=17
    )
    second = paired_bootstrap_difference(
        baseline, candidate, resamples=500, seed=17
    )

    assert first == second
    assert first.estimate == 0.25
    assert first.low <= first.estimate <= first.high


def test_exact_paired_permutation_detects_consistent_gain():
    baseline = [0.0] * 10
    candidate = [1.0] * 10

    result = paired_permutation_test(baseline, candidate)

    assert result.method == "exact_paired_sign_flip"
    assert result.observed_difference == 1.0
    assert result.p_value == pytest.approx(2 / 1024)


def test_paired_cohens_dz_and_input_validation():
    assert math.isinf(paired_cohens_dz([0.0, 0.0], [1.0, 1.0]))

    with pytest.raises(ValueError, match="equal lengths"):
        paired_permutation_test([0.0], [0.0, 1.0])
