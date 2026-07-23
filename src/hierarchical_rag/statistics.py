"""Predeclared paired statistics for model comparisons."""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from statistics import fmean, stdev
from typing import Sequence


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    estimate: float
    low: float
    high: float
    confidence_level: float
    method: str
    resamples: int


@dataclass(frozen=True, slots=True)
class PairedTest:
    observed_difference: float
    p_value: float
    method: str
    permutations: int


def paired_bootstrap_difference(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    confidence_level: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> ConfidenceInterval:
    differences = _paired_differences(baseline, candidate)
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    if resamples < 1:
        raise ValueError("resamples must be positive")

    rng = random.Random(seed)
    sample_size = len(differences)
    estimates = sorted(
        fmean(differences[rng.randrange(sample_size)] for _ in range(sample_size))
        for _ in range(resamples)
    )
    alpha = 1 - confidence_level
    return ConfidenceInterval(
        estimate=fmean(differences),
        low=_quantile(estimates, alpha / 2),
        high=_quantile(estimates, 1 - alpha / 2),
        confidence_level=confidence_level,
        method="paired_bootstrap",
        resamples=resamples,
    )


def paired_permutation_test(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    permutations: int = 10_000,
    seed: int = 0,
) -> PairedTest:
    differences = _paired_differences(baseline, candidate)
    observed = fmean(differences)
    if permutations < 1:
        raise ValueError("permutations must be positive")

    if len(differences) <= 16:
        permuted = (
            fmean(sign * value for sign, value in zip(signs, differences, strict=True))
            for signs in itertools.product((-1, 1), repeat=len(differences))
        )
        total = 2 ** len(differences)
        extreme = sum(abs(value) >= abs(observed) for value in permuted)
        p_value = extreme / total
        method = "exact_paired_sign_flip"
    else:
        rng = random.Random(seed)
        extreme = 0
        for _ in range(permutations):
            value = fmean(
                rng.choice((-1, 1)) * difference for difference in differences
            )
            extreme += abs(value) >= abs(observed)
        total = permutations
        p_value = (extreme + 1) / (total + 1)
        method = "monte_carlo_paired_sign_flip"

    return PairedTest(
        observed_difference=observed,
        p_value=p_value,
        method=method,
        permutations=total,
    )


def paired_cohens_dz(
    baseline: Sequence[float], candidate: Sequence[float]
) -> float:
    differences = _paired_differences(baseline, candidate)
    if len(differences) < 2:
        raise ValueError("at least two pairs are required")
    dispersion = stdev(differences)
    estimate = fmean(differences)
    if dispersion == 0:
        return 0.0 if estimate == 0 else math.copysign(math.inf, estimate)
    return estimate / dispersion


def _paired_differences(
    baseline: Sequence[float], candidate: Sequence[float]
) -> tuple[float, ...]:
    if len(baseline) != len(candidate):
        raise ValueError("paired samples must have equal lengths")
    if not baseline:
        raise ValueError("paired samples cannot be empty")
    values = tuple(
        float(candidate_value) - float(baseline_value)
        for baseline_value, candidate_value in zip(baseline, candidate, strict=True)
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("paired samples must contain finite values")
    return values


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
