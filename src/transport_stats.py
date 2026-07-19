"""
Paired significance testing for steered-vs-baseline transport diagnostics.

The transport experiments (experiments/steered_vs_baseline_transport.py,
experiments/centered_contrast_probe.py) compare a metric (non-abelian ratio
rho, holonomy) between a steered and a baseline run of the SAME prompt. That
is a paired design: each prompt contributes one steered-minus-baseline
difference. This module turns a list of per-prompt paired differences into:

  - a bootstrap confidence interval on the mean difference (resampling
    prompts with replacement -- valid without assuming normality, and small
    n is handled honestly by simply producing a wide interval), and
  - a sign-flip permutation test p-value for the null hypothesis that the
    steered and baseline conditions are exchangeable (i.e. the sign of each
    prompt's difference is a coin flip). This is the paired-design analogue
    of a permutation test and needs no distributional assumptions, which
    matters here: rho and holonomy are bounded, non-negative-denominator
    ratios, not obviously Gaussian.

Both are intentionally simple (no scipy dependency) so they run anywhere
torch does.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from itertools import product
from typing import Sequence


@dataclass
class PairedTestResult:
    n: int
    mean_diff: float
    ci_low: float
    ci_high: float
    ci_level: float
    p_value: float
    p_value_exact: bool

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "mean_diff": self.mean_diff,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_level": self.ci_level,
            "p_value": self.p_value,
            "p_value_exact": self.p_value_exact,
        }


def bootstrap_ci(
    diffs: Sequence[float],
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple:
    """
    Percentile bootstrap CI on the mean of `diffs`, resampling prompts with
    replacement. Returns (mean, lo, hi). With n=1 the interval collapses to
    a point (there is nothing to resample) -- callers should treat n=1 as
    "no interval available", not as a tight/confident result.
    """
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    if not 0 < ci < 1:
        raise ValueError(f"ci must be in (0, 1), got {ci}")
    diffs = list(diffs)
    n = len(diffs)
    mean = sum(diffs) / n if n else float("nan")
    if n <= 1:
        return mean, mean, mean
    rng = random.Random(seed)
    boot_means = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    alpha = 1 - ci
    lo_idx = max(0, int((alpha / 2) * n_boot))
    hi_idx = min(n_boot - 1, int((1 - alpha / 2) * n_boot))
    return mean, boot_means[lo_idx], boot_means[hi_idx]


def sign_permutation_test(
    diffs: Sequence[float],
    n_perm: int = 20000,
    seed: int = 0,
    exact_limit: int = 20,
) -> tuple:
    """
    Two-sided paired sign-flip permutation test.

    Null hypothesis: each prompt's steered-vs-baseline sign is a fair coin
    flip (no systematic direction). Statistic: |mean(diffs)|. For n <=
    exact_limit, enumerates all 2^n sign patterns exactly; otherwise draws
    n_perm random sign patterns (Monte Carlo).

    Returns (p_value, is_exact). Requires at least one non-zero diff to be
    meaningful; an all-zero input returns p=1.0.
    """
    diffs = list(diffs)
    n = len(diffs)
    if n == 0:
        return float("nan"), True
    abs_diffs = [abs(d) for d in diffs]
    observed = abs(sum(diffs) / n)

    if n <= exact_limit:
        count = 0
        total = 0
        for signs in product([1, -1], repeat=n):
            total += 1
            stat = abs(sum(s * a for s, a in zip(signs, abs_diffs)) / n)
            if stat >= observed - 1e-12:
                count += 1
        return count / total, True

    if n_perm < 1:
        raise ValueError(f"n_perm must be >= 1, got {n_perm}")
    rng = random.Random(seed)
    count = 0
    for _ in range(n_perm):
        stat = abs(
            sum((1 if rng.random() < 0.5 else -1) * a for a in abs_diffs) / n
        )
        if stat >= observed - 1e-12:
            count += 1
    # +1 smoothing: a Monte Carlo p-value can never legitimately be 0 --
    # the observed statistic itself is always an admissible permutation.
    return (count + 1) / (n_perm + 1), False


def paired_test(
    diffs: Sequence[float],
    n_boot: int = 10000,
    n_perm: int = 20000,
    ci: float = 0.95,
    seed: int = 0,
) -> PairedTestResult:
    """Convenience wrapper: bootstrap CI + sign-permutation p-value in one call."""
    diffs = [d for d in diffs if d is not None and not (isinstance(d, float) and math.isnan(d))]
    n = len(diffs)
    mean, lo, hi = bootstrap_ci(diffs, n_boot=n_boot, ci=ci, seed=seed)
    p, exact = sign_permutation_test(diffs, n_perm=n_perm, seed=seed)
    return PairedTestResult(
        n=n, mean_diff=mean, ci_low=lo, ci_high=hi, ci_level=ci,
        p_value=p, p_value_exact=exact,
    )
