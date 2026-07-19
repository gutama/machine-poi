"""Tests for src/transport_stats.py."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transport_stats import bootstrap_ci, paired_test, sign_permutation_test


def test_bootstrap_ci_single_sample_is_a_point():
    mean, lo, hi = bootstrap_ci([0.5])
    assert mean == lo == hi == 0.5


def test_bootstrap_ci_contains_true_mean_for_consistent_diffs():
    diffs = [0.10, 0.12, 0.09, 0.11, 0.13, 0.08]
    mean, lo, hi = bootstrap_ci(diffs, n_boot=5000, seed=1)
    assert lo <= mean <= hi
    # All diffs are close together and positive; the CI should not cross 0.
    assert lo > 0


def test_bootstrap_ci_wide_for_noisy_diffs():
    diffs = [0.5, -0.5, 0.4, -0.6, 0.55, -0.45]
    mean, lo, hi = bootstrap_ci(diffs, n_boot=5000, seed=1)
    # Near-zero mean with high variance -> interval should cross zero.
    assert lo < 0 < hi


def test_sign_permutation_exact_for_consistent_direction():
    # All 6 diffs point the same way -> only 1 of 2^6=64 sign patterns
    # (all +) matches or exceeds the observed statistic in this direction,
    # but since the test is on |mean|, the all -1 pattern ties it too.
    diffs = [0.10, 0.12, 0.09, 0.11, 0.13, 0.08]
    p, exact = sign_permutation_test(diffs)
    assert exact is True
    assert p <= 2 / 64  # only the (all +) and (all -) patterns can match
    assert p > 0


def test_sign_permutation_high_p_for_pure_noise():
    # Symmetric diffs around zero with random-looking signs -> should not
    # look significant.
    diffs = [0.5, -0.5, 0.4, -0.4, 0.3, -0.3, 0.2, -0.2]
    p, exact = sign_permutation_test(diffs)
    assert exact is True
    assert p > 0.5  # the observed |mean| (~0) is typical under the null


def test_sign_permutation_monte_carlo_path_for_large_n():
    diffs = [0.1 + 0.001 * i for i in range(25)]  # n=25 > exact_limit
    p, exact = sign_permutation_test(diffs, n_perm=2000, seed=0)
    assert exact is False
    assert 0.0 <= p <= 1.0


def test_paired_test_end_to_end():
    diffs = [0.10, 0.12, 0.09, 0.11, 0.13, 0.08]
    result = paired_test(diffs, n_boot=2000, n_perm=2000, seed=0)
    assert result.n == 6
    assert math.isclose(result.mean_diff, sum(diffs) / len(diffs))
    assert result.ci_low <= result.mean_diff <= result.ci_high
    assert result.p_value_exact is True
    d = result.to_dict()
    assert set(d) == {
        "n", "mean_diff", "ci_low", "ci_high", "ci_level",
        "p_value", "p_value_exact",
    }


def test_paired_test_filters_nan_and_none():
    diffs = [0.1, float("nan"), 0.2, None, 0.15]
    result = paired_test(diffs, n_boot=500, n_perm=500, seed=0)
    assert result.n == 3
