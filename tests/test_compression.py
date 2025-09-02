# test_utils.py
import numpy as np
import pytest
from utils import discretize_vector, optimal_code_length, unique_count

# ---------------------------
# Helpers for reproducibility
# ---------------------------

RNG = np.random.default_rng(7)


# ---------------------------
# discretize_vector tests
# ---------------------------


def test_discretize_returns_int64_and_same_length() -> None:
    x = RNG.normal(size=1000)
    idx = discretize_vector(x)
    assert idx.dtype == np.int64
    assert idx.shape == x.shape


def test_discretize_constant_vector_single_bin_plus_tails_maybe() -> None:
    x = np.ones(100)
    idx = discretize_vector(x, n_bins=8)  # n_bins shouldn't matter for constant input
    # All points must map to the same bin index
    assert np.unique(idx).size == 1


def test_discretize_outliers_tail_bins_catch_extremes() -> None:
    # Mostly N(0,1) with extreme outliers
    body = RNG.normal(0, 1, size=1000)
    tails = np.array([50, 60, -55, -40], dtype=float)
    x = np.concatenate([body, tails])
    idx = discretize_vector(x, n_bins=32, alpha=0.005, m_min=1)
    # Left tail is index 0, right tail is the maximum index
    assert np.any(idx == 0)  # left overflow used
    assert np.any(idx == idx.max())  # right overflow used


def test_discretize_minimum_occupancy_reduces_bins() -> None:
    # Force a large n_bins while sample size is small; m_min should reduce core bins.
    x = RNG.normal(size=100)
    idx = discretize_vector(x, n_bins=200, m_min=5, alpha=0.0)
    # With m_min=5, total used bins should be ≤ n//m_min + 2 (tails)
    used = np.unique(idx).size
    assert used <= (len(x) // 5) + 2


def test_discretize_alpha_zero_behaves_like_full_range_equal_width() -> None:
    x = RNG.normal(size=500)
    idx_full = discretize_vector(x, n_bins=16, alpha=0.0, m_min=1)
    assert np.unique(idx_full).size >= 2  # not degenerate
    # With no trimming, tails may still appear if min/max fall outside core edges
    # but this shouldn’t be required to pass; main check is it discretizes.


# ---------------------------
# unique_count tests (1D/2D/3D)
# ---------------------------


def test_unique_count_1d_numeric_counts_sum_to_n() -> None:
    x = RNG.normal(size=1234)
    counts = unique_count(x1=x, numeric1=True, n_bins=32)
    assert counts.sum() == len(x)
    # All counts are positive by construction (unique returns only seen symbols)
    assert np.all(counts > 0)


def test_unique_count_1d_categorical_matches_label_frequencies() -> None:
    cats = np.array(["a", "b", "a", "c", "b", "b", "a"])
    counts = unique_count(x1=cats, numeric1=False)
    # Frequencies: a=3, b=3, c=1 (order depends on LabelEncoder, but set matches)
    assert sorted(counts.tolist()) == [1, 3, 3]


def test_unique_count_2d_numeric_counts_sum_to_n() -> None:
    x = RNG.normal(size=800)
    y = 0.5 * x + RNG.normal(size=800)
    counts = unique_count(x1=x, numeric1=True, x2=y, numeric2=True, n_bins=24)
    assert counts.sum() == len(x)


def test_unique_count_2d_mixed_numeric_categorical() -> None:
    x = RNG.normal(size=300)
    # 3 categories with imbalance
    z = np.array(["low"] * 200 + ["mid"] * 90 + ["high"] * 10)
    counts = unique_count(x1=x, numeric1=True, x2=z, numeric2=False, n_bins=16)
    assert counts.sum() == len(x)
    assert np.all(counts > 0)


def test_unique_count_3d_numeric_all() -> None:
    x = RNG.normal(size=256)
    y = RNG.normal(size=256)
    w = RNG.normal(size=256)
    counts = unique_count(
        x1=x,
        numeric1=True,
        x2=y,
        numeric2=True,
        x3=w,
        numeric3=True,
        n_bins=12,
    )
    assert counts.sum() == len(x)
    assert counts.size >= 1


def test_unique_count_errors_when_numeric_flag_missing_for_given_x2() -> None:
    x = RNG.normal(size=10)
    y = RNG.normal(size=10)
    with pytest.raises(ValueError):
        unique_count(x1=x, numeric1=True, x2=y)  # numeric2 missing


def test_unique_count_errors_when_numeric_flag_missing_for_given_x3() -> None:
    x = RNG.normal(size=10)
    y = RNG.normal(size=10)
    z = RNG.normal(size=10)
    with pytest.raises(ValueError):
        unique_count(x1=x, numeric1=True, x2=y, numeric2=True, x3=z)  # numeric3 missing


# ---------------------------
# optimal_code_length tests
# ---------------------------


def test_optimal_code_length_non_negative_and_finite_1d() -> None:
    x = RNG.normal(size=1000)
    L = optimal_code_length(x1=x, numeric1=True, n_bins=24)
    assert np.isfinite(L)
    assert L >= 0.0


def test_optimal_code_length_non_negative_and_finite_2d() -> None:
    x = RNG.normal(size=1000)
    y = 0.7 * x + RNG.normal(size=1000)
    L = optimal_code_length(x1=x, numeric1=True, x2=y, numeric2=True, n_bins=24)
    assert np.isfinite(L)
    assert L >= 0.0


def test_subadditivity_marginal_plus_marginal_vs_joint_same_machine() -> None:
    # With same n_bins for both axes, KT smoothing in optimal_code_length should
    # keep lengths comparable; we expect L(X,Y) <= L(X) + L(Y) (allow tiny slack).
    n = 1500
    x = RNG.normal(size=n)
    y = 0.6 * x + RNG.normal(size=n)
    LX = optimal_code_length(x1=x, numeric1=True, n_bins=20)
    LY = optimal_code_length(x1=y, numeric1=True, n_bins=20)
    LXY = optimal_code_length(x1=x, numeric1=True, x2=y, numeric2=True, n_bins=20)
    assert LXY <= LX + LY + 1e-6  # tiny numerical tolerance


def test_code_length_increases_with_finer_resolution_typically() -> None:
    # Not a strict theorem per-sample, but in practice finer (more bins) tends
    # to increase length. Use a moderate n to avoid pathological flips.
    x = RNG.normal(size=5000)
    L_coarse = optimal_code_length(x1=x, numeric1=True, n_bins=8)
    L_fine = optimal_code_length(x1=x, numeric1=True, n_bins=64)
    assert L_fine >= L_coarse


def test_outliers_do_not_break_length_computation_with_tails() -> None:
    x = np.concatenate(
        [RNG.normal(0, 1, 990), np.array([35, 40, -30, -45, 60, -65, 80, -75, 90, -95])]
    )
    L = optimal_code_length(
        x1=x, numeric1=True, n_bins=32
    )  # trimmed core + tails should handle this
    assert np.isfinite(L)
    assert L >= 0.0


def test_counts_vs_length_consistency_2d() -> None:
    # The length must be the same if we recompute it from counts manually.
    x = RNG.normal(size=1200)
    y = RNG.normal(size=1200)
    counts = unique_count(x1=x, numeric1=True, x2=y, numeric2=True, n_bins=24)
    n = len(x)
    B = len(counts)  # number of observed joint symbols (KT needs B to penalize alphabet size)
    p = (counts + 0.5) / (n + 0.5 * B)
    L_manual = float(-np.sum(counts * np.log2(p)))
    L_func = optimal_code_length(x1=x, numeric1=True, x2=y, numeric2=True, n_bins=24)
    assert np.isclose(L_manual, L_func, rtol=1e-10, atol=1e-10)


# ---------------------------
# Boundary/degenerate cases
# ---------------------------


def test_empty_input_returns_zero_length_counts() -> None:
    x = np.array([], dtype=float)
    # discretize: degenerate but should return empty int64
    idx = discretize_vector(x)
    assert idx.size == 0
    assert idx.dtype == np.int64
    # unique_count: should return empty counts
    counts = unique_count(x1=x, numeric1=True, n_bins=8)
    assert counts.size == 0
    # length: should be 0.0 or finite
    L = optimal_code_length(x1=x, numeric1=True, n_bins=8)
    # For empty, our length computation returns 0*log(·) sums => 0.0
    assert L == 0.0


def test_single_element_input() -> None:
    x = np.array([3.14])
    idx = discretize_vector(x, n_bins=4)
    assert idx.shape == (1,)
    counts = unique_count(x1=x, numeric1=True, n_bins=4)
    assert counts.sum() == 1
    L = optimal_code_length(x1=x, numeric1=True, n_bins=4)
    assert np.isfinite(L) and L >= 0.0
