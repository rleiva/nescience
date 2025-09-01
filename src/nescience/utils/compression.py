"""
compression.py

Machine learning
with the minimum nescience principle

@author:  Rafael Garcia Leiva
@mail:    rgarcialeiva@gmail.com
@web:     http://www.mathematicsunknown.com/
@license: Apache v2
"""

from __future__ import annotations

import io
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

__all__ = ["discretize_vector", "unique_count", "optimal_code_length", "model_size_bytes"]

#
# Helpers (internal)
#

"""
Build equal-width bin edges over a trimmed core [q_alpha, q_{1-alpha}]
and add two overflow tails (-inf, +inf). If n_bins is None, start from
Rice's target; then (if m_min>0) reduce core bins until every core bin
has at least m_min samples (avoids empty core bins).

Parameters
----------
x : array-like, shape (n_samples,)
    The vector to be discretized
n_bins : int or None
    Target number of equal-width bins in the core region. If None, use Rice.
alpha : float
    Trim fraction per tail for the core range (e.g., 0.005 = 0.5%).
m_min : int
    Minimum occupancy per core bin (1 avoids empty core bins).    

Returns
-------
edges : np.ndarray
    Array of length (B_core + 2 + 1) with -inf and +inf tails included.
    #bins_total = len(edges) - 1.
"""


def _trimmed_equal_width_edges_with_tails(
    x: np.ndarray, n_bins: int | None, alpha: float = 0.005, m_min: int = 1
) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n == 0:
        return np.array([-np.inf, np.inf], dtype=float)

    # Target number of core bins
    if n_bins is not None and n_bins >= 1:
        B_core = int(n_bins)
    else:
        # Rice rule: B ≈ 2 * n^(1/3)
        B_core = int(max(1, n))
    B_core = max(1, B_core)

    # Robust core range
    a = float(np.quantile(x, alpha))
    b = float(np.quantile(x, 1 - alpha))
    if not np.isfinite(a) or not np.isfinite(b) or a >= b:
        a, b = float(np.min(x)), float(np.max(x))
        if a == b:  # degenerate
            a, b = a - 0.5, b + 0.5

    def build_edges(B: int) -> tuple[np.ndarray, np.ndarray]:
        core_edges = np.linspace(a, b, B + 1)
        edges = np.concatenate(([-np.inf], core_edges, [np.inf]))
        counts, _ = np.histogram(x, bins=edges)
        return edges, counts

    # Reduce resolution until core occupancy is OK (equal-width preserved in core)
    while True:
        edges, counts = build_edges(B_core)
        core_counts = counts[1:-1]  # exclude two tail bins
        if m_min <= 0 or B_core == 1 or not (core_counts < m_min).any():
            break
        B_core -= 1

    return edges


"""
Map each x to a bin index in {0, ..., #bins-1} given full edges
including tails (-inf, ..., +inf). Uses numpy digitize on core edges.

Parameters
----------
x : array-like, shape (n_samples,)
    The vector to be discretized
edgest : array
    Bin edget to be used for the discretization.

Returns
-------
discretized vector : np.ndarray
"""


def _digitize_with_edges(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    # edges: [-inf, e0, e1, ..., eB, +inf]  -> core edges are edges[1:-1]
    core_edges = edges[1:-1]
    idx_core = np.digitize(x, core_edges, right=False)  # 0..B
    # 0 maps to left tail, B maps to right tail — this is already correct
    return idx_core.astype(np.int64)


#
# Public API
#


def discretize_vector(
    x: np.ndarray,
    n_bins: int | None = None,
    alpha: float = 0.005,
    m_min: int = 1,
) -> np.ndarray:
    """
    Discretize a continuous variable with an equal-width strategy that is robust to outliers.

    - Build equal-width bins on a trimmed core [q_alpha, q_{1-alpha}] + two overflow tails.
    - If n_bins is None: start from Rice's rule; else use provided n_bins.
    - If m_min > 0: decrease core bin count until every core bin has >= m_min samples.

    Parameters
    ----------
    x : array-like, shape (n_samples,)
    n_bins : int or None
        Target number of equal-width bins in the core region. If None, use Rice.
    alpha : float
        Trim fraction per tail for the core range (e.g., 0.005 = 0.5%).
    m_min : int
        Minimum occupancy per core bin (1 avoids empty core bins).

    Returns
    -------
    new_x : np.ndarray of int64, shape (n_samples,)
        Discrete indices in {0, ..., B_total-1}, where B_total = B_core + 2 (tails).
    """
    edges = _trimmed_equal_width_edges_with_tails(x, n_bins=n_bins, alpha=alpha, m_min=m_min)
    idx = _digitize_with_edges(x, edges)
    return idx


def unique_count(
    x1: np.ndarray,
    numeric1: bool,
    x2: np.ndarray | None = None,
    numeric2: bool | None = None,
    x3: np.ndarray | None = None,
    numeric3: bool | None = None,
    n_bins: int | None = None,
    alpha: float = 0.005,
    m_min: int = 1,
) -> np.ndarray:
    """
    Count occurrences of a discretized 1D / 2D / 3D space (classification or regression).

    Parameters
    ----------
    x1, x2, x3: array-like, shape (n_samples)
    numeric1, numeric2, numeric3: if the variable is numeric or not
    b_bins: number of bins to be used during the discretization
    alpha: ?
    m_min : (int) Minimum occupancy per core bin.

    Returns
    -------
    count : np.ndarray (counts of the distinct joint symbols)
    """

    # Process first variable

    if not numeric1:
        le1 = LabelEncoder().fit(x1)
        x1d = le1.transform(x1).astype(np.int64)
        B1 = int(x1d.max()) + 1 if x1d.size > 0 else 0
    else:
        idx1 = discretize_vector(np.asarray(x1), n_bins=n_bins, alpha=alpha, m_min=m_min)
        x1d = idx1
        B1 = int(x1d.max()) + 1 if x1d.size > 0 else 0

    # Short circuit if only one variable
    if x2 is None:
        vals, counts = np.unique(x1d, return_counts=True)
        return counts

    # Process second variable

    if numeric2 is None:
        raise ValueError("numeric2 must be provided when x2 is given.")

    if not numeric2:
        le2 = LabelEncoder().fit(x2)
        x2d = le2.transform(x2).astype(np.int64)
        B2 = int(x2d.max()) + 1 if x2d.size > 0 else 0
    else:
        idx2 = discretize_vector(np.asarray(x2), n_bins=n_bins, alpha=alpha, m_min=m_min)
        x2d = idx2
        B2 = int(x2d.max()) + 1 if x2d.size > 0 else 0

    # Pair 2D with mixed-radix
    z = x1d + B1 * x2d

    # Optional third variable

    if x3 is not None:
        if numeric3 is None:
            raise ValueError("numeric3 must be provided when x3 is given.")

        if not numeric3:
            le3 = LabelEncoder().fit(x3)
            x3d = le3.transform(x3).astype(np.int64)
            # B3 = int(x3d.max()) + 1 if x3d.size > 0 else 0
        else:
            idx3 = discretize_vector(np.asarray(x3), n_bins=n_bins, alpha=alpha, m_min=m_min)
            x3d = idx3
            # B3 = int(x3d.max()) + 1 if x3d.size > 0 else 0

        # Pair 3D with mixed-radix
        z = x1d + B1 * (x2d + B2 * x3d)

    # Return compact counts for the observed symbols
    _, counts = np.unique(z, return_counts=True)

    return counts


def optimal_code_length(
    x1: np.ndarray,
    numeric1: bool,
    x2: np.ndarray | None = None,
    numeric2: bool | None = None,
    x3: np.ndarray | None = None,
    numeric3: bool | None = None,
    n_bins: int | None = None,
    alpha: float = 0.005,
    m_min: int = 1,
) -> float:
    """
    Compute the length of a list of features (1d or 2d)
    and / or a target variable (classification or regression)
    using an optimal code using the frequencies of the categorical variables
    or a discretized version of the continuous variables

    Parameters
    ----------
    x1, x2, x3: array-like, shape (n_samples)
    numeric1, numeric2, numeric3: if the variable is numeric or not
    b_bins: number of bins to be used during the discretization
    alpha: ?
    m_min : (int) Minimum occupancy per core bin.

    Returns
    -------
    Return the length of the encoded dataset (float)
    """

    counts = unique_count(
        x1=x1,
        numeric1=numeric1,
        x2=x2,
        numeric2=numeric2,
        x3=x3,
        numeric3=numeric3,
        n_bins=n_bins,
        alpha=alpha,
        m_min=m_min,
    )

    length = len(x1)  # assumes aligned samples across variables
    counts = counts.astype(float)

    # Apply Krichevsky–Trofimov (Jeffreys) smoothing to compute total lenght
    B = int(len(counts))
    p = (counts + 0.5) / (length + 0.5 * B)
    ldm = float(-np.sum(counts * np.log2(p)))

    return ldm


def model_size_bytes(model: Any) -> int:
    """Rudimentary proxy for model description length.

    Serializes the estimator with joblib and returns the length of the buffer in bytes.
    Works for most scikit-learn estimators.

    Parameters
    ----------
    model : Any
        A fitted scikit-learn estimator.

    Returns
    -------
    int
        Size in bytes of the serialized model.
    """
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return len(buf.getvalue())
