from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


def _normalized(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    m = np.max(x)
    if m <= 0:
        return np.zeros_like(x)
    return x / m


def miscoding_score(
    X: np.ndarray,
    y: np.ndarray,
    task: Literal["classification", "regression"],
    n_neighbors: int = 3,
    random_state: int | None = None,
) -> float:
    """Compute **μ (miscoding)** as lack of mutual information between X and y.

    We estimate mutual information per-feature and aggregate as the *deficit*:
        μ = 1 - mean( NMI(x_j; y) )  in  [0, 1]

    Notes
    -----
    - For classification we use `mutual_info_classif`.
    - For regression we use `mutual_info_regression`.
    - Features are assumed numeric; scale beforehand if needed.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    task : {"classification", "regression"}
    n_neighbors : int
        MI estimator parameter.
    random_state : int | None

    Returns
    -------
    float
        μ in [0, 1], higher means worse representation (more miscoding).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y 1D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must share the first dimension.")

    if task == "classification":
        mi = mutual_info_classif(
            X, y, discrete_features=False, n_neighbors=n_neighbors, random_state=random_state
        )
    elif task == "regression":
        mi = mutual_info_regression(X, y, n_neighbors=n_neighbors, random_state=random_state)
    else:
        raise ValueError("task must be 'classification' or 'regression'")

    nmi = _normalized(mi)
    mu = 1.0 - float(np.mean(nmi))
    return float(np.clip(mu, 0.0, 1.0))
