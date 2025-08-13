from __future__ import annotations

import math
from typing import Literal

from sklearn.base import BaseEstimator

from ..utils.compression import model_size_bytes


def surfeit_score(
    model: BaseEstimator,
    X_shape: tuple[int, int] | None,
    mode: Literal["bytes_per_sample", "bytes_log_n"] = "bytes_per_sample",
) -> float:
    """Compute **σ (surfeit)** as a normalized complexity proxy."""
    size = float(model_size_bytes(model))
    n = float(X_shape[0]) if X_shape is not None else 1.0
    if mode == "bytes_per_sample":
        return float(max(0.0, size / max(1.0, n)))
    if mode == "bytes_log_n":
        denom = max(1e-12, math.log2(n + 1.0))
        return float(max(0.0, size / denom))
    raise ValueError("Unknown mode")
