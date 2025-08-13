from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class HoldoutSpec:
    test_size: float = 0.2
    stratify: bool = True
    random_state: int | None = None


def holdout_split(
    X: np.ndarray,
    y: np.ndarray,
    task: Literal["classification", "regression"],
    spec: HoldoutSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    strat = y if (task == "classification" and spec.stratify) else None
    res = train_test_split(
        X, y, test_size=spec.test_size, random_state=spec.random_state, stratify=strat
    )
    return cast(tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], res)
