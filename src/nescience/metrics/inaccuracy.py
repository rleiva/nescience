from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score, mean_squared_error


def inaccuracy_score(
    model: ClassifierMixin | RegressorMixin,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: Literal["classification", "regression"],
    normalise: bool = True,
) -> float:
    """Compute **ι (inaccuracy)** as out-of-sample error.

    - Classification: ι = 1 - accuracy.
    - Regression: ι = RMSE / (std(y_test) + eps)   if normalise else MSE.

    Parameters
    ----------
    model : fitted estimator
    X_test, y_test : arrays
    task : {"classification", "regression"}
    normalise : bool
        For regression, return a unitless error.

    Returns
    -------
    float in [0, 1] for classification; non-negative for regression.
    """
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    if task == "classification":
        y_hat = model.predict(X_test)
        acc = accuracy_score(y_test, y_hat)
        return float(np.clip(1.0 - acc, 0.0, 1.0))

    if task == "regression":
        y_hat = model.predict(X_test)
        mse = mean_squared_error(y_test, y_hat)
        if not normalise:
            return float(mse)
        rmse = float(np.sqrt(mse))
        denom = float(np.std(y_test) + 1e-12)
        return float(rmse / denom)

    raise ValueError("task must be 'classification' or 'regression'")
