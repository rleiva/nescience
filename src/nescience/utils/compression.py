from __future__ import annotations

import io
from typing import Any

import joblib


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
