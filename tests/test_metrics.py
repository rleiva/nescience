from __future__ import annotations

from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.linear_model import Ridge

from nescience.metrics.inaccuracy import inaccuracy_score
from nescience.metrics.miscoding import miscoding_score
from nescience.metrics.nescience import nescience_score
from nescience.metrics.surfeit import surfeit_score


def test_miscoding_classif_reasonable() -> None:
    X, y = load_breast_cancer(return_X_y=True)
    mu = miscoding_score(X, y, task="classification")
    assert 0.0 <= mu <= 1.0


def test_inaccuracy_and_surfeit() -> None:
    X, y = load_diabetes(return_X_y=True)
    model = Ridge().fit(X, y)
    iota = inaccuracy_score(model, X, y, task="regression")
    sigma = surfeit_score(model, X.shape)
    assert iota >= 0.0
    assert sigma >= 0.0


def test_nescience_aggregate() -> None:
    score = nescience_score({"mu": 0.2, "iota": 0.3, "sigma": 0.1})
    assert 0.0 <= score <= 1.0
