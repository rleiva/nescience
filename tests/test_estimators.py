from __future__ import annotations

from sklearn.datasets import load_breast_cancer, load_diabetes

from nescience.estimators import NescienceClassifier, NescienceRegressor


def test_classifier_fit_predict() -> None:
    X, y = load_breast_cancer(return_X_y=True)
    clf = NescienceClassifier(search_budget=20, random_state=0)
    clf.fit(X, y)
    preds = clf.predict(X[:10])
    assert preds.shape[0] == 10


def test_regressor_fit_predict() -> None:
    X, y = load_diabetes(return_X_y=True)
    reg = NescienceRegressor(search_budget=20, random_state=0)
    reg.fit(X, y)
    preds = reg.predict(X[:10])
    assert preds.shape[0] == 10
