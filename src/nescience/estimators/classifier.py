from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from ..metrics.nescience import NescienceAggregator, NescienceBreakdown, NescienceWeights
from ..model_selection.search import NescienceSearchConfig, NescienceSearchCV

_DEFAULT_ESTIMATORS: dict[str, BaseEstimator] = {
    "logreg": LogisticRegression(max_iter=200, n_jobs=None),
    "svm": SVC(probability=False),
    "knn": KNeighborsClassifier(),
    "rf": RandomForestClassifier(random_state=0),
    "gnb": GaussianNB(),
}

_DEFAULT_PARAM_GRIDS: dict[str, dict[str, Iterable[Any]]] = {
    "logreg": {"C": [0.1, 1.0, 10.0], "penalty": ["l2"], "solver": ["lbfgs"]},
    "svm": {"C": [0.1, 1.0, 10.0], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]},
    "knn": {"n_neighbors": [3, 5, 9, 15], "weights": ["uniform", "distance"]},
    "rf": {"n_estimators": [100, 200], "max_depth": [None, 5, 10]},
    "gnb": {},  # no params
}


class NescienceClassifier(BaseEstimator, ClassifierMixin):  # type: ignore[misc]
    """Classifier that selects features+estimator by minimizing Nescience."""

    def __init__(
        self,
        candidates: dict[str, BaseEstimator] | None = None,
        param_grids: dict[str, dict[str, Iterable[Any]]] | None = None,
        search_budget: int = 100,
        feature_beam: int = 10,
        weights: NescienceWeights | None = None,
        aggregator: NescienceAggregator | None = None,
        random_state: int | None = None,
    ) -> None:
        self.candidates = candidates if candidates is not None else _DEFAULT_ESTIMATORS
        self.param_grids = param_grids if param_grids is not None else _DEFAULT_PARAM_GRIDS
        self.search_budget = search_budget
        self.feature_beam = feature_beam
        self.weights = weights or NescienceWeights()
        self.aggregator = aggregator or NescienceAggregator("weighted_sum")
        self.random_state = random_state
        # Fitted attrs (annotated Optionals for mypy strict)
        self.best_estimator_: BaseEstimator | None = None
        self.best_params_: dict[str, Any] | None = None
        self.best_features_: np.ndarray | None = None
        self.nescience_breakdown_: NescienceBreakdown | None = None
        self.nescience_score_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NescienceClassifier:
        cfg = NescienceSearchConfig(
            search_budget=self.search_budget,
            random_state=self.random_state,
            feature_beam=self.feature_beam,
            weights=self.weights,
            aggregator=self.aggregator,
        )
        search = NescienceSearchCV(
            estimators=self.candidates,
            param_grids=self.param_grids,
            task="classification",
            config=cfg,
        )
        search.fit(X, y)
        self.best_estimator_ = search.best_estimator_
        self.best_params_ = search.best_params_
        self.best_features_ = search.best_features_
        self.nescience_breakdown_ = search.best_breakdown_
        self.nescience_score_ = search.best_score_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.best_estimator_ is None or self.best_features_ is None:
            raise RuntimeError("Model not fitted.")
        return self.best_estimator_.predict(X[:, self.best_features_])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        if self.best_estimator_ is None or self.best_features_ is None:
            raise RuntimeError("Model not fitted.")
        return float(np.mean(self.predict(X) == y))
