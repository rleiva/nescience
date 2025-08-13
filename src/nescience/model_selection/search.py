from __future__ import annotations

import math
import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import ParameterGrid

from ..metrics.inaccuracy import inaccuracy_score
from ..metrics.miscoding import miscoding_score
from ..metrics.nescience import (
    NescienceAggregator,
    NescienceBreakdown,
    NescienceWeights,
    nescience_score,
)
from ..metrics.surfeit import surfeit_score
from ..utils.validation import HoldoutSpec, holdout_split


@dataclass
class CandidateResult:
    estimator: BaseEstimator
    params: dict[str, Any]
    selected_features: np.ndarray
    breakdown: NescienceBreakdown
    score: float


def _evaluate_candidate(
    base_estimator: BaseEstimator,
    params: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task: Literal["classification", "regression"],
    selected_features: np.ndarray,
    weights: NescienceWeights,
    aggregator: NescienceAggregator,
    surfeit_mode: Literal["bytes_per_sample", "bytes_log_n"],
) -> CandidateResult:
    est = clone(base_estimator)
    est.set_params(**params)

    # Fit on selected features
    Xtr = X_train[:, selected_features]
    Xva = X_val[:, selected_features]
    est.fit(Xtr, y_train)

    mu = miscoding_score(Xtr, y_train, task=task)
    iota = inaccuracy_score(est, Xva, y_val, task=task)
    sigma = surfeit_score(est, X_shape=Xtr.shape, mode=surfeit_mode)  # proxy

    breakdown: NescienceBreakdown = {"mu": float(mu), "iota": float(iota), "sigma": float(sigma)}
    score = nescience_score(breakdown, weights=weights, aggregator=aggregator)

    return CandidateResult(
        estimator=est,
        params=params,
        selected_features=selected_features,
        breakdown=breakdown,
        score=float(score),
    )


@dataclass
class NescienceSearchConfig:
    search_budget: int = 100
    random_state: int | None = None
    feature_beam: int = 10
    weights: NescienceWeights = NescienceWeights()
    aggregator: NescienceAggregator = NescienceAggregator("weighted_sum")
    surfeit_mode: Literal["bytes_per_sample", "bytes_log_n"] = "bytes_per_sample"
    n_jobs: int = 1


class NescienceSearchCV:
    """A light-weight search that minimizes Nescience over models and features."""

    def __init__(
        self,
        estimators: dict[str, BaseEstimator],
        param_grids: dict[str, dict[str, Iterable[Any]]],
        task: Literal["classification", "regression"],
        config: NescienceSearchConfig | None = None,
        holdout: HoldoutSpec | None = None,
    ) -> None:
        self.estimators = estimators
        self.param_grids = param_grids
        self.task = task
        self.config = config or NescienceSearchConfig()
        self.holdout = holdout or HoldoutSpec()
        self.best_: CandidateResult | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NescienceSearchCV:
        rng = random.Random(self.config.random_state)
        X = np.asarray(X)
        y = np.asarray(y)
        X_tr, X_va, y_tr, y_va = holdout_split(
            X,
            y,
            task=self.task,
            spec=HoldoutSpec(
                test_size=self.holdout.test_size,
                stratify=self.holdout.stratify,
                random_state=self.holdout.random_state,
            ),
        )

        n_features = X_tr.shape[1]
        remaining = list(range(n_features))
        beam: list[np.ndarray] = [np.array([], dtype=int)]

        budget = self.config.search_budget
        best: CandidateResult | None = None

        while remaining and budget > 0:
            new_beam: list[np.ndarray] = []
            # Expand beam by adding one feature to each subset
            for subset in beam:
                for f in remaining:
                    if f in subset:
                        continue
                    cand = np.sort(np.concatenate([subset, [f]])).astype(int)
                    new_beam.append(cand)
            # Deduplicate
            uniq = {tuple(x.tolist()): x for x in new_beam}
            new_beam = list(uniq.values())

            # Score a random sample of estimator+params for each subset
            scored: list[CandidateResult] = []

            for subset in new_beam:
                for est_name, base_estimator in self.estimators.items():
                    grid = list(ParameterGrid(self.param_grids.get(est_name, {})))
                    if len(grid) == 0:
                        grid = [{}]
                    rng.shuffle(grid)
                    sample = grid[: max(1, math.ceil(len(grid) * 0.25))]  # sample ~25%
                    for params in sample:
                        if budget <= 0:
                            break
                        res = _evaluate_candidate(
                            base_estimator,
                            params,
                            X_tr,
                            y_tr,
                            X_va,
                            y_va,
                            task=self.task,
                            selected_features=subset,
                            weights=self.config.weights,
                            aggregator=self.config.aggregator,
                            surfeit_mode=self.config.surfeit_mode,
                        )
                        scored.append(res)
                        budget -= 1
                    if budget <= 0:
                        break
                if budget <= 0:
                    break

            if not scored:
                break

            # Update best
            scored.sort(key=lambda r: r.score)
            if best is None or scored[0].score < best.score:
                best = scored[0]

            # Beam prune by current best score
            subset_scores: dict[tuple[int, ...], list[float]] = {}
            for r in scored:
                key = tuple(r.selected_features.tolist())
                subset_scores.setdefault(key, []).append(r.score)
            # Use median per subset
            entries = sorted(
                ((k, float(np.median(v))) for k, v in subset_scores.items()), key=lambda x: x[1]
            )
            beam = [np.array(k, dtype=int) for k, _ in entries[: self.config.feature_beam]]
            # Update remaining
            all_selected = (
                {int(i) for i in np.unique(np.concatenate(beam)).tolist()} if beam else set()
            )
            remaining = [f for f in range(n_features) if f not in all_selected]

        self.best_ = best
        return self

    @property
    def best_estimator_(self) -> BaseEstimator:
        if self.best_ is None:
            raise RuntimeError("Search has not been run or found no candidate.")
        return self.best_.estimator

    @property
    def best_params_(self) -> dict[str, Any]:
        if self.best_ is None:
            raise RuntimeError("Search has not been run or found no candidate.")
        return self.best_.params

    @property
    def best_features_(self) -> np.ndarray:
        if self.best_ is None:
            raise RuntimeError("Search has not been run or found no candidate.")
        return self.best_.selected_features

    @property
    def best_score_(self) -> float:
        if self.best_ is None:
            raise RuntimeError("Search has not been run or found no candidate.")
        return self.best_.score

    @property
    def best_breakdown_(self) -> NescienceBreakdown:
        if self.best_ is None:
            raise RuntimeError("Search has not been run or found no candidate.")
        return self.best_.breakdown
