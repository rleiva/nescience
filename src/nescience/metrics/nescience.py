from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict


class NescienceBreakdown(TypedDict):
    mu: float
    iota: float
    sigma: float


@dataclass(frozen=True)
class NescienceWeights:
    mu: float = 1.0
    iota: float = 1.0
    sigma: float = 1.0


@dataclass(frozen=True)
class NescienceAggregator:
    kind: Literal["weighted_sum", "geometric_mean"] = "weighted_sum"

    def __call__(self, breakdown: NescienceBreakdown, weights: NescienceWeights) -> float:
        mu, iota, sigma = breakdown["mu"], breakdown["iota"], breakdown["sigma"]
        if self.kind == "weighted_sum":
            wsum = weights.mu * mu + weights.iota * iota + weights.sigma * sigma
            denom = max(1e-12, weights.mu + weights.iota + weights.sigma)
            return float(wsum / denom)
        if self.kind == "geometric_mean":
            eps = 1e-12
            return float((mu + eps) ** (1 / 3) * (iota + eps) ** (1 / 3) * (sigma + eps) ** (1 / 3))
        raise ValueError("Unknown aggregator kind")


def nescience_score(
    breakdown: NescienceBreakdown,
    weights: NescienceWeights | None = None,
    aggregator: NescienceAggregator | None = None,
) -> float:
    """Aggregate (μ, ι, σ) into a single Nescience score.

    Defaults to equal-weighted sum. Lower is better.
    """
    weights = weights or NescienceWeights()
    aggregator = aggregator or NescienceAggregator("weighted_sum")
    return aggregator(breakdown, weights)
