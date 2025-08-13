# nescience

**Machine learning guided by the Minimum Nescience Principle** — pick models and features
by minimizing a three-part objective:
- **μ (miscoding)** — representation inadequacy (low MI between X and y).
- **ι (inaccuracy)** — predictive error on fresh data.
- **σ (surfeit)** — model superfluity/complexity.

The library provides metrics, a unified **Nescience** objective, and search wrappers that choose
both features and estimators with scikit-learn compatibility.

> Research background: R. A. Garcia Leiva, *A Mathematical Theory of the Unknown* (Theory of Nescience).

## Quick start

```bash
pip install -U nescience
```

```python
from nescience.estimators import NescienceClassifier
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)

clf = NescienceClassifier(search_budget=60, random_state=0)
clf.fit(X, y)
print(clf.best_estimator_)
print(clf.nescience_breakdown_)
```

## Philosophy (short)

- **μ**: If the representation barely informs the target, you are doomed. We approximate this with normalized mutual information between features and target.
- **ι**: If predictions fail on unseen data, the description is inaccurate. We estimate with task-appropriate loss.
- **σ**: If the model is overly complex relative to the data, you’re probably memorizing. We approximate with a normalized description-length proxy.

Each component is pluggable.

## Status

This is a reboot (v0.2.0). API may evolve before v1.0. See the roadmap in `CONTRIBUTING.md`.
