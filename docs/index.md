# nescience

> Machine learning guided by the Minimum Nescience Principle.

## Why

- **μ** (miscoding): poor representation makes learning impossible.
- **ι** (inaccuracy): we need out-of-sample performance.
- **σ** (surfeit): avoid superfluous complexity.

## How

Combine μ, ι, σ into a single objective and search over features+estimators,
scoring candidates on a fresh validation fold.

## What

- Pluggable metrics
- `NescienceClassifier`, `NescienceRegressor`
- `NescienceSearchCV`

```python
from nescience.estimators import NescienceClassifier
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
clf = NescienceClassifier(search_budget=60, random_state=0)
clf.fit(X, y)
clf.best_estimator_, clf.best_features_, clf.nescience_breakdown_
```
