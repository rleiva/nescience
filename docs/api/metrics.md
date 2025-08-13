# Metrics

### `miscoding_score(X, y, task)`

μ = 1 - mean(NMI(x_j; y))

- Classification: `mutual_info_classif`
- Regression: `mutual_info_regression`

### `inaccuracy_score(model, X_test, y_test, task)`

- Classification: ι = 1 - accuracy
- Regression: ι = RMSE / std(y)

### `surfeit_score(model, X_shape, mode)`

σ from serialized model size normalized by n or log n.

### `nescience_score(breakdown, weights, aggregator)`

Aggregates (μ, ι, σ) via equal-weighted sum or geometric mean.
