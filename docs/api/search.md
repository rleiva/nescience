# NescienceSearchCV

A pragmatic beam+random search over feature subsets and estimator hyperparameters,
evaluated on a stratified hold-out split.

Configuration:
- `search_budget`: number of model fits
- `feature_beam`: beam width
- `weights`, `aggregator`: control the objective
