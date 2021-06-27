# Cajal Machine Learning

`cajal` is a highly efficient open source library for machine learning based on Python and built on top of [scikit-learn](https://scikit-learn.org/stable/). The library is based on the [_minimum nescience principle_](http://www.mathematicsunknown.com/), a novel mathematical theory that measures how well we understand a problem given a representation and a description. In case of machine learning, representations are based on datasets, and descriptions are based on models.

The minimum nescience principle allow us to automate the common tasks performed by data scientists, from feature selection, model selection, or hyperparameters optimization.

`cajal` can dramatically increase the productivity of the data scientist, reducing the time to analyze and model a dataset. With `cajal` we can have results in very short time, without decreasing the accuracy (in fact, we usually have a better accuracy). `Cajal` is fast because:

* It does not requires cross-validation
* It use a greedy search for hyperparameters
* It is not based on ensembles of models

## The Library

The `cajal` library is composed of the following classes:

* `Miscoding` measures the quality of the dataset we are using to represent our problem.
* `Inaccuracy` measures the error made by the model we have trained.
* `Surfeit` measures how (unnecessarily) complex is the model we have identified.

All these metrics are combined into a single quantity, called `Nescience`, as a measure of how well we understand our problem given a dataset and a model. `Nescience` allow us to evaluate and compare models from different model families.

* `Anomalies` for ...
* `Causal` for ...

Besides to these classes, the `cajal` library provide the following automated machine-learning tools:

* `AutoRegression` for automated regression problems
* `AutoClassification` for automated classification problems
* `AutoTimeSeries` for time series based forecasting

## User Guide

This user guide contains the following sections:

* [Auto Classification](https://github.com/rleiva/fastautoml/wiki/Auto-Classification)
* [Auto Regression](https://github.com/rleiva/fastautoml/wiki/Auto-Regression)
* [Auto Time Series](https://github.com/rleiva/fastautoml/wiki/Auto-Time-Series)
* [Feature Selection](https://github.com/rleiva/fastautoml/wiki/Feature-Selection)
* [Model Inaccuacy](https://github.com/rleiva/fastautoml/wiki/Model-Inaccuracy)
* [Model Complexity](https://github.com/rleiva/fastautoml/wiki/Model-Complexity)
* [Hyperparameters Selection](https://github.com/rleiva/fastautoml/wiki/Hyperparameters-Selection)
