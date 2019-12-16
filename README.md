# fastautoml 

``fastautoml`` is a Python module for fast auto machine learning built on top of ``scikit-learn``.

## Prerequisites

``fastautoml`` requires:

 * scikit-learn (>= 0.22)
 * pandas       (>= 0.25)

## User Installation

If you already have a working installation of ``scikit-learn`` and ``pandas``, the easiest way to install ``fastautoml`` is using ``pip``:

```
pip install -U fastautoml
```

or ``conda``:

```
conda install fastautoml
```

## Running

The following example shows how to compute an optimal model for the MNIST dataset included with ``scikit-learn``.

```
from fastautoml.fastautoml import AutoClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = AutoClassifier()
model.fit(X_train, y_train)
print("Score:", model.score(X_test, y_test))
```

## Help

 * Examples of usage: https://github.com/autofastml/examples

## Authors

[R. Leiva](https://github.com/rleiva) and [contributors](https://github.com/autofastml/Contributors.md).

## License

This project is licensed under the 3-Clause BSD license - see the [LICENSE.md](LICENSE.md) file for details.

## Aknowledgements

This project has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No 732667 [RECAP](https://recap-project.eu/)

