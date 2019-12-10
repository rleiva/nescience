# fastautoml 

``fastautoml`` is a Python module for fast auto machine learning built on top of ``scikit-learn``.

## Prerequisites

``fastautoml`` requires:

 * scikit-learn (>= 0.22)
 * pandas       (>= 0.25)

## User Installation

If you already have a working installation of ``scikit-learn`` and ``pandas``, the easiest way to install ``fastautoml`` is using ``pip``

```
pip install -U fastautoml
```

or ``conda``:

```
conda install fastautoml
```

## Running the tests

The following example shows how to compute an optimal model for the MNIST dataset included with ``scikit-learn``.

```
from fastautoml import AutoClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = AutoClassifier()
model.fit(X_train, y_train)
print("Score:", model.score(X_test, y_test))
```

## Help and Support

 * Website: http://fastautoml.org
 * HTML documentation: http://fastautoml.org

## Authors

* **R. Leiva** - *Initial work* - [R. Leiva](https://github.com/rleiva)

See also the list of [contributors](https://github.com/autofastml/contributors) who participated in this project.

## License

This project is licensed under the 3-Clause BSD license - see the [LICENSE.md](LICENSE.md) file for details.



