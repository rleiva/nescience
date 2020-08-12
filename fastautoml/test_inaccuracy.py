from fastautoml.fastautoml import Inaccuracy
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer

# No error in list
def test_no_error_list():

    y = [0, 1, 2, 3] * 25
    X = [[0, 1]] * 100

    y_hat = y.copy()

    inacc = Inaccuracy()
    inacc.fit(X, y)
    inaccuracy = inacc.inaccuracy_predictions(y_hat)

    assert inaccuracy == 0

# No error in model
def test_no_error_model():

    X, y = load_breast_cancer(return_X_y=True)

    tree = DecisionTreeClassifier()
    tree.fit(X, y)

    inacc = Inaccuracy()
    inacc.fit(X, y)
    inaccuracy = inacc.inaccuracy_model(tree)

    assert inaccuracy == 0

# One error in list
def test_one_error_list():

    y = [0, 1, 2, 3] * 25
    X = [[0, 1]] * 100

    y_hat = y.copy()
    y_hat[0] = 4

    inacc = Inaccuracy()
    inacc.fit(X, y)
    inaccuracy = inacc.inaccuracy_predictions(y_hat)

    assert inaccuracy > 0

# One error in model
def test_one_error_model():

    X, y = load_breast_cancer(return_X_y=True)

    tree = DecisionTreeClassifier()
    tree.fit(X, y)

    y[0] = 1 - y[0]
    inacc = Inaccuracy()
    inacc.fit(X, y)
    inaccuracy = inacc.inaccuracy_model(tree)

    assert inaccuracy > 0

# All errors in list
def test_all_errors_list():

    y = [0, 1, 2, 3] * 25
    X = [[0, 1]] * 100

    y_hat = [4] * 100

    inacc = Inaccuracy()
    inacc.fit(X, y)
    inaccuracy = inacc.inaccuracy_predictions(y_hat)

    assert inaccuracy > 0

# All errors in model
def test_all_errors_model():

    X, y = load_breast_cancer(return_X_y=True)

    tree = DecisionTreeClassifier()
    tree.fit(X, y)

    y = [2] * len(y)
    inacc = Inaccuracy()
    inacc.fit(X, y)
    inaccuracy = inacc.inaccuracy_model(tree)

    assert inaccuracy == 1