# Run tests with pytest

from fastautoml.miscoding import Miscoding

import numpy as np
from scipy.stats import norm, expon

# Regular miscoding, redundancy allowed
def test_redundancy_regular():

    # Feature equal to target
    X = np.arange(100).reshape(-1, 1)
    y = np.arange(100)
    miscoding = Miscoding(X_type="numeric", y_type="numeric", redundancy=False)
    miscoding.fit(X, y)
    mscd = miscoding.miscoding_features(mode='regular')
    assert mscd == 0

    # Feature perfectly correlated with target
    X = np.arange(100)
    y = -X
    X = X.reshape(-1, 1)
    miscoding = Miscoding(X_type="numeric", y_type="numeric", redundancy=False)
    miscoding.fit(X, y)
    mscd = miscoding.miscoding_features(mode='regular')
    assert mscd == 0

    # Feature and target are not related
    X = np.array([1, 2] * 50).reshape(-1, 1)
    y = ["a"] * 50 + ["b"] * 50
    miscoding = Miscoding(X_type="numeric", y_type="categorical", redundancy=False)
    miscoding.fit(X, y)
    mscd = miscoding.miscoding_features(mode='regular')
    assert mscd == 1

    # Compare the miscoding of a relevant feature with a non-relevant feature
    y  = norm.rvs(loc=3, size=10000)
    x1 = y + np.random.randn()
    x2 = expon.rvs(size=10000)
    X = np.column_stack((x1, x2))
    miscoding = Miscoding(X_type="numeric", y_type="numeric", redundancy=False)
    miscoding.fit(X, y)
    mscd = miscoding.miscoding_features(mode='regular')
    assert mscd[0] < mscd[1]

def test_redundancy_adjusted():

    # Feature equal to target
    X = np.arange(100).reshape(-1, 1)
    y = np.arange(100)
    miscoding = Miscoding(X_type="numeric", y_type="numeric", redundancy=False)
    miscoding.fit(X, y)
    mscd = miscoding.miscoding_features(mode='adjusted')
    assert mscd == 1

    # Feature and target are not related
    X = np.array([1, 2] * 50).reshape(-1, 1)
    y = ["a"] * 50 + ["b"] * 50
    miscoding = Miscoding(X_type="numeric", y_type="categorical", redundancy=False)
    miscoding.fit(X, y)
    mscd = miscoding.miscoding_features(mode='adjusted')
    assert mscd == 0

    # Miscoding should add up to 1
    y  = norm.rvs(loc=3, size=10000)
    x1 = y + np.random.randn()
    x2 = expon.rvs(size=10000)
    X = np.column_stack((x1, x2))
    miscoding = Miscoding(X_type="numeric", y_type="numeric", redundancy=False)
    miscoding.fit(X, y)
    mscd = miscoding.miscoding_features(mode='adjusted')
    assert np.sum(mscd) > 0.99

def test_redundancy_partial():

    # Non-relevant fatures should have a negative contribution
    y  = norm.rvs(loc=3, size=10000)
    x1 = y + np.random.randn()
    x2 = expon.rvs(size=10000)
    X = np.column_stack((x1, x2))
    miscoding = Miscoding(X_type="numeric", y_type="numeric", redundancy=False)
    miscoding.fit(X, y)
    mscd = miscoding.miscoding_features(mode='partial')
    assert mscd[1] < 0

def test_featuresmatrix():

    # Non-relevant fatures should have a negative contribution
    x1 = y = norm.rvs(loc=3, size=10000)
    x2 = x1 + np.random.randn()
    x3 = expon.rvs(size=10000)
    X = np.column_stack((x1, x2, x3))
    miscoding = Miscoding(X_type="numeric", y_type="numeric", redundancy=False)
    miscoding.fit(X, y)
    mscd = miscoding.features_matrix()
    assert mscd[0,1] > mscd[0, 2]

def test_noredundancy_regular():

    # Non-redundant features not related with target
    x1 = np.array([1, 2, 3, 4] * 250)
    x2 = np.array([5 , 6] * 500)
    X = np.column_stack((x1, x2))
    y = ["a"] * 500 + ["b"] * 500
    miscoding = Miscoding(X_type="numeric", y_type="categorical", redundancy=True)
    miscoding.fit(X, y)
    mscd = miscoding.miscoding_features(mode='regular')
    assert mscd[0] == 1 
    assert mscd[1] == 1

    # Redundant features not related with target
    x1 = np.array([1, 2, 3, 4] * 250)
    x2 = np.array([5, 6, 7, 8] * 250)
    X = np.column_stack((x1, x2))
    y = ["a"] * 500 + ["b"] * 500
    miscoding = Miscoding(X_type="numeric", y_type="categorical", redundancy=True)
    miscoding.fit(X, y)
    mscd = miscoding.miscoding_features(mode='regular')
    assert mscd[0] == 1
    assert mscd[1] == 1

    # Non-redundant features corelated with target
    x1 = np.array([1, 2, 3, 4] * 250)
    x2 = np.array([5, 6] * 500)
    X = np.column_stack((x1, x2))
    y = ["a", "b"] * 500
    miscoding = Miscoding(X_type="numeric", y_type="categorical", redundancy=True)
    miscoding.fit(X, y)
    mscd = miscoding.miscoding_features(mode='regular')
    assert mscd[0] == 0.5
    assert mscd[1] == 0

    # Redundant features corelated with target
    x1 = np.array([1, 2, 3, 4] * 250)
    x2 = np.array([5, 6, 7, 8] * 250)
    X = np.column_stack((x1, x2))
    y = ["a", "b", "c", "d"] * 250
    miscoding = Miscoding(X_type="numeric", y_type="categorical", redundancy=True)
    miscoding.fit(X, y)
    mscd = miscoding.miscoding_features(mode='regular')
    assert mscd[0] == 0
    assert mscd[1] == 0

def test_noredundancy_adjusted():

    # Miscoding should add up to 1
    y  = norm.rvs(loc=3, size=10000)
    x1 = y + np.random.randn()
    x2 = expon.rvs(size=10000)
    X = np.column_stack((x1, x2))
    miscoding = Miscoding(X_type="numeric", y_type="numeric", redundancy=False)
    miscoding.fit(X, y)
    mscd = miscoding.miscoding_features(mode='adjusted')
    assert np.sum(mscd) > 0.99

def test_noredundancy_partial():

    # Non-relevant fatures should have a negative contribution
    y  = norm.rvs(loc=3, size=10000)
    x1 = y + np.random.randn()
    x2 = expon.rvs(size=10000)
    X = np.column_stack((x1, x2))
    miscoding = Miscoding(X_type="numeric", y_type="numeric", redundancy=False)
    miscoding.fit(X, y)
    mscd = miscoding.miscoding_features(mode='partial')
    assert mscd[1] < 0

def test_MultinomialNB():

    # TODO: Pending
    assert True == True

def test_DecisionTreeClassifier():

    # TODO: Pending
    assert True == True

def test__LinearSVC():

    # TODO: Pending
    assert True == True

def test_MLPClassifier():

    # TODO: Pending
    assert True == True

def test_LinearRegression():

    # TODO: Pending
    assert True == True

def test_DecisionTreeRegressor():

    # TODO: Pending
    assert True == True

def test_LinearSVR():

    # TODO: Pending
    assert True == True

def test_MLPRegressor():

    # TODO: Pending
    assert True == True

def test_subset():

    # TODO: Think a test for this
    assert True == True
