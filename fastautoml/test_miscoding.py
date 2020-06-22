from fastautoml.fastautoml import Miscoding

import numpy as np


def test_redundancy_regular():

    # Feature equal to target
    X = np.arange(100).reshape(-1, 1)
    y = np.arange(100)
    miscoding = Miscoding(redundancy=False)
    miscoding.fit(X, y)
    mscd = miscoding.miscoding_features(mode='regular')
    assert mscd == 0

def test_redundancy_adjusted():

    assert True == False

def test_redundancy_partial():

    assert True == False

def test_featuresmatrix():

    assert True == False

def test_noredundancy_regular():

    assert True == False

def test_noredundancy_adjusted():

    assert True == False

def test_noredundancy_partial():

    assert True == False

def test_MultinomialNB():

    assert True == False

def test_DecisionTreeClassifier():

    assert True == False

def test__LinearSVC():

    assert True == False

def test_MLPClassifier():

    assert True == False

def test_LinearRegression():

    assert True == False

def test_DecisionTreeRegressor():

    assert True == False

def test_LinearSVR():

    assert True == False

def test_MLPRegressor():

    assert True == False

def test_subset():

    assert True == False    