#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 12:21:03 2020

@author: rleiva
"""

from fastautoml.fastautoml import _discretize_vector, _discretize_matrix
from fastautoml.fastautoml import _optimal_code_length, _optimal_code_length_joint, _optimal_code_length_3joint
import numpy as np


def test_discretize_vector():

    # One hundred different values should be discretized in ten bins
    X = np.arange(100)
    new_X = _discretize_vector(X)
    assert len(np.unique(new_X)) == 10

    # Sufficent number of samples per value, so no need to discretize
    X = [0, 1, 2, 3] * 25
    new_X = _discretize_vector(X)
    assert len(np.unique(new_X)) == len(np.unique(X))

    # Check extreme case
    X = np.zeros(100)
    new_X = _discretize_vector(X)
    assert len(np.unique(new_X)) == 1

def test_discretize_matrix():

    # Two variables with one hundred different values each
    # should be discretized in to two variables with ten bins each
    X = np.arange(200)
    X = X.reshape((100,2))
    new_X = _discretize_matrix(X)
    assert len(np.unique(new_X[:,0])) == 10
    assert len(np.unique(new_X[:,1])) == 10

def test_optimal_code_length():

    # Four equally probable values
    X = [0, 1, 2, 3] * 25
    length = _optimal_code_length(X)
    assert length == 4 * 25 * (- np.log2(0.25))

    # One hundred different values requires a previous discretization
    X = np.arange(100)
    length = _optimal_code_length(X)
    assert np.round(length, 5) == np.round(10 * 10 * (- np.log2(0.1)), 5)

    # Check exteme case
    X = np.zeros(100)
    length = _optimal_code_length(X)
    assert length == 0
     
def test_optimal_code_length_joint():

    # Four equally probable values
    X1 = [0, 1, 2, 3] * 25
    X2 = [4, 5, 6, 7] * 25
    length = _optimal_code_length_joint(X1, X2)
    assert length == 4 * 25 * (- np.log2(0.25))

    # One hundred different values requires a previous discretization
    X1 = np.arange(100)
    X2 = np.arange(100)
    length = _optimal_code_length_joint(X1, X2)
    assert np.round(length, 5) == np.round(10 * 10 * (- np.log2(0.1)), 5)

    # Non-equally probable values
    X1 = [0, 1, 2, 3, 4] * 20
    X2 = [10, 11, 12, 13, 14, 15, 16, 17,  18, 19] * 10
    length = _optimal_code_length_joint(X1, X2)
    assert np.round(length, 5) == np.round(10 * 10 * (- np.log2(0.1)), 5)

    # Check extreme case
    X1 = np.zeros(100)
    X2 = np.ones(100)
    length = _optimal_code_length_joint(X1, X2)
    assert length == 0

def test_optimal_code_length_3joint():

    # Four equally probable values
    X1 = [0, 1, 2, 3]   * 25
    X2 = [4, 5, 6, 7]   * 25
    X3 = [8, 9, 10, 11] * 25
    length = _optimal_code_length_3joint(X1, X2, X3)
    assert length == 4 * 25 * (- np.log2(0.25))

    # One hundred different values requires a previous discretization
    X1 = np.arange(100)
    X2 = np.arange(100)
    X3 = np.arange(100)
    length = _optimal_code_length_3joint(X1, X2, X3)
    assert np.round(length, 5) == np.round(10 * 10 * (- np.log2(0.1)), 5)

    # Non-equally probable values
    X1 = [0, 1, 2, 3, 4] * 200
    X2 = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19] * 100
    X3 = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
          30, 31, 32, 33, 34, 35, 36, 37, 38, 39] * 50
    length = _optimal_code_length_3joint(X1, X2, X3)
    assert np.round(length, 5) == np.round(50 * 20 * (- np.log2(0.05)), 5)

    # Check extreme case
    X1 = np.repeat(1, 100)
    X2 = np.repeat(2, 100)
    X3 = np.repeat(3, 100)
    length = _optimal_code_length_3joint(X1, X2, X3)
    assert length == 0