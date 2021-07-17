"""
nescience.py

Machine learning
with the minimum nescience principle

@author:    Rafael Garcia Leiva
@mail:      rgarcialeiva@gmail.com
@web:       http://www.mathematicsunknown.com/
@copyright: GNU GPLv3
"""

import math

from sklearn.base import BaseEstimator
														
from sklearn.utils            import check_X_y
from sklearn.utils.validation import check_is_fitted

from .miscoding  import Miscoding
from .surfeit    import Surfeit
from .inaccuracy import Inaccuracy


class Nescience(BaseEstimator):
    """
    The nescience class allow us to estimate how much
    we do know about a problem given a dataset and a model.

    Example of usage:
        
        from nescience.nescience import Nescience
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.datasets import load_digits

        X, y = load_digits(return_X_y=True)

        tree = DecisionTreeClassifier(min_samples_leaf=5, random_state=42)
        tree.fit(X, y)

        nsc = Nescience()
        nsc.fit(X, y)
        nsc.nescience(tree)
    """ 

    def __init__(self, X_type="numeric", y_type="numeric", compressor="bz2", method="Harmonic"):
        """
        Initialization of the class Nescience
        
        Parameters
        ----------
        X_type     : The type of the features, numeric, mixed or categorical
        y_type     : The type of the target, numeric or categorical
        compressor : The compressor used to encode the model (bz2, lzma or zlib)
        method     : function used to compute the nescience, valid values are
                     "Euclid", "Arithmetic", "Geometric", "Product", "Addition"
                     and "Harmonic"
        """

        valid_X_types = ("numeric", "mixed", "categorical")
        valid_y_types = ("numeric", "categorical")
        valid_methods = ("Euclid", "Arithmetic", "Geometric", "Product",
                         "Addition", "Harmonic")

        if X_type not in valid_X_types:
            raise ValueError("Valid options for 'X_type' are {}. "
                             "Got vartype={!r} instead."
                             .format(valid_X_types, X_type))

        if y_type not in valid_y_types:
            raise ValueError("Valid options for 'y_type' are {}. "
                             "Got vartype={!r} instead."
                             .format(valid_y_types, y_type))

        # Cheking the compressor is left to class Surfeit

        if method not in valid_methods:
            raise ValueError("Valid options for 'method' are {}. "
                             "Got vartype={!r} instead."
                             .format(valid_methods, method))                             

        self.X_type     = X_type
        self.y_type     = y_type
        self.compressor = compressor
        self.method     = method

        return None

    
    def fit(self, X, y):
        """
        Initialization of the class nescience
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Sample vectors from which to compute miscoding.
        y : array-like, shape (n_samples)
            The target values (class labels) as numbers or strings.          
        """
		
        X, y = check_X_y(X, y, dtype=None)

        self.miscoding_  = Miscoding(X_type=self.X_type, y_type=self.y_type, redundancy=False)
        self.miscoding_.fit(X, y)

        self.inaccuracy_ = Inaccuracy(y_type=self.y_type)
        self.inaccuracy_.fit(X, y)        

        self.surfeit_    = Surfeit(y_type=self.y_type, compressor=self.compressor)
        self.surfeit_.fit(X, y)
        
        return self


    def nescience(self, model=None, subset=None, predictions=None, model_string=None):
        """
        Compute the nescience of a model
        
        Parameters
        ----------
        model     : a trained model
        subset    : array-like, shape (n_features)
                    1 if the attribute is in use, 0 otherwise
                    If None, attributes will be infrerred throught model
        predictions : array-like, shape (n_samples) The list of predicted values
        model_str : a string based representation of the model
                    If None, string will be derived from model
                    
        Returns
        -------
        Return the nescience (float)
        """
        
        check_is_fitted(self)

        if subset is None:
            miscoding = self.miscoding_.miscoding_model(model)
        else:
            miscoding = self.miscoding_.miscoding_subset(subset)

        if predictions is None:
            inaccuracy = self.inaccuracy_.inaccuracy_model(model)
        else:
            inaccuracy = self.inaccuracy_.inaccuracy_predictions(predictions)
            
        if model_string is None:
            surfeit = self.surfeit_.surfeit_model(model)
        else:
            surfeit = self.surfeit_.surfeit_string(model_string)            

        # Avoid dividing by zero
        
        if surfeit == 0:
            surfeit = 10e-6
    
        if inaccuracy == 0:
            inaccuracy = 10e-6

        if miscoding == 0:
            miscoding = 10e-6
            
        # TODO: Think about this problem
        if surfeit < inaccuracy:
            # The model is still too small to use surfeit
            surfeit = 1

        # Compute the nescience according to the method specified by the user
        if self.method == "Euclid":
            # Euclidean distance
            nescience = math.sqrt(miscoding**2 + inaccuracy**2 + surfeit**2)
        elif self.method == "Arithmetic":
            # Arithmetic mean
            nescience = (miscoding + inaccuracy + surfeit) / 3
        elif self.method == "Geometric":
            # Geometric mean
            nescience = math.pow(miscoding * inaccuracy * surfeit, 1/3)
        elif self.method == "Product":
            # The product of both quantities
            nescience = miscoding * inaccuracy * surfeit
        elif self.method == "Addition":
            # The product of both quantities
            nescience = miscoding + inaccuracy + surfeit
        # elif self.method == "Weighted":
            # Weigthed sum
            # TODO: Not yet supported
            # nescience = self.weight_miscoding * miscoding + self.weight_inaccuracy * inaccuracy + self.weight_surfeit * surfeit
        elif self.method == "Harmonic":
            # Harmonic mean
            nescience = 3 / ( (1/miscoding) + (1/inaccuracy) + (1/surfeit))
        # else -> rise exception
        
        return nescience