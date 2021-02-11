"""
causality.py

Fast auto machine learning
with the minimum nescience principle

@author:    Rafael Garcia Leiva
@mail:      rgarcialeiva@gmail.com
@web:       http://www.mathematicsunknown.com/
@copyright: GNU GPLv3
@version:   0.8
"""

from .utils import optimal_code_length
from .utils import unique_count

import numpy  as np
import pandas as pd

from sklearn.base             import BaseEstimator														
from sklearn.utils            import check_X_y
from sklearn.utils.validation import check_is_fitted
from sklearn.utils            import check_array
from sklearn.utils            import column_or_1d


class Causality(BaseEstimator):
    """
    The fastautoml causality class allow us to compute the between
    a variable and the target, and between pairs of variables.

    Example of usage:
        
        from fastautoml.causality import Causality
        from sklearn.data import load_boston

        X, y = load_boston(return_X_y=True)

        cause = Causality()
        cause.fit(X, y)
        cause.penalty()
    """    

    def __init__(self, X_type="numeric", y_type="numeric"):
        """
        Initialization of the class Causality
        
        Parameters
        ----------
        X_type:     The type of the features, numeric, mixed or categorical
        y_type:     The type of the target, numeric or categorical
        """        

        valid_X_types = ("numeric", "mixed", "categorical")
        valid_y_types = ("numeric", "categorical")

        if X_type not in valid_X_types:
            raise ValueError("Valid options for 'X_type' are {}. "
                             "Got vartype={!r} instead."
                             .format(valid_X_types, X_type))

        if y_type not in valid_y_types:
            raise ValueError("Valid options for 'y_type' are {}. "
                             "Got vartype={!r} instead."
                             .format(valid_y_types, y_type))

        self.X_type     = X_type
        self.y_type     = y_type
        
        return None
    
    
    def fit(self, X, y=None):
        """
        Learn empirically the casuality among the features of X
        and / or the target y.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Sample vectors from which to compute miscoding.
            array-like, numpy or pandas array in case of numerical types
            pandas array in case of mixed or caregorical types
            
        y : (optional) array-like, shape (n_samples)
            The target values as numbers or strings.
            
        Returns
        -------
        self
        """

        self.X_ = check_array(X)

        if self.X_type == "mixed" or self.X_type == "categorical":

            if isinstance(X, pd.DataFrame):
                self.X_isnumeric = [np.issubdtype(my_type, np.number) for my_type in X.dtypes]
            else:
                raise ValueError("Only DataFrame is allowed for X of type 'mixed' and 'categorical."
                                 "Got type {!r} instead."
                                 .format(type(X)))
                
        else:
            self.X_isnumeric = [True] * X.shape[1]

        # self.X_ = column_or_1d(X)

        if y is not None:

            self.y_ = column_or_1d(y)

            if self.y_type == "numeric":
                self.y_isnumeric = True
            else:
                self.y_isnumeric = False

        else:
            self.y_ = None
        
        return self


    def penalty(self):

        Penalties_xy = list()
        Penalties_yx = list()

        # Counts, length, and Kolmogorov complexity of the target variable
        Cy = unique_count(x1=self.y_, numeric1=self.y_isnumeric)
        Ly = - np.log2(Cy / np.sum(Cy))
        Ky = optimal_code_length(x1=self.y_, numeric1=self.y_isnumeric)

        for i in np.arange(self.X_.shape[1]):

            # Counts, length, and Kolmogorov complexity for X_i 
            Cx = unique_count(x1=self.X_[:,i], numeric1=self.X_isnumeric[i])
            Lx = - np.log2(Cx / np.sum(Cx))
            Kx = optimal_code_length(x1=self.y_, numeric1=self.y_isnumeric)

            if Cy.shape != Cx.shape:
                # TODO: Elements are not comparable
                Penalties_xy.append(np.nan)
                Penalties_yx.append(np.nan)
                continue
            
            # Compute the penalties
            Lx_y = np.sum(Cx * Ly)
            Ly_x = np.sum(Cy * Lx)
            Px_y = Ly_x - Ky
            Py_x = Lx_y - Kx

            Penalties_xy.append(Px_y)
            Penalties_yx.append(Py_x)

        return Penalties_xy, Penalties_yx


    def efficiency(self):
        return None

    def penalty_matrix(self):
        return None
        
    def efficiency_matrix(self):
        return None
