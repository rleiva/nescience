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
            Kx = optimal_code_length(x1=self.X_[:,i], numeric1=self.X_isnumeric[i])

            # Check if elements are not comparable
            if Cy.shape[0] != Cx.shape[0]:
                Penalties_xy.append(np.nan)
                Penalties_yx.append(np.nan)
                continue
 
            #     if Cy.shape[0] < Cx.shape[0]:
            #         Cx = unique_count(x1=self.X_[:,i], numeric1=self.X_isnumeric[i], n_bins=Cy.shape[0])
            #         Lx = - np.log2(Cx / np.sum(Cx))
            #         Kx = optimal_code_length(x1=self.X_[:,i], numeric1=self.X_isnumeric[i], n_bins=Cy.shape[0])
            #     else:
            #         Cy_adapted = unique_count(x1=self.y_, numeric1=self.y_isnumeric, n_bins=Cx.shape[0])
            #         Ly_adapted = - np.log2(Cy_adapted / np.sum(Cy_adapted))
            #         Ky_adapted = optimal_code_length(x1=self.y_, numeric1=self.y_isnumeric, n_bins=Cx.shape[0])
            # 
            # else:

            #     Cy_adapted = Cy
            #     Ly_adapted = Ly
            #     Ky_adapted = Ky

            # Compute the penalties
            # Lx_y = np.sum(Cx * Ly_adapted)
            # Ly_x = np.sum(Cy_adapted * Lx)
            # Px_y = Ly_x - Ky_adapted
            # Py_x = Lx_y - Kx

            Lx_y = np.sum(Cx * Ly)
            Ly_x = np.sum(Cy * Lx)
            Px_y = Ly_x - Ky
            Py_x = Lx_y - Kx

            Penalties_xy.append(Px_y)
            Penalties_yx.append(Py_x)

        return Penalties_xy, Penalties_yx

    def penalty_matrix(self):

        Penalties = np.zeros((self.X_.shape[1], self.X_.shape[1]))

        for i in np.arange(self.X_.shape[1]):

            # Counts, length, and Kolmogorov complexity for X_i 
            Cxi = unique_count(x1=self.X_[:,i], numeric1=self.X_isnumeric[i])
            Lxi = - np.log2(Cxi / np.sum(Cxi))
            Kxi = optimal_code_length(x1=self.X_[:,i], numeric1=self.X_isnumeric[i])

            for j in np.arange(self.X_.shape[1]):

                # Counts, length, and Kolmogorov complexity for X_j 
                Cxj = unique_count(x1=self.X_[:,j], numeric1=self.X_isnumeric[j])
                Lxj = - np.log2(Cxj / np.sum(Cxj))
                Kxj = optimal_code_length(x1=self.X_[:,j], numeric1=self.X_isnumeric[j])

                # Check if elements are not comparable
                if Cxi.shape[0] != Cxj.shape[0]:
                    Penalties[i][j] = np.nan
                    Penalties[j][i] = np.nan
                    continue

                Lxj_xi = np.sum(Cxj * Lxi)
                Lxi_xj = np.sum(Cxi * Lxj)
                Pxj_xi = Lxi_xj - Kxi
                Pxi_xj = Lxj_xi - Kxj

                Penalties[i][j] = Pxj_xi
                Penalties[j][i] = Pxi_xj

        return Penalties


    def causality(self):

        causalities_xy = list()
        causalities_yx = list()
        strengths      = list()

        # Complexity of the target variable
        Ky = optimal_code_length(x1=self.y_, numeric1=self.y_isnumeric)

        for i in np.arange(self.X_.shape[1]):

            # Kolmogorov complexity for X_i 
            Kx = optimal_code_length(x1=self.X_[:,i], numeric1=self.X_isnumeric[i])

            # TODO: Think about this
            if Kx != 0:

                # Kolmogorov complexity joint
                Kjoint = optimal_code_length(x1=self.y_, numeric1=self.y_isnumeric, x2=self.X_[:,i], numeric2=self.X_isnumeric[i])

                cxy = (Kjoint - Kx) / Ky
                cyx = (Kjoint - Ky) / Kx

                causalities_xy.append(cxy)
                causalities_yx.append(cyx)

                strengths.append( abs(cxy - cyx) )

            else:

                causalities_xy.append(np.nan)
                causalities_yx.append(np.nan)

                strengths.append(np.nan)


        return causalities_xy, causalities_yx, strengths


    def causality_matrix(self, return_strength=False):

        np.seterr('raise')

        causalities = np.zeros((self.X_.shape[1], self.X_.shape[1]))
        strengths   = np.zeros((self.X_.shape[1], self.X_.shape[1]))

        for i in np.arange(self.X_.shape[1]):

            # Kolmogorov complexity for X_i 
            Ki = optimal_code_length(x1=self.X_[:,i], numeric1=self.X_isnumeric[i])

            # TODO: Think about this
            if Ki == 0:
                continue

            for j in np.arange(self.X_.shape[1]):

                # Kolmogorov complexity for X_j 
                Kj = optimal_code_length(x1=self.X_[:,j], numeric1=self.X_isnumeric[j])

                # TODO: Think about this
                if Kj == 0:
                    continue

                # Kolmogorov complexity joint
                Kjoint = optimal_code_length(x1=self.X_[:,i], numeric1=self.X_isnumeric[i], x2=self.X_[:,j], numeric2=self.X_isnumeric[j])

                # TODO: RemoveME
                # print("Joint:", Kjoint, "Ki:", Ki, "Kj:", Kj)

                causalities[i][j] = (Kjoint - Ki) / Kj
                causalities[j][i] = (Kjoint - Kj) / Ki

                strengths[i][j] = strengths[j][i] = abs(causalities[i][j] - causalities[j][i])

        if return_strength:
            return causalities, strengths

        return causalities