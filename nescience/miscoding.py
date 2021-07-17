"""
miscoding.py

Machine learning
with the minimum nescience principle

@author:    Rafael Garcia Leiva
@mail:      rgarcialeiva@gmail.com
@web:       http://www.mathematicsunknown.com/
@copyright: GNU GPLv3
"""

from .utils import optimal_code_length

import pandas as pd
import numpy  as np

from sklearn.base import BaseEstimator													
from sklearn.utils            import check_array
from sklearn.utils            import column_or_1d
from sklearn.utils.validation import check_is_fitted

# Supported classifiers

from sklearn.naive_bayes    import MultinomialNB
from sklearn.tree           import DecisionTreeClassifier
from sklearn.svm            import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.svm			import SVC

# Supported regressors

from sklearn.linear_model   import LinearRegression
from sklearn.tree           import DecisionTreeRegressor
from sklearn.svm            import LinearSVR
from sklearn.neural_network import MLPRegressor


class Miscoding(BaseEstimator):
    """
    Given a dataset X = {x1, ..., xp} composed by p features, and a target
    variable y, the miscoding of the feature xj measures how difficult is to
    reconstruct y given xj, and the other way around. We are not only
    interested in to identify how much information xj contains about y, but
    also if xj contains additional information that is not related
    to y (which is a bad thing). Miscoding also takes into account that
    feature xi might be redundant with respect to feature xj.

    The Miscoding class allow us to compute the relevance of
    features, the quality of a dataset, and select the optimal subset of
    features to include in a study

    Example of usage:
        
        from nescience.miscoding import Miscoding
        from sklearn.datasets import load_beast_cancer

        X, y = load_breast_cancer(return_X_y=True)

        miscoding = Miscoding()
        miscoding.fit(X, y)
        mscd = miscoding.miscoding_features()
    """

    def __init__(self, X_type="numeric", y_type="numeric", redundancy=False):
        """
        Initialization of the class Miscoding
        
        Parameters
        ----------
        X_type:     The type of the features, numeric, mixed or categorical
        y_type:     The type of the target, numeric or categorical
        redundancy: if "True" takes into account the redundancy between features
                    to compute the miscoding, if "False" only the miscoding with
                    respect to the target variable is computed.
          
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
        self.redundancy = redundancy
        
        return None
    
    
    def fit(self, X, y=None):
        """
        Learn empirically the miscoding of the features of X
        as a representation of y.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Sample vectors from which to compute miscoding.
            array-like, numpy or pandas array in case of numerical types
            pandas array in case of mixed or categorical types
            
        y : (optional) array-like, shape (n_samples)
            The target values as numbers or strings.
            
        Returns
        -------
        self
        """

        if self.X_type == "mixed" or self.X_type == "categorical":

            if isinstance(X, pd.DataFrame):
                self.X_isnumeric = [np.issubdtype(my_type, np.number) for my_type in X.dtypes]
                self.X_ = np.array(X)
            else:
                raise ValueError("Only DataFrame is allowed for X of type 'mixed' and 'categorical."
                                 "Got type {!r} instead."
                                 .format(type(X)))
                
        else:
            self.X_ = check_array(X)
            self.X_isnumeric = [True] * X.shape[1]

        if y is not None:

            self.y_ = column_or_1d(y)

            # Miscoding wrt the target is computed only if we have a target

            if self.y_type == "numeric":
                self.y_isnumeric = True
            else:
                self.y_isnumeric = False
        
            if self.redundancy:
                self.regular_ = self._miscoding_features_joint()
            else:
                self.regular_ = self._miscoding_features_single()

            self.adjusted_ = 1 - self.regular_

            if np.sum(self.adjusted_) != 0:
                self.adjusted_ = self.adjusted_ / np.sum(self.adjusted_)

            if np.sum(self.regular_) != 0:
                self.partial_  = self.adjusted_ - self.regular_ / np.sum(self.regular_)
            else:
                self.partial_  = self.adjusted_
        
        else:

            self.X_ = column_or_1d(X)
            self.y_isnumeric = self.X_isnumeric[0]

            self.regular_  = []
            self.adjusted_ = []
            self.partial_  = []

        return self


    def miscoding_features(self, mode='adjusted'):
        """
        Return the miscoding of the target given the features

        Parameters
        ----------
        mode  : the mode of miscoding, possible values are 'regular' for
                the true miscoding, 'adjusted' for the normalized inverted
                values, and 'partial' with positive and negative
                contributions to dataset miscoding.
            
        Returns
        -------
        Return a numpy array with the miscodings
        """
        
        check_is_fitted(self)
        
        if mode == 'regular':
            return self.regular_
        elif mode == 'adjusted':
            return self.adjusted_
        elif mode == 'partial':
            return self.partial_
        else:
            valid_modes = ('regular', 'adjusted', 'partial')
            raise ValueError("Valid options for 'mode' are {}. "
                             "Got mode={!r} instead."
                            .format(valid_modes, mode))

        return None 


    def miscoding_model(self, model, mode='partial'):
        """
        Compute the joint miscoding of the dataset used by a model
        
        Parameters
        ----------
        model : a model of one of the supported classes
        mode  : the mode of miscoding, possible values are 'regular' for
                the true miscoding, 'adjusted' for the normalized inverted
                values, and 'partial' with positive and negative
                contributions to dataset miscoding.
                    
        Returns
        -------
        Return the miscoding (float)
        """

        check_is_fitted(self)

        valid_modes = ('adjusted', 'partial')

        if mode not in valid_modes:
            raise ValueError("Valid options for 'mode' are {}. "
                             "Got mode={!r} instead."
                            .format(valid_modes, mode))

        if isinstance(model, MultinomialNB):
            subset = self._MultinomialNB(model)
        elif isinstance(model, DecisionTreeClassifier):
            subset = self._DecisionTreeClassifier(model)
        elif isinstance(model, SVC) and model.get_params()['kernel']=='linear':
            subset = self._LinearSVC(model)
        elif isinstance(model, SVC) and model.get_params()['kernel']=='poly':
            subset = self._SVC(model)
        elif isinstance(model, MLPClassifier):
            subset = self._MLPClassifier(model)
        elif isinstance(model, LinearRegression):
            subset = self._LinearRegression(model)
        elif isinstance(model, DecisionTreeRegressor):
            subset = self._DecisionTreeRegressor(model)
        elif isinstance(model, LinearSVR):
            subset = self._LinearSVR(model)
        elif isinstance(model, MLPRegressor):
            subset = self._MLPRegressor(model)            
        else:
            # Rise exception
            raise NotImplementedError('Model {!r} not supported'
                                     .format(type(model)))

        return self.miscoding_subset(subset, mode)
        

    def miscoding_subset(self, subset, mode='partial'):
        """
        Compute the joint miscoding of a subset of the features
        
        Parameters
        ----------
        subset : array-like, shape (n_features)
                 1 if the attribute is in use, 0 otherwise
        mode   : the mode of miscoding, possible values are 'adjusted' for
                 the normalized inverted values and 'partial' with positive
                 and negative contributions to dataset miscoding.                 
        
        Returns
        -------
        Return the miscoding (float)
        """

        valid_modes = ('adjusted', 'partial')

        check_is_fitted(self)

        if mode == 'adjusted':
            miscoding = 1 - np.dot(subset, self.adjusted_)

        elif mode == 'partial':
            # Avoid miscoding greater than 1
            top_mscd = 1 + np.sum(self.partial_[self.partial_ < 0])
            miscoding = top_mscd - np.dot(subset, self.partial_)
                
        else:
            raise ValueError("Valid options for 'mode' are {}. "
                             "Got mode={!r} instead."
                            .format(valid_modes, mode))

        # Avoid miscoding smaller than zero
        if miscoding < 0:
            miscoding = 0

        return miscoding


    def features_matrix(self, mode="adjusted"):
        """
        Compute a matrix of adjusted miscodings for the features

        Parameters
        ----------
        mode  : the mode of miscoding, possible values are 'regular' for the true
                miscoding and 'adjusted' for the normalized inverted values

        Returns
        -------
        Return the matrix (n_features x n_features) with the miscodings (float)
        """

        check_is_fitted(self)
        
        valid_modes = ('regular', 'adjusted')

        if mode not in valid_modes:
            raise ValueError("Valid options for 'mode' are {}. "
                             "Got mode={!r} instead."
                            .format(valid_modes, mode))

        miscoding = np.zeros([self.X_.shape[1], self.X_.shape[1]])

        # Compute the regular matrix

        for i in np.arange(self.X_.shape[1]-1):
            
            ldm_X1 = optimal_code_length(x1=self.X_[:,i], numeric1=self.X_isnumeric[i])

            for j in np.arange(i+1, self.X_.shape[1]):

                ldm_X2   = optimal_code_length(x1=self.X_[:,j], numeric1=self.X_isnumeric[j])
                ldm_X1X2 = optimal_code_length(x1=self.X_[:,i], numeric1=self.X_isnumeric[i], x2=self.X_[:,j], numeric2=self.X_isnumeric[j])
                       
                mscd = ( ldm_X1X2 - min(ldm_X1, ldm_X2) ) / max(ldm_X1, ldm_X2)
                
                miscoding[i, j] = mscd
                miscoding[j, i] = mscd

        if mode == "regular":
            return miscoding
                
        # Compute the normalized matrix
        
        normalized = np.zeros([self.X_.shape[1], self.X_.shape[1]])
        
        for i in np.arange(self.X_.shape[1]):

            normalized[i,:] = 1 - miscoding[i,:]
            normalized[i,:] = normalized[i,:] / np.sum(normalized[i,:])

        return normalized


    """
    Return the regular miscoding of the target given the features
            
    Returns
    -------
    Return a numpy array with the regular miscodings
    """
    def _miscoding_features_single(self):
                 
        miscoding = list()
                
        ldm_y = optimal_code_length(x1=self.y_, numeric1=self.y_isnumeric)

        for i in np.arange(self.X_.shape[1]):
                        
            ldm_X  = optimal_code_length(x1=self.X_[:,i], numeric1=self.X_isnumeric[i])
            ldm_Xy = optimal_code_length(x1=self.X_[:,i], numeric1=self.X_isnumeric[i], x2=self.y_, numeric2=self.y_isnumeric)
                       
            mscd = ( ldm_Xy - min(ldm_X, ldm_y) ) / max(ldm_X, ldm_y)
                
            miscoding.append(mscd)
                
        miscoding = np.array(miscoding)

        return miscoding


    """
    Return the joint regular miscoding of the target given pairs features
            
    Returns
    -------
    Return a numpy array with the regular miscodings
    """
    # TODO: Warning! Experimental implementation, do not use in production.
    def _miscoding_features_joint(self):

        # Compute non-redundant miscoding
        mscd = self._miscoding_features_single()

        if self.X_.shape[1] == 1:
            # With one single attribute we cannot compute the joint miscoding
            return mscd

        #
        # Compute the joint miscoding matrix
        #         
               
        red_matrix = np.ones([self.X_.shape[1], self.X_.shape[1]])

        ldm_y = optimal_code_length(x1=self.y_, numeric1=self.y_isnumeric)

        for i in np.arange(self.X_.shape[1]-1):
                        
            for j in np.arange(i+1, self.X_.shape[1]):
                
                ldm_X1X2  = optimal_code_length(x1=self.X_[:,i], numeric1=self.X_isnumeric[i], x2=self.X_[:,j], numeric2=self.X_isnumeric[j])
                ldm_X1X2Y = optimal_code_length(x1=self.X_[:,i], numeric1=self.X_isnumeric[i], x2=self.X_[:,j], numeric2=self.X_isnumeric[j], x3=self.y_, numeric3=self.y_isnumeric)
                
                tmp = ( ldm_X1X2Y - min(ldm_X1X2, ldm_y) ) / max(ldm_X1X2, ldm_y)
                                
                red_matrix[i, j] = tmp
                red_matrix[j, i] = tmp
        
        #
        # Compute the joint miscoding 
        #

        viu       = np.zeros(self.X_.shape[1], dtype=np.int8)
        miscoding = np.zeros(self.X_.shape[1])

        # Select the first two variables with smaller joint miscoding
        
        loc1, loc2 = np.unravel_index(np.argmin(red_matrix, axis=None), red_matrix.shape)
        jmscd1 = jmscd2 = red_matrix[loc1, loc2]
        
        viu[loc1] = 1
        viu[loc2] = 1

        # Scale down one of them
                
        tmp1 = mscd[loc1]
        tmp2 = mscd[loc2]
        
        if tmp1 < tmp2:
            jmscd1 = jmscd1 * tmp1 / tmp2
        elif tmp1 > tmp2:
            jmscd2 = jmscd2 * tmp2 / tmp1
        
        miscoding[loc1] = jmscd1
        miscoding[loc2] = jmscd2
 
        # Iterate over the number of features
        
        tmp = np.ones(self.X_.shape[1]) * np.inf
        
        for i in np.arange(2, self.X_.shape[1]):

            for j in np.arange(self.X_.shape[1]):
            
                if viu[j] == 1:
                    continue

                tmp[j] = (1 / np.sum(viu)) * np.sum(red_matrix[np.where(viu == 1), j])

            viu[np.argmin(tmp)] = 1
            miscoding[np.argmin(tmp)] = np.min(tmp)

            tmp = np.ones(self.X_.shape[1]) * np.inf
        
        return miscoding


    """
    Compute the attributes in use for a multinomial naive Bayes classifier
    
    Return array with the attributes in use
    """
    def _MultinomialNB(self, estimator):

        # All the attributes are in use
        attr_in_use = np.ones(self.X_.shape[1], dtype=int)
            
        return attr_in_use
    

    """
    Compute the attributes in use for a decision tree
    
    Return array with the attributes in use
    """
    def _DecisionTreeClassifier(self, estimator):

        attr_in_use = np.zeros(self.X_.shape[1], dtype=int)
        features = set(estimator.tree_.feature[estimator.tree_.feature >= 0])
        for i in features:
            attr_in_use[i] = 1
            
        return attr_in_use


    """
    Compute the attributes in use for a linear support vector classifier
    
    Return array with the attributes in use
    """
    def _LinearSVC(self, estimator):

        # All the attributes are in use
        attr_in_use = np.ones(self.X_.shape[1], dtype=int)
            
        return attr_in_use


    """
    Compute the attributes in use for a support vector classifier with a polynomial kernel
    
    Return array with the attributes in use
    """
    def _SVC(self, estimator):

        # All the attributes are in use
        attr_in_use = np.ones(self.X_.shape[1], dtype=int)
            
        return attr_in_use


    """
    Compute the attributes in use for a multilayer perceptron classifier
    
    Return array with the attributes in use
    """
    def _MLPClassifier(self, estimator):

        attr_in_use = np.ones(self.X_.shape[1], dtype=int)
            
        return attr_in_use


    """
    Compute the attributes in use for a linear regression
    
    Return array with the attributes in use
    """
    def _LinearRegression(self, estimator):
        
        attr_in_use = np.ones(self.X_.shape[1], dtype=int)
            
        return attr_in_use


    """
    Compute the attributes in use for a decision tree regressor
    
    Return array with the attributes in use
    """
    def _DecisionTreeRegressor(self, estimator):
        
        attr_in_use = np.zeros(self.X_.shape[1], dtype=int)
        features = set(estimator.tree_.feature[estimator.tree_.feature >= 0])
        for i in features:
            attr_in_use[i] = 1
            
        return attr_in_use


    """
    Compute the attributes in use for a linear support vector regressor
    
    Return array with the attributes in use
    """
    def _LinearSVR(self, estimator):

        attr_in_use = np.ones(self.X_.shape[1], dtype=int)
            
        return attr_in_use


    """
    Compute the attributes in use for a multilayer perceptron regressor
    
    Return array with the attributes in use
    """
    def _MLPRegressor(self, estimator):

        attr_in_use = np.ones(self.X_.shape[1], dtype=int)
            
        return attr_in_use
