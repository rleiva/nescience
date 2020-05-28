"""
Fast auto machine learning
with the minimum nescience principle

@author:    Rafael Garcia Leiva
@mail:      rgarcialeiva@gmail.com
@web:       http://www.mathematicsunknown.com/
@copyright: GNU GPLv3
@version:   0.7

"""

import numpy  as np

import warnings
import math
import re

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.feature_selection.base import SelectorMixin
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import KBinsDiscretizer

from scipy.optimize import differential_evolution

# Compressors

import bz2
import lzma
import zlib

# Supported classifiers

from sklearn.naive_bayes    import MultinomialNB
from sklearn.tree           import DecisionTreeClassifier
from sklearn.svm            import LinearSVC
from sklearn.neural_network import MLPClassifier

# Supported regressors

from sklearn.linear_model   import LinearRegression
from sklearn.tree           import DecisionTreeRegressor
from sklearn.svm            import LinearSVR
from sklearn.neural_network import MLPRegressor

# Supported time series
# 
# - Autoregression
# - Moving Average
# - Simple Exponential Smoothing

#
# Private Helper Functions
#

"""
Discretize the variable x if needed
    
Parameters
----------
x :     array-like, shape (n_samples)
        The variable to be discretized, if needed.
       
Returns
-------
A new discretized vector.
"""
def _discretize_vector(x):

    new_x = x.copy()

    # Optimal number of bins
    n_bins = int(np.sqrt(len(new_x)))
    
    # Correct the number of bins if it is too small
    if n_bins <= 1:
        n_bins = 2

    # Check if we have too many unique values wrt samples
    if len(np.unique(new_x)) > n_bins:
        
        new_x = new_x.reshape(-1, 1)
    
        # Avoid those annoying warnings
        with warnings.catch_warnings():
            
            warnings.simplefilter("ignore")

            est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans')
            est.fit(new_x)
            new_x = est.transform(new_x)
            
        new_x = new_x[:,0]

    return new_x


"""
Discretize a dataset X
    
Parameters
----------
X :     array-like, shape (n_samples, n_features)
       
Returns
-------
A new discretized version of the dataset (numpy array)
"""
def _discretize_matrix(X):

    newX = list()
    
    for i in np.arange(X.shape[1]):
            
        new_x = _discretize_vector(X[:,i])
        newX.append(new_x)    
    
    newX = np.array(newX)
    newX = np.transpose(newX)
    
    return newX


"""
Compute the lenght of a list of values encoded using an optimal code
    
Parameters
----------
x :         array-like, shape (n_samples)
            The values to be encoded.
       
Returns
-------
Return the length of the encoded dataset (float)
"""
def _optimal_code_length(x):
    
    # Discretize the variable x if needed
    new_x = _discretize_vector(x)
        
    # Compute the optimal length
    unique, count = np.unique(new_x, return_counts=True)
    ldm = np.sum(count * ( - np.log2(count / len(new_x))))
    
    return ldm


"""
Compute the joint lenght of two variables
encoded using an optimal code
    
Parameters
----------
x1 :        array-like, shape (n_samples)
            The values of the first variable.
x2 :        array-like, shape (n_samples)
            The values of the second variable.
   
Returns
-------
Return the length of the encoded joint dataset (float)    
"""
def _optimal_code_length_joint(x1, x2):
    
    # Discretize the variables X1 and X2 if needed
    new_x1 = _discretize_vector(x1)
    new_x2 = _discretize_vector(x2)
    
    # Compute the optimal length
    Joint =  list(zip(new_x1, new_x2))
    unique, count = np.unique(Joint, return_counts=True, axis=0)
    ldm = np.sum(count * ( - np.log2(count / len(Joint))))
                                    
    return ldm


"""
Compute the joint lenght of two variables and the target
encoded using an optimal code
    
Parameters
----------
x1 :        array-like, shape (n_samples)
            The values of the first variable.         
x2 :        array-like, shape (n_samples)
            The values of the second variable.         
y  :        array-like, shape (n_samples)
            The target values as numbers or strings.
   
Returns
-------
Return the length of the encoded joint dataset (float)    
"""
def _optimal_code_length_3joint(x1, x2, y):
  
    # Discretize the variables X1 and X2 if needed
    new_x1 = _discretize_vector(x1)
    new_x2 = _discretize_vector(x2)
    new_y  = _discretize_vector(y)
            
    # Compute the optimal length
    Joint =  list(zip(new_x1, new_x2, new_y))
    unique, count = np.unique(Joint, return_counts=True, axis=0)
    ldm = np.sum(count * ( - np.log2(count / len(Joint))))
                    
    return ldm


#
# Class Miscoding
# 

class Miscoding(BaseEstimator, SelectorMixin):
    """
    Given a dataset X = {x1, ..., xp} composed by p features, and a target
    variable y, the miscoding of the feature xj measures how difficult is to
    reconstruct y given xj, and the other way around. We are not only
    interested in to identify how much information xj contains about y, but
    also if xj contains additional information that is not related
    to y (which is a bad thing).

    The fastautoml.Miscoding class allow us to compute the relevance of
    features, the quality of a dataset, and select the optimal subset of
    features to include in a study

    Example of usage:
        
        from fastautoml.fastautoml import Miscoding
        miscoding = Miscoding()
        miscoding.fit(X, y)
        msd = miscoding.miscoding_features()

    """
    
    def __init__(self):
        
        return None
    
    
    def fit(self, X, y):
        """
        Learn empirically the miscoding of the features of X
        as a representation of y.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Sample vectors from which to compute miscoding.
            
        y : array-like, shape (n_samples)
            The target values as numbers or strings.
            
        Returns
        -------
        self
        """
        
        self.X, self.y = check_X_y(X, y, dtype=None)
        
        return self


    def miscoding_features(self, mode='adjusted', redundancy=True):
        """
        Return the miscoding of the target given the features

        Parameters
        ----------
        mode  : the mode of miscoding, possible values are 'regular' for
                the true miscoding, 'normalized' for normalized values that
                sum one, and 'partial' with positive and negative
                contritutions to dataset miscoding.
        redundancy: if True avoid redundant features during the
                    computation of miscoding
            
        Returns
        -------
        Return a numpy array with the miscodings
        """
        
        check_is_fitted(self, 'X')
        
        if redundancy:
            miscoding = self._miscoding_features_joint(mode)
        else:
            miscoding = self._miscoding_features_single(mode)
            
        return miscoding
            

    def miscoding_model(self, model):
        """
        Compute the partial joint miscoding of the dataset used by a model
        
        Parameters
        ----------
        model : a model of one of the supported classes
                    
        Returns
        -------
        Return the miscoding (float)
        """

        check_is_fitted(self, 'X')
        
        if isinstance(model, MultinomialNB):
            subset = self._MultinomialNB(model)
        elif isinstance(model, DecisionTreeClassifier):
            subset = self._DecisionTreeClassifier(model)
        elif isinstance(model, LinearSVC):
            subset = self._LinearSVC(model)
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

        return self.miscoding_subset(subset)
        

    def miscoding_subset(self, subset):
        """
        Compute the partial joint miscoding of a subset of the dataset
        
        Parameters
        ----------
        subset : array-like, shape (n_features)
                 1 if the attribute is in use, 0 otherwise
        
        Returns
        -------
        Return the miscoding (float)
        """

        check_is_fitted(self, 'X')

        partial = self.miscoding_features(mode='partial')

        miscoding = np.dot(subset, partial)
        miscoding = 1 - miscoding

        return miscoding


    def features_matrix(self):
        """
        Compute a matrix of adjusted miscodings of each feature
        assuming the others
        
        Returns
        -------
        Return the matrix with the miscodings (float)
        """
                
        miscoding = np.zeros([self.X.shape[1], self.X.shape[1]])

        # Compute the regular matrix

        for i in np.arange(self.X.shape[1]-1):
            
            ldm_X1 = _optimal_code_length(self.X[:,i])

            for j in np.arange(i+1, self.X.shape[1]):
                 
                ldm_X2   = _optimal_code_length(self.X[:,j])
                ldm_X1X2 = _optimal_code_length_joint(self.X[:,i], self.X[:,j])
                       
                mscd = ( ldm_X1X2 - min(ldm_X1, ldm_X2) ) / max(ldm_X1, ldm_X2)
                
                miscoding[i, j] = mscd
                miscoding[j, i] = mscd
                
        # Compute the normalized matrix
        
        normalized = np.zeros([self.X.shape[1], self.X.shape[1]])
        
        for i in np.arange(self.X.shape[1]):

            normalized[i,:] = 1 - miscoding[i,:]
            normalized[i,:] = normalized[i,:] / np.sum(normalized[i,:])

        return normalized


    # TODO: do we have to support this?
    def _get_support_mask(self):
        
        check_is_fitted(self, 'tcc_')
        
        return None


    """
    Return the miscoding of the target given the features

    Parameters
    ----------
    mode  : the mode of miscoding, possible values are 'regular' for
            the true miscoding, 'normalized' for normalized values that
            sum one, and 'partial' with positive and negative
            contritutions to dataset miscoding.
            
    Returns
    -------
    Return a numpy array with the miscodings
    """
    def _miscoding_features_single(self, mode='adjusted'):
                 
        miscoding = list()
        
        # Discretize y and compute the encoded length
        
        ldm_y = _optimal_code_length(self.y)

        for i in np.arange(self.X.shape[1]):

            # Discretize feature and compute lengths
            
            ldm_X  = _optimal_code_length(self.X[:,i])
            ldm_Xy = _optimal_code_length_joint(self.y, self.X[:,i])

            # Compute miscoding
                       
            mscd = ( ldm_Xy - min(ldm_X, ldm_y) ) / max(ldm_X, ldm_y)
                
            miscoding.append(mscd)
                
        miscoding = np.array(miscoding)

        if mode == 'regular':
            return miscoding
        elif mode == 'adjusted':
            adjusted = 1 - miscoding
            adjusted = adjusted / np.sum(adjusted)
            return adjusted
        elif mode == 'partial':
            adjusted = 1 - miscoding
            adjusted = adjusted / np.sum(adjusted)
            partial  = adjusted - miscoding / np.sum(miscoding)
            return partial

        valid_mode = ('regular', 'adjusted', 'partial')
        raise ValueError("Valid options for 'mode' are {}. "
                         "Got mode={!r} instead."
                         .format(valid_mode, mode))        


    """
    Return the joint redundancy of the target given pairs features

    Parameters
    ----------
    mode  : the mode of miscoding, possible values are 'regular' for
            the joint redundancy, 'normalized' for normalized values that
            sum one, and 'partial' with positive and negative
            contritutions to dataset joint redundancy.
            
    Returns
    -------
    Return a numpy array with the miscodings
    """
    def _miscoding_features_joint(self, mode='adjusted'):

        # Compute non-redundant miscoding
        mscd = self._miscoding_features_single(mode='regular')

        if self.X.shape[1] == 1:
            # With one single attribute we cannot compute the joint miscoding
            return mscd

        #
        # Compute the joint miscoding matrix
        #         
               
        red_matrix = np.ones([self.X.shape[1], self.X.shape[1]])

        ldm_y = _optimal_code_length(self.y)
        new_y = _discretize_vector(self.y)
        new_X = _discretize_matrix(self.X)

        for i in np.arange(self.X.shape[1]-1):
            
            new_x1 = new_X[:,i]

            for j in np.arange(i+1, self.X.shape[1]):
                
                new_x2 = new_X[:,j]
                
                Joint =  list(zip(new_x1, new_x2))
                unique, count = np.unique(Joint, return_counts=True, axis=0)
                ldm = np.sum(count * ( - np.log2(count / len(Joint))))
                
                ldm_X1X2  = ldm
                
                Joint =  list(zip(new_x1, new_x2, new_y))
                unique, count = np.unique(Joint, return_counts=True, axis=0)
                ldm = np.sum(count * ( - np.log2(count / len(Joint))))
                            
                ldm_X1X2Y = ldm
                       
                tmp = ( ldm_X1X2Y - min(ldm_X1X2, ldm_y) ) / max(ldm_X1X2, ldm_y)
                                
                red_matrix[i, j] = tmp
                red_matrix[j, i] = tmp
         
               
        #
        # Compute the joint miscoding 
        #
        
        viu       = np.zeros(self.X.shape[1], dtype=np.int8)
        miscoding = np.zeros(self.X.shape[1])

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
            miscoding[loc1] = jmscd1
            miscoding[loc2] = jmscd2
        else:
            jmscd2 = jmscd2 * tmp2 / tmp1
            miscoding[loc1] = jmscd1
            miscoding[loc2] = jmscd2
 
        # Iterate over the number of features
        
        tmp = np.ones(self.X.shape[1]) * np.inf
        
        for i in np.arange(2, self.X.shape[1]):

            for j in np.arange(self.X.shape[1]):
            
                if viu[j] == 1:
                    continue

                tmp[j] = (1 / np.sum(viu)) * np.sum(red_matrix[np.where(viu == 1), j])
                            
            viu[np.argmin(tmp)] = 1
            miscoding[np.argmin(tmp)] = np.min(tmp)

            tmp = np.ones(self.X.shape[1]) * np.inf

        if mode == 'regular':
            return miscoding
        elif mode == 'adjusted':
            adjusted = 1 - miscoding
            adjusted = adjusted / np.sum(adjusted)
            return adjusted
        elif mode == 'partial':
            adjusted = 1 - miscoding
            adjusted = adjusted / np.sum(adjusted)
            partial  = adjusted - miscoding / np.sum(miscoding)
            return partial

        valid_mode = ('regular', 'adjusted', 'partial')
        raise ValueError("Valid options for 'mode' are {}. "
                         "Got mode={!r} instead."
                         .format(valid_mode, mode))


    """
    Compute the attributes in use for a multinomial naive Bayes classifier
    
    Return array with the attributes in use
    """
    def _MultinomialNB(self, estimator):

        # All the attributes are in use
        attr_in_use = np.ones(self.X.shape[1], dtype=int)
            
        return attr_in_use
    

    """
    Compute the attributes in use for a decision tree
    
    Return array with the attributes in use
    """
    def _DecisionTreeClassifier(self, estimator):

        attr_in_use = np.zeros(self.X.shape[1], dtype=int)
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
        attr_in_use = np.ones(self.X.shape[1], dtype=int)
            
        return attr_in_use


    """
    Compute the attributes in use for a multilayer perceptron classifier
    
    Return array with the attributes in use
    """
    def _MLPClassifier(self, estimator):

        attr_in_use = np.ones(self.X.shape[1], dtype=int)
            
        return attr_in_use


    """
    Compute the attributes in use for a linear regression
    
    Return array with the attributes in use
    """
    def _LinearRegression(self, estimator):
        
        attr_in_use = np.ones(self.X.shape[1], dtype=int)
            
        return attr_in_use


    """
    Compute the attributes in use for a decision tree regressor
    
    Return array with the attributes in use
    """
    def _DecisionTreeRegressor(self, estimator):
        
        attr_in_use = np.zeros(self.X.shape[1], dtype=int)
        features = set(estimator.tree_.feature[estimator.tree_.feature >= 0])
        for i in features:
            attr_in_use[i] = 1
            
        return attr_in_use


    """
    Compute the attributes in use for a linear support vector regressor
    
    Return array with the attributes in use
    """
    def _LinearSVR(self, estimator):

        attr_in_use = np.ones(self.X.shape[1], dtype=int)
            
        return attr_in_use

    """
    Compute the attributes in use for a multilayer perceptron regressor
    
    Return array with the attributes in use
    """
    def _MLPRegressor(self, estimator):

        attr_in_use = np.ones(self.X.shape[1], dtype=int)
            
        return attr_in_use


#
# Class Inaccuracy
#
        
class Inaccuracy(BaseEstimator, SelectorMixin):
    
    # TODO: Class documentation
    
    def __init__(self):
        
        return None
    
    
    def fit(self, X, y):
        """Initialize the inaccuracy class with dataset
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Sample vectors from which models have been trained.
            
        y : array-like, shape (n_samples)
            The target values (class labels) as integers or strings.
            
        Returns
        -------
        self
        """
        
        self.X, self.y = check_X_y(X, y, dtype=None)
                
        self.len_y = _optimal_code_length(self.y)
        
        return self


    def inaccuracy_model(self, model):
        """
        Compute the inaccuracy of a model
        
        model : trained model with a predict() method
         
        Return the inaccuracy (float)
        """        
        
        check_is_fitted(self, 'X')
        
        Pred = model.predict(self.X)
        len_pred = _optimal_code_length(Pred)
        
        len_joint = _optimal_code_length_joint(Pred, self.y)
        
        inacc = ( len_joint - min(self.len_y, len_pred) ) / max(self.len_y, len_pred)

        return inacc

    
    def inaccuracy_predictions(self, predictions):
        """
        Compute the inaccuracy of a list of predicted values
        
         pred : array-like, shape (n_samples)
                The list of predicted values
                
        Return the inaccuracy (float)
        """        
        
        check_is_fitted(self, 'X')
        
        len_pred = _optimal_code_length(predictions)
        
        len_joint = _optimal_code_length_joint(predictions, self.y)
        
        inacc = ( len_joint - min(self.len_y, len_pred) ) / max(self.len_y, len_pred)

        return inacc    


    # TODO
    def _get_support_mask(self):
        
        check_is_fitted(self, 'tcc_')

 
#
# Class Surfeit
# 
    
class Surfeit(BaseEstimator, SelectorMixin):
    
    def __init__(self):
        
        return None
    

    def fit(self, X, y, compressor="bz2"):
        """Initialize the surfeit class with dataset
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Sample vectors from which models have been trained.
            
        y : array-like, shape (n_samples)
            The target values (class labels) as integers or strings.
            
        Returns
        -------
        self
        """

        self.compressor = compressor
        
        self.X, self.y = check_X_y(X, y, dtype=None)
                
        self.len_y = _optimal_code_length(self.y)
        
        return self
    

    def surfeit_model(self, model):
        """
        Compute the redundancy of a model

        Parameters
        ----------
        model : a model of one of the supported classeses
        
        Supported classes:
            MultinomialNB
            DecisionTreeClassifier
            MLPClassifier
            
        Returns
        -------
        Redundancy (float) of the model
        """
    
        if isinstance(model, MultinomialNB):
            model_str = self._MultinomialNB(model)
        elif isinstance(model, DecisionTreeClassifier):
            model_str = self._DecisionTreeClassifier(model)
        elif isinstance(model, LinearSVC):
            model_str = self._LinearSVC(model)
        elif isinstance(model, MLPClassifier):
            model_str = self._MLPClassifier(model)
        elif isinstance(model, LinearRegression):
            model_str = self._LinearRegression(model)
        elif isinstance(model, DecisionTreeRegressor):
            model_str = self._DecisionTreeRegressor(model)
        elif isinstance(model, LinearSVR):
            model_str = self._LinearSVR(model)
        elif isinstance(model, MLPRegressor):
            model_str = self._MLPRegressor(model)
        else:
            # Rise exception
            raise NotImplementedError('Model {!r} not supported'
                                     .format(type(model)))

        return self.surfeit_string(model_str)
        

    def surfeit_string(self, model_string):
        """
        Compute the redundancy of a model given as a string

        Parameters
        ----------
        model : a string based representation of the model
            
        Returns
        -------
        Redundancy (float) of the model
        """
    
        # Compute the model string and its compressed version
        emodel = model_string.encode()
        
        if self.compressor == "lzma":
            compressed = lzma.compress(emodel, preset=9)
        elif self.compressor == "zlib":
            compressed = zlib.compress(emodel, level=9)
        else: # By default use bz2
            compressed = bz2.compress(emodel, compresslevel=9)
        
        km = len(compressed)
        lm = len(emodel)

        # Check if the model is too small to compress        
        if km > lm:
            return 1 - 3/4    # Experimental value

        if self.len_y < km:
            # redundancy = 1 - l(C(y)) / l(m)
            redundancy = 1 - self.len_y / lm
        else:
            # redundancy = 1 - l(m*) / l(m)
            redundancy = 1 - km / lm
                            
        return redundancy


    def _redundancy(self):

        # Compute the model string and its compressed version
        model = self._tree2str().encode()

        if self.compressor == "lzma":
            compressed = lzma.compress(model, preset=9)
        elif self.compressor == "zlib":
            compressed = zlib.compress(model, level=9)
        else: # By default use bz2
            compressed = bz2.compress(model, compresslevel=9)

        # Check if the model is too small to compress
        if len(compressed) > len(model):
            return 1 - 3/4    # Experimental values for bzip

        if self.lcd < len(compressed):
            # redundancy = 1 - l(C(y)) / l(m)
            redundancy = 1 - self.lcd / len(model)
        else:
            # redundancy = 1 - l(m*) / l(m)
            redundancy = 1 - len(compressed) / len(model)

        return redundancy

    
    # TODO
    def _get_support_mask(self):
        
        check_is_fitted(self, 'tcc_')
        
        return None
    

    """
    Convert a MultinomialNB classifier into a string
    """
    def _MultinomialNB(self, estimator):
  
        
        #
        # Discretize probabilities
        #

        py    = _discretize_vector(np.exp(estimator.class_log_prior_))
        
        theta = np.exp(estimator.feature_log_prob_)
        theta = theta.flatten()
        theta = _discretize_vector(theta)
        theta = np.array(theta)
        theta = theta.reshape(estimator.feature_log_prob_.shape)
        
        #
        # Create the model
        #

        # Header
        string = "def Bayes(X):\n"
 
        # Target probabilities
        string = string + "    Py["
        for i in np.arange(len(py)):
            string = string + str(py) + ", "
        string = string + "]\n"
            
        # Conditional probabilities
        string = string + "    theta["        
        for i in np.arange(len(theta)):
            string = string + str(theta[i]) + ", "
        string = string + "]\n"

        string = string + "    y_hat    = None\n"
        string = string + "    max_prob = 0\n"
        string = string + "    for i in range(len(estimator.classes_)):\n"
        string = string + "        prob = 1\n"
        string = string + "        for j in range(len(theta[i])):\n"
        string = string + "            prob = prob * theta[i][j]\n"
        string = string + "        prob = py[i] *  prob\n"
        string = string + "        if prob > max_prob:\n"
        string = string + "            y_hat = estimator.classes_[i]\n"
        string = string + "    return y_hat\n"
                
        return string
    
    
    """
    Convert a LinearSVC classifier into a string
    TODO: Review
    """
    def _LinearSVC(self, estimator):
  
        #
        # Discretize similarities
        #
        
        M = estimator.coef_
        M = M.flatten()
        M = _discretize_vector(M)
        M = np.array(M)
        M = M.reshape(estimator.coef_.shape)
        
        #
        # Create the model
        #

        # Header
        string = "def LinearSVC(X):\n"
             
        # Similarities
        string = string + "    M["        
        for i in np.arange(len(M)):
            string = string + str(M[i]) + ", "
        string = string + "]\n"

        string = string + "    y_hat    = None\n"
        string = string + "    max_prob = 0\n"
        string = string + "    for i in range(len(estimator.classes_)):\n"
        string = string + "        prob = 1\n"
        string = string + "        for j in range(len(M[i])):\n"
        string = string + "            prob = prob * M[i][j]\n"
        string = string + "        prob = py[i] *  prob\n"
        string = string + "        if prob > max_prob:\n"
        string = string + "            y_hat = estimator.classes_[i]\n"
        string = string + "    return y_hat\n"
                
        return string


    """
    Helper function to recursively compute the body of a DecisionTreeClassifier
    """
    def _treebody2str(self, estimator, node_id, depth):

        children_left  = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        feature        = estimator.tree_.feature
        threshold      = estimator.tree_.threshold
        
        my_string = ""
        
        if children_left[node_id] == children_right[node_id]:
            
            # It is a leaf
            my_string = my_string + '%sreturn %s\n' % (' '*depth*4, estimator.classes_[np.argmax(estimator.tree_.value[node_id][0])])

        else:

            # Print the decision to take at this level
            my_string = my_string + '%sif X%d < %.3f:\n' % (' '*depth*4, (feature[node_id]+1), threshold[node_id])
            my_string = my_string + self._treebody2str(estimator, children_left[node_id],  depth+1)
            my_string = my_string + '%selse:\n' % (' '*depth*4)
            my_string = my_string + self._treebody2str(estimator, children_right[node_id], depth+1)
                
        return my_string


    """
    Convert a DecisionTreeClassifier into a string
    """
    def _DecisionTreeClassifier(self, estimator):

        # TODO: sanity check over estimator
        
        n_nodes        = estimator.tree_.node_count
        children_left  = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        feature        = estimator.tree_.feature

        tree_string = ""
        
        #
        # Compute the tree header
        #
        
        features_set = set()
                
        for node_id in range(n_nodes):

            # If we have a test node
            if (children_left[node_id] != children_right[node_id]):
                features_set.add('X%d' % (feature[node_id]+1))
        
        tree_string = tree_string + "def tree" + str(features_set) + ":\n"

        #
        # Compute the tree body
        # 
        
        tree_string = tree_string + self._treebody2str(estimator, 0, 1)

        return tree_string


    """
    Convert a MLPClassifier into a string
    """
    def _MLPClassifier(self, estimator):
        
        # TODO: sanity check over estimator
        
        # TODO: Computation code should be optimized
        
        # TODO: Provide support to other activation functions
                
        #
        # Discretize coeficients
        #
        
        annw = []        
        for layer in estimator.coefs_:
            for node in layer:
                for coef in node:
                    annw.append(coef)

        annw = np.array(annw)
        annw = _discretize_vector(annw)
        
        ind  = 0
        coefs = list()
        for i in np.arange(len(estimator.coefs_)):
            layer = list()
            for j in np.arange(len(estimator.coefs_[i])):
                node = list()
                for k in np.arange(len(estimator.coefs_[i][j])):
                    node.append(annw[ind])
                    ind = ind + 1
                layer.append(node)
            coefs.append(layer)
            
        #
        # Discretize intercepts
        #
                    
        annb = []
        for layer in estimator.intercepts_:
            for node in layer:
                annb.append(node)

        annb = np.array(annb)
        annb = _discretize_vector(annb)
        
        ind  = 0
        inters = list()
        for i in np.arange(len(estimator.intercepts_)):
            layer = list()
            for j in np.arange(len(estimator.intercepts_[i])):
                layer.append(annb[ind])
                ind = ind + 1
            inters.append(layer)
                    
        #
        # Create the model
        #

        # Header
        string = "def NN(X):\n"
 
        # Weights
        string = string + "    W["
        for i in np.arange(len(coefs)):
            string = string + str(coefs[i]) + ", "
        string = string + "]\n"
            
        # Bias
        string = string + "    b["        
        for i in np.arange(len(coefs)):
            string = string + str(inters[i]) + ", "
        string = string + "]\n"
       
        # First layer
        
        string = string + "    Z = [0] * W[0].shape[0]\n"
        string = string + "    for i in range(W[0].shape[0]):\n"
        string = string + "        for j in range(W[0].shape[1]):\n"
        string = string + "            Z[i] = Z[i] + W[0, i, j] * X[j]\n"
        string = string + "        Z[i] = Z[i] + b[0][i] \n"
            
        string = string + "    A = [0] * W[0].shape[0]\n"
        string = string + "    for i in range(Z.shape[0]):\n"
        string = string + "        A[i] = max(Z[i], 0)\n"
        
        # Hiddent layers
        
        string = string + "    for i in range(1, " + str(len(estimator.coefs_)) + "):\n"
            
        string = string + "        Z = [0] * W[i].shape[0]\n"
        string = string + "        for j in range(W[i].shape[0]):\n"
        string = string + "            for k in range(W[i].shape[1]):\n"
        string = string + "                Z[j] = Z[j] + W[i, j, k] * A[k]\n"
        string = string + "            Z[j] = Z[j] + b[i][j] \n"
            
        string = string + "        A = [0] * W[i].shape[0]\n"
        string = string + "        for j in range(Z.shape[0]):\n"
        string = string + "            A = max(Z[j], 0)\n"
        
        # Predictions
        
        string = string + "    softmax = 0\n"
        string = string + "    prediction = 0\n"
        string = string + "    totalmax = 0\n"
        string = string + "    for i in range(A.shape[0]):\n"
        string = string + "        totalmax = totalmax + exp(A[i])\n"
        string = string + "    for i in range(A.shape[0]):\n"
        string = string + "        newmax = exp(A[i])\n"        
        string = string + "        if newmax > softmax:\n"        
        string = string + "            softmax = newmax \n"
        string = string + "            prediction = i\n"
        
        string = string + "    return prediction\n"

        return string
    

    """
    Convert a LinearRegression into a string
    """
    def _LinearRegression(self, estimator):

        #
        # Retrieve weigths
        #
                
        coefs     = estimator.coef_
        intercept = estimator.intercept_
        
        # Header
        string = "def LinearRegression(X):\n"
             
        # Similarities
        string = string + "    W = ["        
        for i in np.arange(len(coefs)):
            string = string + str(coefs[i]) + ", "
        string = string + "]\n"
        string = string + "    b = "
        string = string + str(intercept) + "\n"
            
        string = string + "    y_hat    = 0\n"
        string = string + "    for i in range(len(W)):\n"
        string = string + "        y_hat = W[i] * X[i]\n"
        string = string + "    y_hat = y_hat + b\n"        
        string = string + "    return y_hat\n"
                
        return string


    """
    Helper function to recursively compute the body of a DecisionTreeRegressor
    """
    def _treeregressorbody2str(self, estimator, node_id, depth):

        children_left  = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        feature        = estimator.tree_.feature
        threshold      = estimator.tree_.threshold
        
        my_string = ""
        
        if children_left[node_id] == children_right[node_id]:
            
            # It is a leaf
            my_string = my_string + '%sreturn %s\n' % (' '*depth*4, np.argmax(estimator.tree_.value[node_id][0]))

        else:

            # Print the decision to take at this level
            my_string = my_string + '%sif X%d < %.3f:\n' % (' '*depth*4, (feature[node_id]+1), threshold[node_id])
            my_string = my_string + self._treeregressorbody2str(estimator, children_left[node_id],  depth+1)
            my_string = my_string + '%selse:\n' % (' '*depth*4)
            my_string = my_string + self._treeregressorbody2str(estimator, children_right[node_id], depth+1)
            
        return my_string


    """
    Convert a LinearSVR into a string
    """
    def _LinearSVR(self, estimator):
        
        # TODO: Adapt to LinearSVR

        #
        # Discretize similarities
        #
        
        M = estimator.coef_
        M = M.flatten()
        M = _discretize_vector(M)
        M = np.array(M)
        M = M.reshape(estimator.coef_.shape)
        
        #
        # Create the model
        #

        # Header
        string = "def LinearSVC(X):\n"
             
        # Similarities
        string = string + "    M["        
        for i in np.arange(len(M)):
            string = string + str(M[i]) + ", "
        string = string + "]\n"

        string = string + "    y_hat    = None\n"
        string = string + "    max_prob = 0\n"
        string = string + "    for i in range(len(estimator.classes_)):\n"
        string = string + "        prob = 1\n"
        string = string + "        for j in range(len(M[i])):\n"
        string = string + "            prob = prob * M[i][j]\n"
        string = string + "        prob = py[i] *  prob\n"
        string = string + "        if prob > max_prob:\n"
        string = string + "            y_hat = estimator.classes_[i]\n"
        string = string + "    return y_hat\n"

        return string

    """
    Convert a DecisionTreeRegressor into a string
    """
    def _DecisionTreeRegressor(self, estimator):
        
        # TODO: sanity check over estimator
        
        n_nodes        = estimator.tree_.node_count
        children_left  = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        feature        = estimator.tree_.feature

        tree_string = ""
        
        #
        # Compute the tree header
        #
        
        features_set = set()
                
        for node_id in range(n_nodes):

            # If we have a test node
            if (children_left[node_id] != children_right[node_id]):
                features_set.add('X%d' % (feature[node_id]+1))
        
        tree_string = tree_string + "def DecisionTreeRegressor" + str(features_set) + ":\n"

        #
        # Compute the tree body
        # 
        
        tree_string = tree_string + self._treeregressorbody2str(estimator, 0, 1)

        return tree_string

        
    """
    Convert a MLPRegressor into a string
    """
    def _MLPRegressor(self, estimator):
        
        # TODO: Adapt to MLPRegressor
    
        # TODO: sanity check over estimator
        
        # TODO: Computation code should be optimized
        
        # TODO: Provide support to other activation functions
                
        #
        # Discretize coeficients
        #
        
        annw = []        
        for layer in estimator.coefs_:
            for node in layer:
                for coef in node:
                    annw.append(coef)

        annw = np.array(annw)
        annw = _discretize_vector(annw)
        
        ind  = 0
        coefs = list()
        for i in np.arange(len(estimator.coefs_)):
            layer = list()
            for j in np.arange(len(estimator.coefs_[i])):
                node = list()
                for k in np.arange(len(estimator.coefs_[i][j])):
                    node.append(annw[ind])
                    ind = ind + 1
                layer.append(node)
            coefs.append(layer)
            
        #
        # Discretize intercepts
        #
                    
        annb = []
        for layer in estimator.intercepts_:
            for node in layer:
                annb.append(node)

        annb = np.array(annb)
        annb = _discretize_vector(annb)
        
        ind  = 0
        inters = list()
        for i in np.arange(len(estimator.intercepts_)):
            layer = list()
            for j in np.arange(len(estimator.intercepts_[i])):
                layer.append(annb[ind])
                ind = ind + 1
            inters.append(layer)
                    
        #
        # Create the model
        #

        # Header
        string = "def NN(X):\n"
 
        # Weights
        string = string + "    W["
        for i in np.arange(len(coefs)):
            string = string + str(coefs[i]) + ", "
        string = string + "]\n"
            
        # Bias
        string = string + "    b["        
        for i in np.arange(len(coefs)):
            string = string + str(inters[i]) + ", "
        string = string + "]\n"
       
        # First layer
        
        string = string + "    Z = [0] * W[0].shape[0]\n"
        string = string + "    for i in range(W[0].shape[0]):\n"
        string = string + "        for j in range(W[0].shape[1]):\n"
        string = string + "            Z[i] = Z[i] + W[0, i, j] * X[j]\n"
        string = string + "        Z[i] = Z[i] + b[0][i] \n"
            
        string = string + "    A = [0] * W[0].shape[0]\n"
        string = string + "    for i in range(Z.shape[0]):\n"
        string = string + "        A[i] = max(Z[i], 0)\n"
        
        # Hiddent layers
        
        string = string + "    for i in range(1, " + str(len(estimator.coefs_)) + "):\n"
            
        string = string + "        Z = [0] * W[i].shape[0]\n"
        string = string + "        for j in range(W[i].shape[0]):\n"
        string = string + "            for k in range(W[i].shape[1]):\n"
        string = string + "                Z[j] = Z[j] + W[i, j, k] * A[k]\n"
        string = string + "            Z[j] = Z[j] + b[i][j] \n"
            
        string = string + "        A = [0] * W[i].shape[0]\n"
        string = string + "        for j in range(Z.shape[0]):\n"
        string = string + "            A = max(Z[j], 0)\n"
        
        # Predictions
        
        string = string + "    softmax = 0\n"
        string = string + "    prediction = 0\n"
        string = string + "    totalmax = 0\n"
        string = string + "    for i in range(A.shape[0]):\n"
        string = string + "        totalmax = totalmax + exp(A[i])\n"
        string = string + "    for i in range(A.shape[0]):\n"
        string = string + "        newmax = exp(A[i])\n"        
        string = string + "        if newmax > softmax:\n"        
        string = string + "            softmax = newmax \n"
        string = string + "            prediction = i\n"
        
        string = string + "    return prediction\n"

        return string        
    
        
class Nescience(BaseEstimator, SelectorMixin):
    
    def __init__(self):
        
        return None
    
    
    def fit(self, X, y, method="Arithmetic", compressor="bz2"):
        """
        Initialization of the class nescience
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Sample vectors from which to compute miscoding.
            
        y : array-like, shape (n_samples)
            The target values (class labels) as numbers or strings.

        method (string):     method used to comput the nescience. Valid
                             values are: "Euclid", "Arithmetic",
                             "Geometric", "Product", "Addition" and
                             "Harmonic".
                             
        compressor (string): compressor used to compute redudancy. Valid
                             values are: "bz2", "lzma" and "zlib".
          
        """
        
        self.method       = method
        self.compressor   = compressor

        self.X, self.y = check_X_y(X, y, dtype=None)

        self.miscoding  = Miscoding()
        self.inaccuracy = Inaccuracy()
        self.surfeit    = Surfeit()
        
        self.miscoding.fit(X, y)
        self.inaccuracy.fit(X, y)
        self.surfeit.fit(X, y, self.compressor)
        
        return self


    def nescience(self, model, subset=None, predictions=None, model_string=None):
        """
        Compute the nescience of a model
        
        Parameters
        ----------
        model       : a trained model

        subset      : array-like, shape (n_features)
                      1 if the attribute is in use, 0 otherwise
                      If None, attributes will be infrerred throught model
                      
        model_str   : a string based representation of the model
                      If None, string will be derived from model
                    
        Returns
        -------
        Return the nescience (float)
        """
        
        check_is_fitted(self, 'X')

        if subset is None:
            miscoding = self.miscoding.miscoding_model(model)
        else:
            miscoding = self.miscoding.miscoding_subset(subset)

        if predictions is None:
            inaccuracy = self.inaccuracy.inaccuracy_model(model)
        else:
            inaccuracy = self.inaccuracy.inaccuracy_predictions(predictions)
            
        if model_string is None:
            surfeit = self.surfeit.surfeit_model(model)
        else:
            surfeit = self.surfeit.surfeit_string(model_string)            

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
        elif self.method == "Weighted":
            # Weigthed sum
            # TODO: Not yet supported
            nescience = self.weight_miscoding * miscoding + self.weight_inaccuracy * inaccuracy + self.weight_surfeit * surfeit
        elif self.method == "Harmonic":
            # Harmonic mean
            nescience = 3 / ( (1/miscoding) + (1/inaccuracy) + (1/surfeit))
        # else -> rise exception
                
        return nescience

    
    # TODO
    def _get_support_mask(self):
                
        return None


class AutoClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self):
        
        return None

    
    def fit(self, X, y, auto=True):
        
        # TODO: document
        
        # Supported Classifiers
        
        self._classifiers = [
            self._MultinomialNB,
            self._DecisionTreeClassifier,
            self._LinearSVC,
            self._MLPClassifier
        ]

        self._X, self._y = check_X_y(X, y, dtype="numeric")

        self._nescience = Nescience()
        self._nescience.fit(self._X, self._y)
        
        nsc = 1
        self._model = None
        self._viu   = None
        self.classes_ = None

        # Find optimal model
        if auto:
        
            for clf in self._classifiers:
            
                # TODO: print classifier if verbose
                
                # If X contains negative values, MultinomialNB is skipped
                if clf == self._MultinomialNB and not (self._X>=0).all():
                    continue
                
                (new_nsc, new_model, new_viu) = clf()
                        
                if new_nsc < nsc:
                    nsc   = new_nsc
                    self._model = new_model
                    self._viu   = new_viu
        self.classes_ = self._model.classes_
        return self


    def predict(self, X):
        """
        Predict class given a dataset
    
          * X = list([[x11, x12, x13, ...], ..., [xn1, xn2, ..., xnm]])
    
        Return a list of classes predicted
        """
        
        # TODO: Check that we have a model trained
        
        check_is_fitted(self, '_X')
        
        if self._viu is None:
            msdX = X
        else:
            msdX = X[:,np.where(self._viu)[0]]
                
        return self._model.predict(msdX)


    def predict_proba(self, X):
        """
        Predict the probability of being in a class given a dataset
    
          * X = list([[x11, x12, x13, ...], ..., [xn1, xn2, ..., xnm]])
      
        Return an array of probabilities. The order of the list match the order
        the internal attribute classes_
        """
        
        # TODO: Check that we have a model trained
        
        check_is_fitted(self, '_X')

        if self._viu is None:
            msdX = X
        else:
            msdX = X[:,np.where(self._viu)[0]]
                    
        return self._model.predict_proba(msdX)


    def score(self, X, y):
        """
        Evaluate the performance of the current model given a test dataset

           * X = list([[x11, x12, x13, ...], ..., [xn1, xn2, ..., xnm]])
           * y = list([y1, ..., yn])
    
        Return one minus the mean error
        """
        
        # TODO: Check that we have a model trained
        
        check_is_fitted(self, '_X')
        
        if self._viu is None:
            msdX = X
        else:
            msdX = X[:,np.where(self._viu)[0]]        
        
        return self._model.score(msdX, y)


    def _MultinomialNB(self):
        
        # No hyperparameters to optimize
        
        model = MultinomialNB()
        model.fit(self._X, self._y)

        nsc = self._nescience.nescience(model)
            
        return (nsc, model, None)

    
    def _LinearSVC(self):
        
        # No hyperparameters to optimize
                    
        model = LinearSVC(multi_class="crammer_singer")
        model.fit(self._X, self._y)

        nsc = self._nescience.nescience(model)
            
        return (nsc, model, None)    


    def _DecisionTreeClassifier(self):
        
        depth   = 3
        nsc     = 1
        new_nsc = 0.99
        
        while new_nsc < nsc:

            nsc = new_nsc
            
            model = DecisionTreeClassifier(max_depth=depth)
            model.fit(self._X, self._y)

            new_nsc = self._nescience.nescience(model)
            
            depth = depth + 1

        return (nsc, model, None)
    
    
    def _MLPClassifier(self):
        
        # Relevance of features
        tmp_msd = msd = self._nescience.miscoding.miscoding_features()
        
        # Variables in use
        tmp_viu = viu = np.zeros(self._X.shape[1], dtype=np.int)

        # Create the initial neural network
        #  - two features
        #  - one hidden layer
        #  - three units
        
        tmp_hu = hu = [3]

        # Select the two most relevant features
        viu[np.argmax(msd)] = 1        
        msd[np.where(viu)] = -1
        viu[np.argmax(msd)] = 1
        msd[np.where(viu)] = -1
        
        msdX = self._X[:,np.where(viu)[0]]
        tmp_nn = nn = MLPClassifier(hidden_layer_sizes = hu)
        nn.fit(msdX, self._y)
        prd  = nn.predict(msdX)
        tmp_nsc = nsc = self._nescience.nescience(nn, subset=viu, predictions=prd)
        
        # While the nescience decreases
        decreased = True        
        while (decreased):
            
            decreased = False

            #
            # Test adding a new feature  
            #
            
            # Check if therer are still more variables to add
            if np.sum(viu) != viu.shape[0]:
            
                new_msd = msd.copy()
                new_viu = viu.copy()
            
                new_viu[np.argmax(new_msd)] = 1
                new_msd[np.where(viu)] = -1

                msdX    = self._X[:,np.where(new_viu)[0]]
                new_nn  = MLPClassifier(hidden_layer_sizes = hu)        
                new_nn.fit(msdX, self._y)
                prd     = new_nn.predict(msdX)
                new_nsc = self._nescience.nescience(new_nn, subset=new_viu, predictions=prd)
            
                # Save data if nescience has been reduced                        
                if new_nsc < tmp_nsc:                                
                    decreased = True
                    tmp_nn  = new_nn
                    tmp_nsc = new_nsc
                    tmp_msd = new_msd
                    tmp_viu = new_viu
                    tmp_hu  = hu
                    
            #
            # Test adding a new layer
            #
            
            new_hu = hu.copy()
            new_hu.append(3)

            msdX    = self._X[:,np.where(viu)[0]]
            new_nn  = MLPClassifier(hidden_layer_sizes = new_hu)
            new_nn.fit(msdX, self._y)
            prd     = new_nn.predict(msdX)
            new_nsc = self._nescience.nescience(new_nn, subset=new_viu, predictions=prd)
            
            # Save data if nescience has been reduced 
            if new_nsc < tmp_nsc:                                
                decreased = True
                tmp_nn  = new_nn
                tmp_nsc = new_nsc
                tmp_msd = msd
                tmp_viu = viu
                tmp_hu  = new_hu

            #
            # Test adding a new unit
            #
            
            for i in np.arange(len(hu)):
                
                new_hu    = hu.copy()
                new_hu[i] = new_hu[i] + 1            

                msdX    = self._X[:,np.where(viu)[0]]
                new_nn  = MLPClassifier(hidden_layer_sizes = new_hu)
                new_nn.fit(msdX, self._y)
                prd     = new_nn.predict(msdX)
                new_nsc = self._nescience.nescience(new_nn, subset=viu, predictions=prd)
            
                # Save data if nescience has been reduced                        
            if new_nsc < tmp_nsc:                                
                decreased = True
                tmp_nn  = new_nn
                tmp_nsc = new_nsc
                tmp_msd = msd
                tmp_viu = viu
                tmp_hu  = new_hu
                
            # Update neural network
            nn      = tmp_nn
            nsc     = tmp_nsc
            viu     = tmp_viu
            msd     = tmp_msd
            hu      = tmp_hu

        # -> end while

        return (nsc, nn, viu)


class AutoRegressor(BaseEstimator, RegressorMixin):
    
    # TODO: Class documentation

    def __init__(self):
        
        return None

    
    def fit(self, X, y, auto=True):
        """
        Select the best model that explains y given X.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Sample vectors from which to compute miscoding.
            
        y : array-like, shape (n_samples)
            The target values (class labels) as numbers or strings.
        
        auto: find automatically the optimal model
            
        Returns
        -------
        self
        """
        
        # Supported Regressors
        
        self._regressors = [
            self._LinearRegression,
            self._LinearSVR,
            self._DecisionTreeRegressor,
            self._MLPRegressor
        ]

        self._X, self._y = check_X_y(X, y, dtype="numeric")

        self._nescience = Nescience()
        self._nescience.fit(self._X, self._y)
        
        nsc = 1
        self._model = None
        self._viu   = None
        
        # Find automatically the optimal model
        
        if auto:
            
            for reg in self._regressors:
            
                # TODO: Should be based on a verbose flag
                print("Regressor: " + str(reg))
            
                (new_nsc, new_model, new_viu) = reg()
            
                if new_nsc < nsc:
                    nsc   = new_nsc
                    self._model = new_model
                    self._viu   = new_viu
        
        return self


    def predict(self, X):
        """
        Predict class given a dataset
    
          * X = list([[x11, x12, x13, ...], ..., [xn1, xn2, ..., xnm]])
    
        Return the predicted value
        """
        
        # TODO: Check that we have a model trained
        
        check_is_fitted(self, '_X')
        
        if self._viu is None:
            msdX = X
        else:
            msdX = X[:,np.where(self._viu)[0]]
                
        return self._model.predict(msdX)


    def score(self, X, y):
        """
        Evaluate the performance of the current model given a test dataset

           * X = list([[x11, x12, x13, ...], ..., [xn1, xn2, ..., xnm]])
           * y = list([y1, ..., yn])
    
        Return one minus the mean error
        """
        
        # TODO: Check that we have a model trained
        
        check_is_fitted(self, '_X')
        
        if self._viu is None:
            msdX = X
        else:
            msdX = X[:,np.where(self._viu)[0]]        
        
        return self._model.score(msdX, y)


    def _LinearRegression(self):
        
        # Relevance of features
        msd = self._nescience.miscoding.miscoding_features()
        
        # Variables in use
        viu = np.zeros(self._X.shape[1], dtype=np.int)

        # Select the the most relevant feature
        viu[np.argmax(msd)] = 1        
        msd[np.where(viu)] = -1

        # Evaluate the model        
        msdX = self._X[:,np.where(viu)[0]]        
        model = LinearRegression()
        model.fit(msdX, self._y)
        
        prd  = model.predict(msdX)
        nsc = self._nescience.nescience(model, subset=viu, predictions=prd)
        
        decreased = True
        while (decreased):
                        
            decreased = False
            
            new_msd = msd.copy()
            new_viu = viu.copy()
            
            # Select the the most relevant feature
            new_viu[np.argmax(new_msd)] = 1        
            new_msd[np.where(new_viu)] = -1

            # Evaluate the model        
            msdX = self._X[:,np.where(new_viu)[0]]        
            new_model = LinearRegression()
            new_model.fit(msdX, self._y)        
            
            prd  = new_model.predict(msdX)
            new_nsc = self._nescience.nescience(new_model, subset=new_viu, predictions=prd)
            
            # Save data if nescience has been reduced                        
            if new_nsc < nsc:                                
                decreased = True
                model     = new_model
                nsc       = new_nsc
                msd       = new_msd
                viu       = new_viu
        
        return (nsc, model, viu)


    def _LinearSVR(self):
        # TODO: Optimize hyperparameters
                    
        # model = LinearSVR(multi_class="crammer_singer")
        model = LinearSVR()
        model.fit(self._X, self._y)

        nsc = self._nescience.nescience(model)
            
        return (nsc, model, None)    


    def _DecisionTreeRegressor(self):
        
        depth   = 3
        nsc     = 1
        new_nsc = 0.99
        
        while new_nsc < nsc:

            nsc = new_nsc
                        
            model = DecisionTreeRegressor(max_depth=depth)
            model.fit(self._X, self._y)

            new_nsc = self._nescience.nescience(model)
            
            depth = depth + 1

        return (nsc, model, None)


    def _MLPRegressor(self):
        
        # Relevance of features
        tmp_msd = msd = self._nescience.miscoding.miscoding_features()
        
        # Variables in use
        tmp_viu = viu = np.zeros(self._X.shape[1], dtype=np.int)

        # Create the initial neural network
        #  - two features
        #  - one hidden layer
        #  - three units
        
        tmp_hu = hu = [3]

        # Select the two most relevant features
        viu[np.argmax(msd)] =  1        
        msd[np.where(viu)]  = -1
        viu[np.argmax(msd)] =  1
        msd[np.where(viu)]  = -1
        
        msdX = self._X[:,np.where(viu)[0]]
        tmp_nn = nn = MLPRegressor(hidden_layer_sizes = hu)
        nn.fit(msdX, self._y)
        prd  = nn.predict(msdX)
        tmp_nsc = nsc = self._nescience.nescience(nn, subset=viu, predictions=prd)
        
        # While the nescience decreases
        decreased = True        
        while (decreased):
            
            decreased = False

            #
            # Test adding a new feature  
            #
            
            # Check if therer are still more variables to add
            if np.sum(viu) != viu.shape[0]:
            
                new_msd = msd.copy()
                new_viu = viu.copy()
            
                new_viu[np.argmax(new_msd)] = 1
                new_msd[np.where(viu)] = -1

                msdX    = self._X[:,np.where(new_viu)[0]]
                new_nn  = MLPRegressor(hidden_layer_sizes = hu)        
                new_nn.fit(msdX, self._y)
                prd     = new_nn.predict(msdX)
                new_nsc = self._nescience.nescience(new_nn, subset=new_viu, predictions=prd)
            
                # Save data if nescience has been reduced                        
                if new_nsc < tmp_nsc:                                
                    decreased = True
                    tmp_nn  = new_nn
                    tmp_nsc = new_nsc
                    tmp_msd = new_msd
                    tmp_viu = new_viu
                    tmp_hu  = hu
                    
            #
            # Test adding a new layer
            #
            
            new_hu = hu.copy()
            new_hu.append(3)

            msdX    = self._X[:,np.where(viu)[0]]
            new_nn  = MLPRegressor(hidden_layer_sizes = new_hu)
            new_nn.fit(msdX, self._y)
            prd     = new_nn.predict(msdX)
            new_nsc = self._nescience.nescience(new_nn, subset=new_viu, predictions=prd)
            
            # Save data if nescience has been reduced 
            if new_nsc < tmp_nsc:                                
                decreased = True
                tmp_nn  = new_nn
                tmp_nsc = new_nsc
                tmp_msd = msd
                tmp_viu = viu
                tmp_hu  = new_hu

            #
            # Test adding a new unit
            #
            
            for i in np.arange(len(hu)):
                
                new_hu    = hu.copy()
                new_hu[i] = new_hu[i] + 1            

                msdX    = self._X[:,np.where(viu)[0]]
                new_nn  = MLPRegressor(hidden_layer_sizes = new_hu)
                new_nn.fit(msdX, self._y)
                prd     = new_nn.predict(msdX)
                new_nsc = self._nescience.nescience(new_nn, subset=viu, predictions=prd)
            
                # Save data if nescience has been reduced                        
                if new_nsc < tmp_nsc:                                
                    decreased = True
                    tmp_nn  = new_nn
                    tmp_nsc = new_nsc
                    tmp_msd = msd
                    tmp_viu = viu
                    tmp_hu  = new_hu
                
            # Update neural network
            nn      = tmp_nn
            nsc     = tmp_nsc
            viu     = tmp_viu
            msd     = tmp_msd
            hu      = tmp_hu

        # -> end while

        return (nsc, nn, viu)


    # WARNING: Experimental, do not use in production
    # TODO: build a sklearn wrapper around the model
    def GrammaticalEvolution(self):
        
        # A grammar is a dictionary keyed by non terminal symbols
        #     Each value is a list with the posible replacements
        #         Each replacement contains a list with tokens
        #
        # The grammar in use is:
        #
        #     <expression> ::= self._X[:,<feature>] |
        #                      <number> <scale> self._X[:,<feature>] |
        #                      self._X[:,<feature>]) ** <exponent> |
        #                      (<expression>) <operator> (<expression>)
        #                 
        #     <operator>   ::= + | - | * | /
        #     <scale>      ::= *
        #     <number>     ::= <digit> | <digit><digit0> | | <digit><digit0><digit0>
        #     <digit>      ::= 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
        #     <digit0>     ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
        #     <exponent>   ::= 2 | 3 | (1/2) | (1/3)
        #     <feature>    ::= 1 .. self._X.shape[1]

        self.grammar = {
            "expression": [
                            ["self._X[:,", "<feature>", "]"],
                            ["<number>", "<scale>", "self._X[:,", "<feature>", "]"],
                            ["self._X[:,", "<feature>", "]**", "<exponent>"],
                            ["(", "<expression>", ")", "<operator>", "(", "<expression>", ")"]
                          ],
            "operator":   ["+", "-", "*", "/"],
            "scale":      ["*"],
            "number":     [
                            ["<digit>"], 
                            ["<digit>", "<digit0>"],
                            ["<digit>", "<digit0>", "<digit0>"]
                          ],
            "digit":      ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "digit0":     ["0", "5"],
            "exponent":   ["2", "3", "(1/2)", "(1/3)"],
            "feature":    None
        }

        # Fill in features         
        self.grammar["feature"] = [str(i) for i in np.arange(0, self._X.shape[1])]

        self.max_num_tokens  = 10 # Sufficient to cover all possible tokens from grammar
        self.max_num_derivations = self.max_num_tokens * self.max_num_tokens # TODO: Think about that

        # Use differential evolution to find the optimal model
        bounds = [(0, self.max_num_tokens)] * self.max_num_derivations
        result = differential_evolution(self._evaluate_genotype, bounds)
        
        # Retrieve model
        model = self._parse_grammar(result.x)
        
        # Compute the predicted values
        pred = eval(model)

        # Compute model string
        model_str = model.replace("self.", "")
        
        # Compute the variables in use
        viu          = np.zeros(self._X.shape[1], dtype=int)                    
        match        = re.compile(r'self._X\[:,(\d+)\]') 
        indices      = match.findall(model) 
        indices      = [int(i) for i in indices] 
        viu[indices] = 1

        # Compute the nescience
        nsc = self._nescience.nescience(None, subset=viu, predictions=pred, model_string=model_str)
        
        return (nsc, model, viu)


    """
    Given a genotype (a list of integers) compute the nescience of the
    corresponding phenotype given the grammar.
    
    Return the nescience of the phenotype
    """
    def _evaluate_genotype(self, x):
                
        # Retrieve model
        model = self._parse_grammar(x)
                
        # Compute the predicted values
        try:
            pred = eval(model)
        except:
            # In case of non-evaluable model, return a nescience of 1
            return 1 
                            
        # Compute a simplified version of model string
        model_str = model.replace("self.", "")
                
        # Compute the variables in use
        viu          = np.zeros(self._X.shape[1], dtype=int)                    
        match        = re.compile(r'self._X\[:,(\d+)\]') 
        indices      = match.findall(model) 
        indices      = [int(i) for i in indices] 
        viu[indices] = 1
        
        # Compute the nescience
        try:
            nsc = self._nescience.nescience(None, subset=viu, predictions=pred, model_string=model_str)
        except:
            # In case of non-computable nesciencee, return a value of 1
            return 1 
                
        return nsc


    """
    Given a genotype (a list of integers) compute the  corresponding phenotype
    given the grammar.
    
    Return a string based phenotype
    """
    def _parse_grammar(self, x):
        
        x = [int(round(i)) for i in x]
        
        phenotype = ["<expression>"]
        ind       = 0
        modified  = True
        
        # Meanwhile there are no more non-terminal symbols
        while modified:
            
            modified = False
            new_phenotype = list()
                        
            for token in phenotype:
                            
                if token[0] == '<' and token[-1] == '>':
                    
                    token     = token[1:-1]
                    new_token = self.grammar[token][x[ind] % len(self.grammar[token])]
                                        
                    if type(new_token) == str:
                        new_token = list(new_token)
                                            
                    new_phenotype = new_phenotype + new_token
                    modified = True
                    ind = ind + 1
                    ind = ind % self.max_num_derivations
                                        
                else:
                                   
                    # new_phenotype = new_phenotype + list(token)
                    new_phenotype.append(token)
                         
            phenotype = new_phenotype
                    
        model = "".join(phenotype)

        return model


class AutoTimeSeries(BaseEstimator, RegressorMixin):
    
    # TODO: Class documentation

    def __init__(self):
        
        return None

    
    # TODO: provide support to autofit
    def fit(self, ts, auto=True):
        """
        Select the best model that explains the time series ts.
        
        Parameters
        ----------            
        ts : array-like, shape (n_samples)
            The time series as numbers.
        auto: compute automatically the optimal model
            
        Returns
        -------
        self
        """

        # Supported time series models
        
        self._models = [
            self._AutoRegressive,
            self._MovingAverage,
            self._ExponentialSmoothing
        ]

        self._X, self._y = self._whereIsTheX(ts)

        self._nescience = Nescience()
        self._nescience.fit(self._X, self._y)
        
        nsc = 1
        self._model = None
        self._viu   = None

        # Find optimal model
        if auto:
        
            for reg in self._models:
            
                (new_nsc, new_model, new_viu) = reg()
            
                if new_nsc < nsc: 
                    nsc   = new_nsc
                    self._model = new_model
                    self._viu   = new_viu
        
        return self


    """
       Transfrom a unidimensional time series ts into a classical X, y dataset
       
       * size: size of the X, that is, number of attributes
    """
    def _whereIsTheX(self, ts, size=None):
                
        X = list()
        y = list()

        lts = len(ts)
        
        if size == None:
            size = int(np.sqrt(lts))

        for i in np.arange(lts - size):
            X.append(ts[i:i+size])
            y.append(ts[i+size])
            
        X = np.array(X)
        y = np.array(y)
        
        return X, y
    

    def predict(self, X):
        """
        Predict class given a dataset
    
          * X = list([[x11, x12, x13, ...], ..., [xn1, xn2, ..., xnm]])
    
        Return the predicted value
        """
        
        # TODO: Check that we have a model trained
        
        check_is_fitted(self, '_X')
        
        if self._viu is None:
            msdX = X
        else:
            msdX = X[:,np.where(self._viu)[0]]
                
        return self._model.predict(msdX)


    def score(self, ts):
        """
        Evaluate the performance of the current model given a test time series

        Parameters
        ----------            
        ts : array-like, shape (n_samples)
            The time series as numbers.
            
        Returns
        -------    
        Return one minus the mean error
        """
        
        # TODO: Check that we have a model trained
        
        check_is_fitted(self, '_X')

        X, y = self._whereIsTheX(ts)
        
        if self._viu is None:
            msdX = X
        else:
            msdX = X[:,np.where(self._viu)[0]]        
        
        return self._model.score(msdX, y)


    def _AutoRegressive(self):
        
        # Relevance of features
        msd = self._nescience.miscoding.miscoding_features()
        
        # Variables in use
        viu = np.zeros(self._X.shape[1], dtype=np.int)

        # Select the the most relevant feature
        viu[np.argmax(msd)] = 1        
        msd[np.where(viu)] = -1

        # Evaluate the model        
        msdX = self._X[:,np.where(viu)[0]]        
        model = LinearRegression()
        model.fit(msdX, self._y)
        
        prd  = model.predict(msdX)
        nsc = self._nescience.nescience(model, subset=viu, predictions=prd)
        
        decreased = True
        while (decreased):
                        
            decreased = False
            
            new_msd = msd.copy()
            new_viu = viu.copy()
            
            # Select the the most relevant feature
            new_viu[np.argmax(new_msd)] = 1        
            new_msd[np.where(new_viu)] = -1

            # Evaluate the model        
            msdX = self._X[:,np.where(new_viu)[0]]        
            new_model = LinearRegression()
            new_model.fit(msdX, self._y)        
            
            prd  = new_model.predict(msdX)
            new_nsc = self._nescience.nescience(new_model, subset=new_viu, predictions=prd)
            
            # Save data if nescience has been reduced                        
            if new_nsc < nsc:                                
                decreased = True
                model     = new_model
                nsc       = new_nsc
                msd       = new_msd
                viu       = new_viu
        
        return (nsc, model, viu)


    def _MovingAverage(self):
        
        # Variables in use
        viu = np.zeros(self._X.shape[1], dtype=np.int)

        # Select the t-1 feature
        viu[-1] = 1        

        # Evaluate the model        
        msdX = self._X[:,np.where(viu)[0]]        
        model = LinearRegression()
        model.coef_ = np.array([1])
        model.intercept_ = np.array([0])
        
        prd  = model.predict(msdX)
        nsc = self._nescience.nescience(model, subset=viu, predictions=prd)
        
        for i in np.arange(2, self._X.shape[1] - 1):
            
            new_viu = viu.copy()
            
            # Select the the most relevant feature
            new_viu[-i] = 1        

            # Evaluate the model        
            msdX = self._X[:,np.where(new_viu)[0]]
            new_model = LinearRegression()
            new_model.coef_ = np.repeat([1/i], i)
            new_model.intercept_ = np.array([0])

            prd  = new_model.predict(msdX)
            new_nsc = self._nescience.nescience(new_model, subset=new_viu, predictions=prd)
                        
            # Save data if nescience has been reduced                        
            if new_nsc > nsc:
                break
              
            model     = new_model
            nsc       = new_nsc
            viu       = new_viu
        
        return (nsc, model, viu)


    def _ExponentialSmoothing(self):
        
        alpha = 0.2
        
        # Variables in use
        viu = np.zeros(self._X.shape[1], dtype=np.int)

        # Select the t-1 feature
        viu[-1] = 1        

        # Evaluate the model        
        msdX = self._X[:,np.where(viu)[0]]        
        model = LinearRegression()
        model.coef_ = np.array([1])
        model.intercept_ = np.array([0])
        
        prd  = model.predict(msdX)
        nsc = self._nescience.nescience(model, subset=viu, predictions=prd)
        
        for i in np.arange(2, self._X.shape[1] - 1):
            
            new_viu = viu.copy()
            
            # Select the the most relevant feature
            new_viu[-i] = 1        

            # Evaluate the model        
            msdX = self._X[:,np.where(new_viu)[0]]
            new_model = LinearRegression()
            new_model.coef_ = np.repeat([(1-alpha)**i], i)
            new_model.intercept_ = np.array([0])

            prd  = new_model.predict(msdX)
            new_nsc = self._nescience.nescience(new_model, subset=new_viu, predictions=prd)
                        
            # Save data if nescience has been reduced                        
            if new_nsc > nsc:
                break
              
            model     = new_model
            nsc       = new_nsc
            viu       = new_viu
        
        return (nsc, model, viu)
