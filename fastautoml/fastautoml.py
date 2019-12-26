"""

Fast auto machine learning
with the minimum nescience principle

@author:    Rafael Garcia Leiva
@mail:      rgarcialeiva@gmail.com
@web:       http://www.mathematicsunknown.com/
@copyright: GNU GPLv3
@version:   0.6

"""

import numpy  as np
import pandas as pd

import math

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.feature_selection.base import SelectorMixin

from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted

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
# - Holt Winter’s Exponential Smoothing

#
# Helper Functions
#

def discretize(x):
    """
    Discretize the variable x if needed
    
    Parameters
    ----------
    x : array-like, shape (n_samples)
        The variable to be discretized, if needed.
       
    Returns
    -------
    Return the miscoding (float)
    """

    if len(np.unique(x)) == 1:
        # Do not discretize if all the points belong to the same category
        new_x = np.zeros(len(x))
            
    elif len(np.unique(x)) > int(np.sqrt(len(x))):
        # Too many unique values wrt samples
        nbins = int(np.sqrt(len(x)))
        tmp   = pd.qcut(x, q=nbins, duplicates='drop')
        new_x = list(pd.Series(tmp).cat.codes)
            
        if len(np.unique(new_x)) == 1:
            # Discretization went too far
            new_x = x
                
    else:
        new_x = x
        
    return new_x


def optimal_code_length(x):
    """
    Compute the lenght of a list of values encoded using an optimal code
    
    Parameters
    ----------
    x : array-like, shape (n_samples)
        The values to be encoded.
       
    Returns
    -------
    Return the length of the encoded dataset (float)
    """
    
    # Discretize the variable x if needed
    
    new_x = discretize(x)
        
    # Compute the optimal length
        
    unique, count = np.unique(new_x, return_counts=True)
    ldm = np.sum(count * ( - np.log2(count / len(new_x))))
                    
    return ldm


def optimal_code_length_join(x1, x2):
    """
    Compute the lenght of the join of two variable
    encoded using an optimal code
    
    Parameters
    ----------
    x1 : array-like, shape (n_samples)
         The values of the first variable.
         
    x2 : array-like, shape (n_samples)
         The values of the second variable.         
       
    Returns
    -------
    Return the length of the encoded join dataset (float)    
    """
    
    # Discretize the variables X1 and X2 if needed
        
    new_x1 = discretize(x1)
    new_x2 = discretize(x2)
    
    # Compute the optimal length
    Join =  list(zip(new_x1, new_x2))
    unique, count = np.unique(Join, return_counts=True, axis=0)
    ldm = np.sum(count * ( - np.log2(count / len(Join))))
                    
    return ldm


#
# Class Miscoding
# 

class Miscoding(BaseEstimator, SelectorMixin):
    
    # TODO: Class documentation
    
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
            The target values (class labels) as numbers or strings.
            
        Returns
        -------
        self
        """
        
        self.X, self.y = check_X_y(X, y, dtype=None)

        # Regular miscoding
        self.regular = self._regular_miscoding()

        # Adjusted miscoding
        self.adjusted = 1 - self.regular
        self.adjusted = self.adjusted / np.sum(self.adjusted)

        # Partial miscoding
        self.partial = self.adjusted - self.regular / np.sum(self.regular)

        return self

    
    # TODO: do we have to support this?
    def _get_support_mask(self):
        
        check_is_fitted(self, 'tcc_')
        
        return None


    def miscoding_features(self, type='adjusted'):
        """
        Return the miscoding of the target given individual features

        Parameters
        ----------
        type : the type of miscoding we want to compute, possible values
               are 'regular', 'adjusted' and 'partial'.
            
        Returns
        -------
        Return a numpy array with the miscodings
        """
        
        check_is_fitted(self, 'regular')
        
        if type == 'regular':
            return self.regular
        elif type == 'adjusted':
            return self.adjusted
        elif type == 'partial':
            return self.partial
        
        # TODO: rise exception
        return None


    def miscoding_model(self, model):
        """
        Compute the global miscoding of the dataset given a model
        
        Parameters
        ----------
        model : a model of one of the supported classeses
                    
        Returns
        -------
        Return the miscoding (float)
        """
        
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
            # TODO: Disinguish between TypeError and NotImplementedError
            raise NotImplementedError('Model not supported')

        return self.miscoding_subset(subset)
        

    def miscoding_subset(self, subset):
        """
        Compute the global miscoding of a subset of the dataset
        
        Parameters
        ----------
        subset : array-like, shape (n_features)
                 1 if the attribute is in use, 0 otherwise
            
        Returns
        -------
        Return the miscoding (float)
        """
        
        # TODO: check the format of subset
        
        miscoding = np.dot(subset, self.partial)
        miscoding = 1 - miscoding
                       
        return miscoding

    
    """    
    Compute the regular miscoding of each feature
          
    Return a numpy array with the regular miscodings
    """
    def _regular_miscoding(self):
         
        miscoding = list()
        
        # Discretize y and compute the encoded length
        
        Resp  = discretize(self.y)
        ldm_y = optimal_code_length(self.y)

        for i in np.arange(self.X.shape[1]):

            # Discretize feature and compute lengths
            
            Pred  = discretize(self.X[:,i])
            ldm_X = optimal_code_length(self.X[:,i])
            
            ldm_Xy = optimal_code_length_join(Resp, Pred)

            # Compute miscoding
                       
            mscd = ( ldm_Xy - min(ldm_X, ldm_y) ) / max(ldm_X, ldm_y)
                
            miscoding.append(mscd)
                
        return np.array(miscoding)


    """
    Compute the attributes in use for a multinomial naive Bayes classifier
    
    Return array with the attributes in use
    """
    def _MultinomialNB(self, estimator):

        # TODO: sanity check over estimator

        # All the attributes are in use
        attr_in_use = np.ones(self.X.shape[1], dtype=int)
            
        return attr_in_use
    

    """
    Compute the attributes in use for a decision tree
    
    Return array with the attributes in use
    """
    def _DecisionTreeClassifier(self, estimator):

        # TODO: sanity check over estimator

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

        # TODO: sanity check over estimator

        # All the attributes are in use
        attr_in_use = np.ones(self.X.shape[1], dtype=int)
            
        return attr_in_use


    """
    Compute the attributes in use for a multilayer perceptron classifier
    
    Return array with the attributes in use
    """
    def _MLPClassifier(self, estimator):

        # TODO: sanity check over estimator

        attr_in_use = np.ones(self.X.shape[1], dtype=int)
            
        return attr_in_use


    """
    Compute the attributes in use for a linear regression
    
    Return array with the attributes in use
    """
    def _LinearRegression(self, estimator):
        
        # TODO: sanity check over estimator

        attr_in_use = np.ones(self.X.shape[1], dtype=int)
            
        return attr_in_use


    """
    Compute the attributes in use for a decision tree regressor
    
    Return array with the attributes in use
    """
    def _DecisionTreeRegressor(self, estimator):
        
        # TODO: sanity check over estimator

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

        # TODO: sanity check over estimator

        attr_in_use = np.ones(self.X.shape[1], dtype=int)
            
        return attr_in_use

    """
    Compute the attributes in use for a multilayer perceptron regressor
    
    Return array with the attributes in use
    """
    def _MLPRegressor(self, estimator):

        # TODO: sanity check over estimator

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
                
        self.len_y = optimal_code_length(self.y)
        
        return self


    def inaccuracy_model(self, model):
        """
        Compute the inaccuracy of a model
        
        model : trained model with a predict() method
         
        Return the inaccuracy (float)
        """        
        
        check_is_fitted(self, 'X')
        
        Pred = model.predict(self.X)
        len_pred = optimal_code_length(Pred)
        
        len_join = optimal_code_length_join(Pred, self.y)
        
        inacc = ( len_join - min(self.len_y, len_pred) ) / max(self.len_y, len_pred)

        return inacc

    
    def inaccuracy_predictions(self, predictions):
        """
        Compute the inaccuracy of a list of predicted values
        
         pred : array-like, shape (n_samples)
                The list of predicted values
                
        Return the inaccuracy (float)
        """        
        
        check_is_fitted(self, 'X')
        
        len_pred = optimal_code_length(predictions)
        
        len_join = optimal_code_length_join(predictions, self.y)
        
        inacc = ( len_join - min(self.len_y, len_pred) ) / max(self.len_y, len_pred)

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
                
        self.len_y = optimal_code_length(self.y)
        
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
            # TODO: Disinguish between TypeError and NotImplementedError
            raise NotImplementedError('Model not supported')

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

        py    = discretize(np.exp(estimator.class_log_prior_))
        
        theta = np.exp(estimator.feature_log_prob_)
        theta = theta.flatten()
        theta = discretize(theta)
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
        M = discretize(M)
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

        annw = discretize(annw)
        
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

        annb = discretize(annb)
        
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
        M = discretize(M)
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

        annw = discretize(annw)
        
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

        annb = discretize(annb)
        
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

        # Supported Classifiers
        
        self.classifiers = [
            self.MultinomialNB,
            self.DecisionTreeClassifier,
            self.LinearSVC,
            self.MLPClassifier
        ]
        
        return None

    
    def fit(self, X, y):

        self.X, self.y = check_X_y(X, y, dtype=None)

        self.nescience = Nescience()
        self.nescience.fit(self.X, self.y)
        
        nsc = 1
        self.model = None
        self.viu   = None
        
        for clf in self.classifiers:
            
            (new_nsc, new_model, new_viu) = clf()
                        
            if new_nsc < nsc:
                nsc   = new_nsc
                self.model = new_model
                self.viu   = new_viu
        
        return self


    def predict(self, X):
        """
        Predict class given a dataset
    
          * X = list([[x11, x12, x13, ...], ..., [xn1, xn2, ..., xnm]])
    
        Return a list of classes predicted
        """
        
        # TODO: Check that we have a model trained
        
        if self.viu is None:
            msdX = X
        else:
            msdX = X[:,np.where(self.viu)[0]]
                
        return self.model.predict(msdX)


    def predict_proba(self, X):
        """
        Predict the probability of being in a class given a dataset
    
          * X = list([[x11, x12, x13, ...], ..., [xn1, xn2, ..., xnm]])
      
        Return an array of probabilities. The order of the list match the order
        the internal attribute classes_
        """
        
        # TODO: Check that we have a model trained

        if self.viu is None:
            msdX = X
        else:
            msdX = X[:,np.where(self.viu)[0]]
                    
        return self.model.predict_proba(msdX)


    def score(self, X, y):
        """
        Evaluate the performance of the current model given a test dataset

           * X = list([[x11, x12, x13, ...], ..., [xn1, xn2, ..., xnm]])
           * y = list([y1, ..., yn])
    
        Return one minus the mean error
        """
        
        # TODO: Check that we have a model trained
        
        if self.viu is None:
            msdX = X
        else:
            msdX = X[:,np.where(self.viu)[0]]        
        
        return self.model.score(msdX, y)


    def MultinomialNB(self):
        
        # No hyperparameters to optimize
        
        model = MultinomialNB()
        model.fit(self.X, self.y)

        nsc = self.nescience.nescience(model)
            
        return (nsc, model, None)

    
    def LinearSVC(self):
        
        # No parameters to optimize
                    
        model = LinearSVC(multi_class="crammer_singer")
        model.fit(self.X, self.y)

        nsc = self.nescience.nescience(model)
            
        return (nsc, model, None)    


    def DecisionTreeClassifier(self):
        
        depth   = 3
        nsc     = 1
        new_nsc = 0.99
        
        while new_nsc < nsc:

            nsc = new_nsc
            
            model = DecisionTreeClassifier(max_depth=depth)
            model.fit(self.X, self.y)

            new_nsc = self.nescience.nescience(model)
            
            depth = depth + 1

        return (nsc, model, None)
    
    
    def MLPClassifier(self):
        
        # Relevance of features
        tmp_msd = msd = self.nescience.miscoding.miscoding_features()
        
        # Variables in use
        tmp_viu = viu = np.zeros(self.X.shape[1], dtype=np.int)

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
        
        msdX = self.X[:,np.where(viu)[0]]
        tmp_nn = nn = MLPClassifier(hidden_layer_sizes = hu)
        nn.fit(msdX, self.y)
        prd  = nn.predict(msdX)
        tmp_nsc = nsc = self.nescience.nescience(nn, subset=viu, predictions=prd)
        
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

                msdX    = self.X[:,np.where(new_viu)[0]]
                new_nn  = MLPClassifier(hidden_layer_sizes = hu)        
                new_nn.fit(msdX, self.y)
                prd     = new_nn.predict(msdX)
                new_nsc = self.nescience.nescience(new_nn, subset=new_viu, predictions=prd)
            
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

            msdX    = self.X[:,np.where(viu)[0]]
            new_nn  = MLPClassifier(hidden_layer_sizes = new_hu)
            new_nn.fit(msdX, self.y)
            prd     = new_nn.predict(msdX)
            new_nsc = self.nescience.nescience(new_nn, subset=new_viu, predictions=prd)
            
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

                msdX    = self.X[:,np.where(viu)[0]]
                new_nn  = MLPClassifier(hidden_layer_sizes = new_hu)
                new_nn.fit(msdX, self.y)
                prd     = new_nn.predict(msdX)
                new_nsc = self.nescience.nescience(new_nn, subset=viu, predictions=prd)
            
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

        # Supported Regressors
        
        self.regressors = [
            self.LinearRegression,
            self.LinearSVR,
            self.DecisionTreeRegressor,
            self.MLPRegressor
        ]
        
        return None

    
    def fit(self, X, y):
        """
        Select the best model that explains y given X.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Sample vectors from which to compute miscoding.
            
        y : array-like, shape (n_samples)
            The target values (class labels) as numbers or strings.
            
        Returns
        -------
        self
        """

        self.X, self.y = check_X_y(X, y, dtype=None)

        self.nescience = Nescience()
        self.nescience.fit(self.X, self.y)
        
        nsc = 1
        self.model = None
        self.viu   = None
        
        for reg in self.regressors:
            
            print("Regressor: " + str(reg))
            
            (new_nsc, new_model, new_viu) = reg()
            
            if new_nsc < nsc:
                nsc   = new_nsc
                self.model = new_model
                self.viu   = new_viu
        
        return self


    def predict(self, X):
        """
        Predict class given a dataset
    
          * X = list([[x11, x12, x13, ...], ..., [xn1, xn2, ..., xnm]])
    
        Return the predicted value
        """
        
        # TODO: Check that we have a model trained
        
        if self.viu is None:
            msdX = X
        else:
            msdX = X[:,np.where(self.viu)[0]]
                
        return self.model.predict(msdX)


    def score(self, X, y):
        """
        Evaluate the performance of the current model given a test dataset

           * X = list([[x11, x12, x13, ...], ..., [xn1, xn2, ..., xnm]])
           * y = list([y1, ..., yn])
    
        Return one minus the mean error
        """
        
        # TODO: Check that we have a model trained
        
        if self.viu is None:
            msdX = X
        else:
            msdX = X[:,np.where(self.viu)[0]]        
        
        return self.model.score(msdX, y)


    def LinearRegression(self):
        
        # Relevance of features
        msd = self.nescience.miscoding.miscoding_features()
        
        # Variables in use
        viu = np.zeros(self.X.shape[1], dtype=np.int)

        # Select the the most relevant feature
        viu[np.argmax(msd)] = 1        
        msd[np.where(viu)] = -1

        # Evaluate the model        
        msdX = self.X[:,np.where(viu)[0]]        
        model = LinearRegression()
        model.fit(msdX, self.y)
        
        prd  = model.predict(msdX)
        nsc = self.nescience.nescience(model, subset=viu, predictions=prd)
        
        decreased = True
        while (decreased):
                        
            decreased = False
            
            new_msd = msd.copy()
            new_viu = viu.copy()
            
            # Select the the most relevant feature
            new_viu[np.argmax(new_msd)] = 1        
            new_msd[np.where(new_viu)] = -1

            # Evaluate the model        
            msdX = self.X[:,np.where(new_viu)[0]]        
            new_model = LinearRegression()
            new_model.fit(msdX, self.y)        
            
            prd  = new_model.predict(msdX)
            new_nsc = self.nescience.nescience(new_model, subset=new_viu, predictions=prd)
            
            # Save data if nescience has been reduced                        
            if new_nsc < nsc:                                
                decreased = True
                model     = new_model
                nsc       = new_nsc
                msd       = new_msd
                viu       = new_viu
        
        return (nsc, model, viu)


    def LinearSVR(self):
        # TODO: Optimize hyperparameters
                    
        # model = LinearSVR(multi_class="crammer_singer")
        model = LinearSVR()
        model.fit(self.X, self.y)

        nsc = self.nescience.nescience(model)
            
        return (nsc, model, None)    


    def DecisionTreeRegressor(self):
        
        depth   = 3
        nsc     = 1
        new_nsc = 0.99
        
        while new_nsc < nsc:

            nsc = new_nsc
                        
            model = DecisionTreeRegressor(max_depth=depth)
            model.fit(self.X, self.y)

            new_nsc = self.nescience.nescience(model)
            
            depth = depth + 1

        return (nsc, model, None)


    def MLPRegressor(self):
        
        # Relevance of features
        tmp_msd = msd = self.nescience.miscoding.miscoding_features()
        
        # Variables in use
        tmp_viu = viu = np.zeros(self.X.shape[1], dtype=np.int)

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
        
        msdX = self.X[:,np.where(viu)[0]]
        tmp_nn = nn = MLPRegressor(hidden_layer_sizes = hu)
        nn.fit(msdX, self.y)
        prd  = nn.predict(msdX)
        tmp_nsc = nsc = self.nescience.nescience(nn, subset=viu, predictions=prd)
        
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

                msdX    = self.X[:,np.where(new_viu)[0]]
                new_nn  = MLPRegressor(hidden_layer_sizes = hu)        
                new_nn.fit(msdX, self.y)
                prd     = new_nn.predict(msdX)
                new_nsc = self.nescience.nescience(new_nn, subset=new_viu, predictions=prd)
            
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

            msdX    = self.X[:,np.where(viu)[0]]
            new_nn  = MLPRegressor(hidden_layer_sizes = new_hu)
            new_nn.fit(msdX, self.y)
            prd     = new_nn.predict(msdX)
            new_nsc = self.nescience.nescience(new_nn, subset=new_viu, predictions=prd)
            
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

                msdX    = self.X[:,np.where(viu)[0]]
                new_nn  = MLPRegressor(hidden_layer_sizes = new_hu)
                new_nn.fit(msdX, self.y)
                prd     = new_nn.predict(msdX)
                new_nsc = self.nescience.nescience(new_nn, subset=viu, predictions=prd)
            
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


class AutoTimeSeries(BaseEstimator, RegressorMixin):
    
    # TODO: Class documentation

    def __init__(self):

        # Supported time series models
        
        self.models = [
            self.AutoRegressive,
            self.MovingAverage,
            self.ExponentialSmoothing
        ]
        
        return None

    
    def fit(self, ts):
        """
        Select the best model that explains the time series ts.
        
        Parameters
        ----------            
        ts : array-like, shape (n_samples)
            The time series as numbers.
            
        Returns
        -------
        self
        """

        self.X, self.y = self._whereIsTheX(ts)

        self.nescience = Nescience()
        self.nescience.fit(self.X, self.y)
        
        nsc = 1
        self.model = None
        self.viu   = None
        
        for reg in self.models:
            
            (new_nsc, new_model, new_viu) = reg()
            
            if new_nsc < nsc:
                nsc   = new_nsc
                self.model = new_model
                self.viu   = new_viu
        
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
        
        if self.viu is None:
            msdX = X
        else:
            msdX = X[:,np.where(self.viu)[0]]
                
        return self.model.predict(msdX)


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

        X, y = self._whereIsTheX(ts)
        
        if self.viu is None:
            msdX = X
        else:
            msdX = X[:,np.where(self.viu)[0]]        
        
        return self.model.score(msdX, y)


    def AutoRegressive(self):
        
        # Relevance of features
        msd = self.nescience.miscoding.miscoding_features()
        
        # Variables in use
        viu = np.zeros(self.X.shape[1], dtype=np.int)

        # Select the the most relevant feature
        viu[np.argmax(msd)] = 1        
        msd[np.where(viu)] = -1

        # Evaluate the model        
        msdX = self.X[:,np.where(viu)[0]]        
        model = LinearRegression()
        model.fit(msdX, self.y)
        
        prd  = model.predict(msdX)
        nsc = self.nescience.nescience(model, subset=viu, predictions=prd)
        
        decreased = True
        while (decreased):
                        
            decreased = False
            
            new_msd = msd.copy()
            new_viu = viu.copy()
            
            # Select the the most relevant feature
            new_viu[np.argmax(new_msd)] = 1        
            new_msd[np.where(new_viu)] = -1

            # Evaluate the model        
            msdX = self.X[:,np.where(new_viu)[0]]        
            new_model = LinearRegression()
            new_model.fit(msdX, self.y)        
            
            prd  = new_model.predict(msdX)
            new_nsc = self.nescience.nescience(new_model, subset=new_viu, predictions=prd)
            
            # Save data if nescience has been reduced                        
            if new_nsc < nsc:                                
                decreased = True
                model     = new_model
                nsc       = new_nsc
                msd       = new_msd
                viu       = new_viu
        
        return (nsc, model, viu)


    def MovingAverage(self):
        
        # Variables in use
        viu = np.zeros(self.X.shape[1], dtype=np.int)

        # Select the t-1 feature
        viu[-1] = 1        

        # Evaluate the model        
        msdX = self.X[:,np.where(viu)[0]]        
        model = LinearRegression()
        model.coef_ = np.array([1])
        model.intercept_ = np.array([0])
        
        prd  = model.predict(msdX)
        nsc = self.nescience.nescience(model, subset=viu, predictions=prd)
        
        for i in np.arange(2, self.X.shape[1] - 1):
            
            new_viu = viu.copy()
            
            # Select the the most relevant feature
            new_viu[-i] = 1        

            # Evaluate the model        
            msdX = self.X[:,np.where(new_viu)[0]]
            new_model = LinearRegression()
            new_model.coef_ = np.repeat([1/i], i)
            new_model.intercept_ = np.array([0])

            prd  = new_model.predict(msdX)
            new_nsc = self.nescience.nescience(new_model, subset=new_viu, predictions=prd)
                        
            # Save data if nescience has been reduced                        
            if new_nsc > nsc:
                break
              
            model     = new_model
            nsc       = new_nsc
            viu       = new_viu
        
        return (nsc, model, viu)


    def ExponentialSmoothing(self):
        
        alpha = 0.2
        
        # Variables in use
        viu = np.zeros(self.X.shape[1], dtype=np.int)

        # Select the t-1 feature
        viu[-1] = 1        

        # Evaluate the model        
        msdX = self.X[:,np.where(viu)[0]]        
        model = LinearRegression()
        model.coef_ = np.array([1])
        model.intercept_ = np.array([0])
        
        prd  = model.predict(msdX)
        nsc = self.nescience.nescience(model, subset=viu, predictions=prd)
        
        for i in np.arange(2, self.X.shape[1] - 1):
            
            new_viu = viu.copy()
            
            # Select the the most relevant feature
            new_viu[-i] = 1        

            # Evaluate the model        
            msdX = self.X[:,np.where(new_viu)[0]]
            new_model = LinearRegression()
            new_model.coef_ = np.repeat([(1-alpha)**i], i)
            new_model.intercept_ = np.array([0])

            prd  = new_model.predict(msdX)
            new_nsc = self.nescience.nescience(new_model, subset=new_viu, predictions=prd)
                        
            # Save data if nescience has been reduced                        
            if new_nsc > nsc:
                break
              
            model     = new_model
            nsc       = new_nsc
            viu       = new_viu
        
        return (nsc, model, viu)
