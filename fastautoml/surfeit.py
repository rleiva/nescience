"""
surfeit.py

Fast auto machine learning
with the minimum nescience principle

@author:    Rafael Garcia Leiva
@mail:      rgarcialeiva@gmail.com
@web:       http://www.mathematicsunknown.com/
@copyright: GNU GPLv3
@version:   0.8
"""

from .utils import optimal_code_length
from .utils import discretize_vector

import numpy  as np

from sklearn.base  import BaseEstimator
from sklearn.utils import check_X_y

# Compressors

import bz2
import lzma
import zlib

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

# Supported time series
# 
# - Autoregression
# - Moving Average
# - Simple Exponential Smoothing


class Surfeit(BaseEstimator):

    def __init__(self, y_type="numeric", compressor="bz2"):
        """
        Initialization of the class Surfeit

        Parameters
        ----------
        y_type:     The type of the target, numeric or categorical
        compressor: The compressor used to encode the model (bz2, lzma or zlib)
        """

        valid_y_types = ("numeric", "categorical")

        if y_type not in valid_y_types:
            raise ValueError("Valid options for 'y_type' are {}. "
                             "Got vartype={!r} instead."
                             .format(valid_y_types, y_type))

        if y_type == "numeric":
            self.y_isnumeric = True
        else:
            self.y_isnumeric = False	        

        valid_compressors = ("bz2", "lzma", "zlib")

        if compressor not in valid_compressors:
            raise ValueError("Valid options for 'compressor' are {}. "
                             "Got vartype={!r} instead."
                             .format(valid_compressor, compressor))

        self.compressor  = compressor
        
        return None
    

    def fit(self, X, y):
        """Initialize the Surfeit class with dataset
        
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
        
        self.X_, self.y_ = check_X_y(X, y, dtype=None)
        self.len_y_ = optimal_code_length(x1=self.y_, numeric1=self.y_isnumeric)
        
        return self
    

    def surfeit_model(self, model):
        """
        Compute the redundancy of a model

        Parameters
        ----------
        model : a model of one of the supported classeses
        
        Supported classifiers
            MultinomialNB
            DecisionTreeClassifier
            LinearSVC
            MLPClassifier
            SVC

        Supported regressors
            LinearRegression
            DecisionTreeRegressor
            LinearSVR
            MLPRegressor

        Supported time series
            Autoregression
            Moving Average
            Simple Exponential Smoothing
            
        Returns
        -------
        Redundancy (float) of the model
        """

        if isinstance(model, MultinomialNB):
            model_str = self._MultinomialNB(model)
        elif isinstance(model, DecisionTreeClassifier):
            model_str = self._DecisionTreeClassifier(model)
        elif isinstance(model, SVC) and model.get_params()['kernel']=='linear':
            model_str = self._LinearSVC(model)
        elif isinstance(model, SVC) and model.get_params()['kernel']=='poly':
            model_str = self._SVC(model)
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

        if self.len_y_ < km:
            # redundancy = 1 - l(C(y)) / l(m)
            redundancy = 1 - self.len_y_ / lm
        else:
            # redundancy = 1 - l(m*) / l(m)
            redundancy = 1 - km / lm
                            
        return redundancy


    """
    Convert a MultinomialNB classifier into a string
    """
    def _MultinomialNB(self, estimator):
        #
        # Discretize probabilities
        #

        py    = discretize_vector(np.exp(estimator.class_log_prior_))
        
        theta = np.exp(estimator.feature_log_prob_)
        theta = theta.flatten()
        theta = discretize_vector(theta)
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
    """
    def _LinearSVC(self, estimator):
  
        #
        # Discretize similarities
        #

        M = estimator.coef_
        shape = M.shape
        M = M.flatten()
        M = discretize_vector(M)
        M = np.array(M)
        M = M.reshape(shape)
        
        intercept = estimator.intercept_
        shape = intercept.shape
        intercept = intercept.flatten()
        intercept = discretize_vector(intercept)
        intercept = np.array(intercept)
        intercept = intercept.reshape(shape)
        
        classes = estimator.classes_
        
        if len(classes) == 2:
        
            #
            # Create the model
            #

            # Header
            string = "def LinearSVC(X):\n"
                 
            # Similarities
            string = string + "    M = ["        
            for j in np.arange(len(M)-1):
                string = string + str(M[j]) + ", "
            string = string + str(M[-1])
            string = string + "]]\n"
            
            string = string + "    intercept = ["        
            string = string + str(intercept)
            string = string + "]\n"
            
            # Computation of the decision function
            string = string + '    y_hat = [None]*len(X)'
            string = string + '    for i in range(len(X)):\n'
            string = string + '        prob = 0\n'
            string = string + '        for k in range(len(M)):\n'
            string = string + '            prob = prob + X[i][k] * M[k]\n'
            string = string + '        prob = prob + intercept\n'
			
            #Prediction
            string = string + '        if prob > 0:\n'
            string = string + '            y_hat[i] = 0\n'
            string = string + '        else:\n'
            string = string + '            y_hat[i] = 1\n'
            string = string + '    return y_hat\n'
        
        else:
        
            #
            # Create the model
            #

            # Header
            string = "def LinearSVC(X):\n"
                 
            # Similarities
            string = string + "    M = ["        
            for i in np.arange(len(M)-1):
                string = string + "["
                for j in np.arange(len(M[i])-1):
                    string = string + str(M[i][j]) + ", "
                string = string + str(M[i][-1])
                string = string + "], "
            string = string + "["
            for j in np.arange(len(M[-1])-1):
                string = string + str(M[-1][j]) + ", "
            string = string + str(M[-1][-1])
            string = string + "]]\n"
            
            string = string + "    intercept = ["        
            for i in np.arange(len(intercept)-1):
                string = string + str(intercept[i])
                string = string + ", "
            string = string + str(intercept[-1])
            string = string + "]\n"
            
            string = string + "    classes = ["        
            for i in np.arange(len(classes-1)):
                string = string + str(classes[i]) + ", "
            string = string + str(classes[-1])
            string = string + "]]\n"
           
		    # Computation of the decision function ('ovo' strategy)
            string = string + '    y_hat = [None]*len(X)'
            string = string + '    for i in range(len(X)):\n'
            string = string + '        votes = [0]*len(classes)\n'
            string = string + '        idx = 0\n'
            string = string + '        for j in range(len(classes)):\n'
            string = string + '            for l in range(len(classes)-j-1):\n'
            string = string + '                prob = 0\n'
            string = string + '                for k in range(len(M[idx])):\n'
            string = string + '                    prob = prob + X[i][k] * M[idx][k]\n'
            string = string + '                prob = prob + intercept[idx]\n'
            string = string + '                if prob > 0:\n'
            string = string + '                    votes[j] = votes[j] + 1\n'
            string = string + '                else:\n'
            string = string + '                    votes[l+j+1] = votes[l+j+1] + 1\n'
            string = string + '                idx = idx + 1\n'
            
            # Prediction
            string = string + '        max_vote = 0\n'
            string = string + '        i_max_vote = 0\n'
            string = string + '        for k in range(len(votes)):\n'
            string = string + '            if votes[k]>max_vote:\n'
            string = string + '                max_vote = votes[k]\n'
            string = string + '                i_max_vote = k\n'
            string = string + '        y_hat[i] = classes[i_max_vote]\n'
            string = string + '    return y_hat\n'
        
        return string


    """
	Convert a SVC classifier into a string
	"""
    string = ""
    affiche = 1
    def _SVC(self, estimator):
	
        #
        # Discretize similarities
        #


        M = estimator._dual_coef_
        shape = M.shape
        M = M.flatten()
        M = discretize_vector(M)
        M = np.array(M)
        M = M.reshape(shape)
        
        support_vectors = estimator.support_vectors_
        shape = support_vectors.shape
        support_vectors = support_vectors.flatten()
        support_vectors = discretize_vector(support_vectors)
        support_vectors = np.array(support_vectors)
        support_vectors = support_vectors.reshape(shape)
        
        if len(estimator.classes_) == 2:
        
            #
            # Create the model

            # Header
            string = "def SVC(X):\n"
                 
            # Similarities
            string = string + "    dual_coef = ["        
            for i in np.arange(len(M)-1):
                string = string + "["
                for j in np.arange(len(M[i])-1):
                    string = string + str(M[i][j]) + ", "
                string = string + str(M[i][-1])
                string = string + "], "
            string = string + "["
            for j in np.arange(len(M[-1])-1):
                string = string + str(M[-1][j]) + ", "
            string = string + str(M[-1][-1])
            string = string + "]]\n"
            
            string = string + "    intercept = ["        
            string = string + str(estimator.intercept_)
            string = string + "]\n"
            
            string = string + "    classes = ["        
            for i in np.arange(len(estimator.classes_)-1):
                string = string + str(estimator.classes_[i]) + ", "
            string = string + str(estimator.classes_[-1])
            string = string + "]\n"
            
            string = string + "    support_vectors = ["        
            for i in np.arange(len(support_vectors)-1):
                string = string + "["
                for j in np.arange(len(support_vectors[i])-1):
                    string = string + str(support_vectors[i][j]) + ", "
                string = string + str(support_vectors[i][-1])
                string = string + "], "
            string = string + "["
            for j in np.arange(len(support_vectors[-1])-1):
                string = string + str(support_vectors[-1][j]) + ", "
            string = string + str(support_vectors[-1][-1])
            string = string + "]]\n"
            
            string = string + "    n_support = ["        
            for i in np.arange(len(estimator.n_support_)-1):
                string = string + str(estimator.n_support_[i]) + ", "
            string = string + str(estimator.n_support_[-1])
            string = string + "]\n"
            
            string = string + "    degree = "
            string = string + str(estimator.degree)
            string = string + "    \n"
            
            string = string + "    gamma = "
            if estimator.gamma == 'scale':
                string = string + str(1/(len(support_vectors[0])*np.var(self.X_)))
            elif estimator.gamma == 'auto':
                string = string + str(1/len(support_vectors[0]))
            else:
                string = string + str(estimator.gamma)
            string = string + "    \n"
            
            string = string + "    r = "
            string = string + str(estimator.coef0)
            string = string + "    \n"

            # Computation of the decision function ('ovo' strategy)
            string = string + "    y_hat    = [None]*len(X)\n"
            string = string + "    for i in range(len(X)):\n"
            string = string + "        prob = 0\n" 
            string = string + "        for i_sv in range(len(support_vectors)):\n"
            string = string + "            sum = 0\n" 
            string = string + "            for k in range(len(X[i])):\n"
            string = string + "                sum = sum + support_vectors[i_sv][k] * X[i][k]\n"
            string = string + "            x = 1\n"
            string = string + "            for k in range(degree):\n"
            string = string + "                x = x * (gamma * sum + r)\n"
            string = string + "            prob = prob + x * dual_coef[i_sv]\n"
            string = string + "        prob = prob + intercept[0]\n"
	
            # Prediction
            string = string + "        if prob > 0:\n"
            string = string + "            y_hat[i] = 0\n"
            string = string + "        else:\n"
            string = string + "            y_hat[i] = 1\n"
            string = string + "    return y_hat\n"

        else:
        
            #
            # Create the model
            #

            # Header
            string = "def SVC(X):\n"
                 
            # Similarities
            string = string + "    dual_coef = ["        
            for i in np.arange(len(M)-1):
                string = string + "["
                for j in np.arange(len(M[i])-1):
                    string = string + str(M[i][j]) + ", "
                string = string + str(M[i][-1])
                string = string + "], "
            string = string + "["
            for j in np.arange(len(M[-1])-1):
                string = string + str(M[-1][j]) + ", "
            string = string + str(M[-1][-1])
            string = string + "]]\n"
            
            string = string + "    intercept = ["        
            for i in np.arange(len(estimator.intercept_)-1):
                string = string + str(estimator.intercept_[i]) + ", "
            string = string + str(estimator.intercept_[-1])
            string = string + "]\n"
            
            string = string + "    classes = ["        
            for i in np.arange(len(estimator.classes_)-1):
                string = string + str(estimator.classes_[i]) + ", "
            string = string + str(estimator.classes_[-1])
            string = string + "]\n"
            
            string = string + "    support_vectors = ["        
            for i in np.arange(len(support_vectors)-1):
                string = string + "["
                for j in np.arange(len(support_vectors[i])-1):
                    string = string + str(support_vectors[i][j]) + ", "
                string = string + str(support_vectors[i][-1])
                string = string + "], "
            string = string + "["
            for j in np.arange(len(support_vectors[-1])-1):
                string = string + str(support_vectors[-1][j]) + ", "
            string = string + str(support_vectors[-1][-1])
            string = string + "]]\n"
            
            string = string + "    n_support = ["        
            for i in np.arange(len(estimator.n_support_)-1):
                string = string + str(estimator.n_support_[i]) + ", "
            string = string + str(estimator.n_support_[-1])
            string = string + "]\n"
            
            string = string + "    idx_support = ["        
            for i in np.arange(len(estimator.n_support_)):
                string = string + str(np.sum(estimator.n_support_[:i])) + ", "
            string = string + str(np.sum(estimator.n_support_))
            string = string + "]\n"
            
            string = string + "    degree = "
            string = string + str(estimator.degree)
            string = string + "    \n"
            
            string = string + "    gamma = "
            if estimator.gamma == 'scale':
                string = string + str(1/(len(support_vectors[0])*np.var(self.X_)))
            elif estimator.gamma == 'auto':
                string = string + str(1/len(support_vectors[0]))
            else:
                string = string + str(estimator.gamma)
            string = string + "    \n"
            
            string = string + "    r = "
            string = string + str(estimator.coef0)
            string = string + "    \n"

            # Computation of the decision function ('ovo' strategy)
            string = string + "    y_hat    = [None]*len(X)\n"
            string = string + "    for i in range(len(X)):\n"
            string = string + "        votes = [0]*len(classes)\n"
            string = string + "        idx = 0\n"
            string = string + "        for j in range(len(classes)):\n"
            string = string + "            for l in range(len(classes)-j-1):\n"
            string = string + "                prob = 0\n" 
            string = string + "                sum = 0\n" 
            string = string + "                for i_sv in range(idx_support[j],idx_support[j+1]):\n"
            string = string + "                    for k in range(len(X[i])):\n"
            string = string + "                        sum = sum + support_vectors[i_sv][k] * X[i][k]\n"
            string = string + "                    x = 1\n"
            string = string + "                    for k in range(degree):\n"
            string = string + "                        x = x * (gamma * sum + r)\n"
            string = string + "                    prob = prob + x * dual_coef[l+j][i_sv]\n"
            string = string + "                sum = 0\n"
            string = string + "                for i_sv in range(idx_support[j+l],idx_support[j+l+1]):\n"
            string = string + "                    for k in range(len(X[i])):\n"
            string = string + "                        sum = sum + support_vectors[i_sv][k] * X[i][k]\n"
            string = string + "                    x = 1\n"
            string = string + "                    for k in range(degree):\n"
            string = string + "                        x = x * (gamma * sum + r)\n"
            string = string + "                    prob = prob + x * dual_coef[j][i_sv]\n"
            string = string + "                prob = prob + intercept[idx]\n"
            string = string + "                if prob > 0:\n"
            string = string + "                    votes[j] = votes[j] + 1\n"
            string = string + "                else:\n"
            string = string + "                    votes[l+j+1] = votes[l+j+1] + 1\n"
            string = string + "                idx = idx + 1\n"
		
            # Prediction
            string = string + "        max_vote = 0\n"
            string = string + "        i_max_vote = 0\n"
            string = string + "        for k in range(len(votes)):\n"
            string = string + "            if votes[k]>max_vote:\n"
            string = string + "                max_vote, i_max_vote = votes[k], k\n"
            string = string + "        y_hat[i] = classes[i_max_vote]\n"
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
        
        # Compute the tree header
        
        features_set = set()
                
        for node_id in range(n_nodes):

            # If we have a test node
            if (children_left[node_id] != children_right[node_id]):
                features_set.add('X%d' % (feature[node_id]+1))
        
        tree_string = tree_string + "def tree" + str(features_set) + ":\n"

        # Compute the tree body
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
        annw = discretize_vector(annw)
        
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
        annb = discretize_vector(annb)
        
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
            
        string = string + "    y_hat = 0\n"
        string = string + "    for i in range(len(W)):\n"
        string = string + "        y_hat = y_hat + W[i] * X[i]\n"
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
        M = discretize_vector(M)
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
        annw = discretize_vector(annw)
        
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
        annb = discretize_vector(annb)
        
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