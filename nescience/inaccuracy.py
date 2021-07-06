"""
inaccuracy.py

Fast auto machine learning
with the minimum nescience principle

@author:    Rafael Garcia Leiva
@mail:      rgarcialeiva@gmail.com
@web:       http://www.mathematicsunknown.com/
@copyright: GNU GPLv3
@version:   0.8
"""

from .utils import optimal_code_length

import numpy  as np

from sklearn.base             import BaseEstimator														
from sklearn.utils            import check_X_y
from sklearn.utils.validation import check_is_fitted


class Inaccuracy(BaseEstimator):
    """
    The fastautoml inaccuracy class allow us to compute the quality of
    the predictions made by a trained model.

    Example of usage:
        
        from fastautoml.inaccuracy import Inaccuracy
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.datasets import load_digits

        X, y = load_digits(return_X_y=True)

        tree = DecisionTreeClassifier(min_samples_leaf=5, random_state=42)
        tree.fit(X, y)

        inacc = Inaccuracy()
        inacc.fit(X, y)
        inacc.inaccuracy_model(tree)
    """    

    def __init__(self, y_type="numeric"):
        """
        Initialization of the class Inaccuracy
        
        Parameters
        ----------
        y_type : The type of the target, numeric or categorical
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

        return None
    
    
    def fit(self, X, y):
        """
        Fit the inaccuracy class with a dataset
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Sample vectors from which models have been trained.
            
        y : array-like, shape (n_samples)
            Continuous and categorical variables are supported
            if the trained model support them.
            
        Returns
        -------
        self
        """
        
        self.X_, self.y_ = check_X_y(X, y, dtype=None)
        self.y_ = np.array(self.y_)   
        self.len_y = optimal_code_length(x1=self.y_, numeric1=self.y_isnumeric)
        
        return self


    def inaccuracy_model(self, model):
        """
        Compute the inaccuracy of a model

        Parameters
        ----------       
        model : a trained model with a predict() method

        Returns
        -------         
        Return the inaccuracy (float)
        """        
        
        check_is_fitted(self)
        
        Pred      = model.predict(self.X_)
        len_pred  = optimal_code_length(x1=Pred, numeric1=self.y_isnumeric)
        len_joint = optimal_code_length(x1=Pred, numeric1=self.y_isnumeric, x2=self.y_, numeric2=self.y_isnumeric)
        inacc     = ( len_joint - min(self.len_y, len_pred) ) / max(self.len_y, len_pred)

        return inacc 

    
    def inaccuracy_predictions(self, predictions):
        """
        Compute the inaccuracy of a list of predicted values

        Parameters
        ----------       
        pred : array-like, shape (n_samples)
               The list of predicted values

        Returns
        -------                
        Return the inaccuracy (float)
        """        
        
        check_is_fitted(self)

        pred = np.array(predictions)
        
        len_pred  = optimal_code_length(x1=pred, numeric1=self.y_isnumeric)
        len_joint = optimal_code_length(x1=pred, numeric1=self.y_isnumeric, x2=self.y_, numeric2=self.y_isnumeric)
        inacc     = ( len_joint - min(self.len_y, len_pred) ) / max(self.len_y, len_pred)

        return inacc    
