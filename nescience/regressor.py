"""
regressor.py

Machine learning
with the minimum nescience principle

@author:    Rafael Garcia Leiva
@mail:      rgarcialeiva@gmail.com
@web:       http://www.mathematicsunknown.com/
@copyright: GNU GPLv3
"""

import numpy  as np

from sklearn.base import BaseEstimator, RegressorMixin														
														
from sklearn.utils            import check_X_y
from sklearn.utils            import check_array
from sklearn.utils.validation import check_is_fitted

# Supported regressors

from sklearn.linear_model   import LinearRegression
from sklearn.tree           import DecisionTreeRegressor
from sklearn.svm            import LinearSVR
from sklearn.neural_network import MLPRegressor

from .nescience import Nescience
    
class Regressor(BaseEstimator, RegressorMixin):
    
    # TODO: Class documentation

    def __init__(self, auto=True, random_state=None):
        
        self.random_state = random_state
        self.auto = auto
        
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
        
        auto: find automatically the optimal model
            
        Returns
        -------
        self
        """
        
        # Supported Regressors
        
        self.regressors_ = [
            self.LinearRegression,
            self.LinearSVR,
            self.DecisionTreeRegressor,
            self.MLPRegressor
        ]

        self.X_, self.y_ = check_X_y(X, y, dtype=None)

        self.nescience_ = Nescience(X_type="numeric", y_type="numeric")
        self.nescience_.fit(self.X_, self.y_)
        
        nsc = 1
        self.model_ = None
        self.viu_   = None
        
        # Find automatically the optimal model
        
        if self.auto:
            
            for reg in self.regressors_:
            
                # TODO: Should be based on a verbose flag
                print("Regressor: " + str(reg), end='')
            
                (new_nsc, new_model, new_viu) = reg()
                
                print("Nescience:", new_nsc)
            
                if new_nsc < nsc:
                    nsc   = new_nsc
                    self.model_ = new_model
                    self.viu_   = new_viu
        return self


    def predict(self, X):
        """
        Predict class given a dataset
    
          * X = list([[x11, x12, x13, ...], ..., [xn1, xn2, ..., xnm]])
    
        Return the predicted value
        """
        
        check_is_fitted(self)
        X = check_array(X)
        
        if self.viu_ is None:
            msdX = X
        else:
            msdX = X[:,np.where(self.viu_)[0]]
                
        return self.model_.predict(msdX)


    def score(self, X, y):
        """
        Evaluate the performance of the current model given a test dataset

           * X = list([[x11, x12, x13, ...], ..., [xn1, xn2, ..., xnm]])
           * y = list([y1, ..., yn])
    
        Return one minus the mean error
        """

        check_is_fitted(self)
        X = check_array(X)
        
        if self.viu_ is None:
            msdX = X
        else:
            msdX = X[:,np.where(self.viu_)[0]]        
        
        return self.model_.score(msdX, y)


    def get_model(self):
        """
        Get access to the private attribute model
		
        Return self.model_
        """
        return self.model_
		
		
    def LinearRegression(self):
        
        # Relevance of features
        msd = self.nescience_.miscoding_.miscoding_features()
        
        # Variables in use
        viu = np.zeros(self.X_.shape[1], dtype=np.int)

        # Select the the most relevant feature
        viu[np.argmax(msd)] = 1        
        msd[np.where(viu)] = -1

        # Evaluate the model
        msdX = self.X_[:,np.where(viu)[0]]        
        model = LinearRegression()
        model.fit(msdX, self.y_)
        
        prd  = model.predict(msdX)
        nsc = self.nescience_.nescience(model, subset=viu, predictions=prd)
        
        decreased = True
        while (decreased):
            
            decreased = False
            
            new_msd = msd.copy()
            new_viu = viu.copy()
            
            # Select the the most relevant feature
            new_viu[np.argmax(new_msd)] = 1        
            new_msd[np.where(new_viu)] = -1

            # Evaluate the model        
            msdX = self.X_[:,np.where(new_viu)[0]]        
            new_model = LinearRegression()
            new_model.fit(msdX, self.y_)        
            
            prd  = new_model.predict(msdX)
            new_nsc = self.nescience_.nescience(new_model, subset=new_viu, predictions=prd)
            
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
        model = LinearSVR(random_state=self.random_state)
        model.fit(self.X_, self.y_)

        nsc = self.nescience_.nescience(model)
            
        return (nsc, model, None)    


    def DecisionTreeRegressor(self):
        
        clf  = DecisionTreeRegressor(random_state=self.random_state)
        path = clf.cost_complexity_pruning_path(self.X_, self.y_)

        previous_nodes = -1
        best_nsc       = 1
        best_model     = None
        
        # For every possible prunning point in reverse order
        for ccp_alpha in reversed(path.ccp_alphas):
                
            model = DecisionTreeRegressor(ccp_alpha=ccp_alpha, random_state=self.random_state)
            model.fit(self.X_, self.y_)
    
            # Skip if nothing has changed
            if model.tree_.node_count == previous_nodes:
                continue
    
            previous_nodes = model.tree_.node_count
    
            new_nsc = self.nescience_.nescience(model)
            
            if new_nsc < best_nsc:
                best_nsc   = new_nsc
                best_model = model
            else:
                break
    
        return (best_nsc, best_model, None)       


    def MLPRegressor(self):
        
        # Relevance of features
        tmp_msd = msd = self.nescience_.miscoding_.miscoding_features()
        
        # Variables in use
        tmp_viu = viu = np.zeros(self.X_.shape[1], dtype=np.int)

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
        
        msdX = self.X_[:,np.where(viu)[0]]
        tmp_nn = nn = MLPRegressor(hidden_layer_sizes = hu, random_state=self.random_state)
        nn.fit(msdX, self.y_)
        prd  = nn.predict(msdX)
        tmp_nsc = nsc = self.nescience_.nescience(nn, subset=viu, predictions=prd)
        
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

                msdX    = self.X_[:,np.where(new_viu)[0]]
                new_nn  = MLPRegressor(hidden_layer_sizes = hu, random_state=self.random_state)        
                new_nn.fit(msdX, self.y_)
                prd     = new_nn.predict(msdX)
                new_nsc = self.nescience_.nescience(new_nn, subset=new_viu, predictions=prd)
            
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

            msdX    = self.X_[:,np.where(viu)[0]]
            new_nn  = MLPRegressor(hidden_layer_sizes = new_hu, random_state=self.random_state)
            new_nn.fit(msdX, self.y_)
            prd     = new_nn.predict(msdX)
            new_nsc = self.nescience_.nescience(new_nn, subset=viu, predictions=prd)
            
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

                msdX    = self.X_[:,np.where(viu)[0]]
                new_nn  = MLPRegressor(hidden_layer_sizes = new_hu, random_state=self.random_state)
                new_nn.fit(msdX, self.y_)
                prd     = new_nn.predict(msdX)
                new_nsc = self.nescience_.nescience(new_nn, subset=viu, predictions=prd)
            
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