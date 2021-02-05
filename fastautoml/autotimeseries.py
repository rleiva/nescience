"""
autotimeseries.py

Fast auto machine learning
with the minimum nescience principle

@author:    Rafael Garcia Leiva
@mail:      rgarcialeiva@gmail.com
@web:       http://www.mathematicsunknown.com/
@copyright: GNU GPLv3
@version:   0.8
"""

import numpy  as np

from sklearn.base import BaseEstimator, RegressorMixin																					
from sklearn.utils.validation import check_is_fitted

# Supported time series
# 
# - Autoregression
# - Moving Average
# - Simple Exponential Smoothing

class AutoTimeSeries(BaseEstimator, RegressorMixin):
    
    # TODO: Class documentation

    def __init__(self, auto=True):
        
        self.auto = auto
		
        return None

    
    # TODO: provide support to autofit
    def fit(self, ts):
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
        
        self.models_ = [
            self.AutoRegressive,
            self.MovingAverage,
            self.ExponentialSmoothing
        ]

        self.X_, self.y_ = self._whereIsTheX(ts)

        self.nescience_ = Nescience(X_type="numeric", y_type="numeric")
        self.nescience_.fit(self.X_, self.y_)
        
        nsc = 1
        self.model_ = None
        self.viu_   = None

        # Find optimal model
        if self.auto:
        
            for reg in self.models_:
            
                (new_nsc, new_model, new_viu) = reg()
            
                if new_nsc < nsc: 
                    nsc   = new_nsc
                    self.model_ = new_model
                    self.viu_   = new_viu
        
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
        
        check_is_fitted(self)
        
        if self.viu_ is None:
            msdX = X
        else:
            msdX = X[:,np.where(self.viu_)[0]]
                
        return self.model_.predict(msdX)


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
        
        check_is_fitted(self)

        X, y = self._whereIsTheX(ts)
        
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

		
    def AutoRegressive(self):
        
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


    def MovingAverage(self):
        
        # Variables in use
        viu = np.zeros(self.X_.shape[1], dtype=np.int)

        # Select the t-1 feature
        viu[-1] = 1        

        # Evaluate the model        
        msdX = self.X_[:,np.where(viu)[0]]        
        model = LinearRegression()
        model.coef_ = np.array([1])
        model.intercept_ = np.array([0])
        
        prd  = model.predict(msdX)
        nsc = self.nescience_.nescience(model, subset=viu, predictions=prd)
        
        for i in np.arange(2, self.X_.shape[1] - 1):
            
            new_viu = viu.copy()
            
            # Select the the most relevant feature
            new_viu[-i] = 1        

            # Evaluate the model        
            msdX = self.X_[:,np.where(new_viu)[0]]
            new_model = LinearRegression()
            new_model.coef_ = np.repeat([1/i], i)
            new_model.intercept_ = np.array([0])

            prd  = new_model.predict(msdX)
            new_nsc = self.nescience_.nescience(new_model, subset=new_viu, predictions=prd)
                        
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
        viu = np.zeros(self.X_.shape[1], dtype=np.int)

        # Select the t-1 feature
        viu[-1] = 1        

        # Evaluate the model        
        msdX = self.X_[:,np.where(viu)[0]]        
        model = LinearRegression()
        model.coef_ = np.array([1])
        model.intercept_ = np.array([0])
        
        prd  = model.predict(msdX)
        nsc = self.nescience_.nescience(model, subset=viu, predictions=prd)
        
        for i in np.arange(2, self.X_.shape[1] - 1):
            
            new_viu = viu.copy()
            
            # Select the the most relevant feature
            new_viu[-i] = 1        

            # Evaluate the model        
            msdX = self.X_[:,np.where(new_viu)[0]]
            new_model = LinearRegression()
            new_model.coef_ = np.repeat([(1-alpha)**i], i)
            new_model.intercept_ = np.array([0])

            prd  = new_model.predict(msdX)
            new_nsc = self.nescience_.nescience(new_model, subset=new_viu, predictions=prd)
                        
            # Save data if nescience has been reduced                        
            if new_nsc > nsc:
                break
              
            model     = new_model
            nsc       = new_nsc
            viu       = new_viu
        
        return (nsc, model, viu)