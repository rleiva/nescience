"""
timeseries.py

Machine learning
with the minimum nescience principle

@author:    Rafael Garcia Leiva
@mail:      rgarcialeiva@gmail.com
@web:       http://www.mathematicsunknown.com/
@copyright: GNU GPLv3
"""

import numpy  as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin																					
from sklearn.utils.validation import check_is_fitted
from sklearn.utils            import column_or_1d
from sklearn.utils            import check_array

from sklearn.linear_model     import LinearRegression

from nescience.utils import optimal_code_length
from nescience.nescience import Nescience

# from .nescience import Nescience

# Supported time series models
# 
# - Autoregression
# - Moving Average
# - Simple Exponential Smoothing

class TimeSeries(BaseEstimator, RegressorMixin):
    """
    Given a time series ts = {x1, ..., xn} composed by n samples, 
    computes a model to forecast t future values of the series.

    Example of usage:
        
        from nescience.autotimeseries import AutoTimeSeries

        ts = ...

        model = AutoTimeSeries()
        mode.fit(ts)
        model.predict(1)
    """

    def __init__(self, y_type="numeric", multivariate=False, X_type="numeric"):
        """
        Initialization of the class TimeSeries
        
        Parameters
        ----------
        y_type:       The type of the time series, numeric or categorical
        multivariate: "True" if we have other time series available as predictors
        X_type:       The type of the predictors, numeric, mixed or categorical,
                      in case of having a multivariate time series
        """        

        valid_y_types = ("numeric", "categorical")
        if y_type not in valid_y_types:
            raise ValueError("Valid options for 'y_type' are {}. "
                             "Got vartype={!r} instead."
                             .format(valid_y_types, y_type))

        if multivariate:
            valid_X_types = ("numeric", "mixed", "categorical")
            if X_type not in valid_X_types:
                raise ValueError("Valid options for 'X_type' are {}. "
                                 "Got vartype={!r} instead."
                                 .format(valid_X_types, X_type))

        self.X_type       = X_type
        self.y_type       = y_type
        self.multivariate = multivariate


    def fit(self, y, X=None):
        """
        Initialize the time series class with data
        
        Parameters
        ----------
        y : array-like, shape (n_samples)
            The target time series.
        X : (optional) array-like, shape (n_samples, n_features)
            Time series features in case of a multivariate time series problem
            
        Returns
        -------
        self
        """

        self.y_ = column_or_1d(y)

        if self.y_type == "numeric":
            self.y_isnumeric = True
        else:
            self.y_isnumeric = False
        
        # Process X in case of a multivariate time series
        if self.multivariate:
            
            if X is None:
                raise ValueError("X argument is mandatory in case of multivariate time series.")

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
    

    def auto_timeseries(self):
        """
        Select the best model that explains the time series ts.
        """

        # Supported time series models
        
        self.models_ = [
            self.AutoRegressive,
            self.MovingAverage,
            self.ExponentialSmoothing
        ]

        self.X_, self.y_ = self._whereIsTheX(self.y_)

        self.nescience_ = Nescience(X_type="numeric", y_type="numeric")
        self.nescience_.fit(self.X_, self.y_)
        
        nsc = 1
        self.model_ = None
        self.viu_   = None

        # Find optimal model
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


    def ExponentialSmoothing(self, alpha=0.2):
        """
        Learn empirically the miscoding of the features of X
        as a representation of y.
        
        Parameters
        ----------
        alpha : decay factor
            
        Returns
        -------
        (nescience, model, variables_in_use)
        """

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


    def auto_miscoding(self, min_lag=1, max_lag=None, mode='adjusted'):
        """
        Return the auto-miscoding of a time series, for a given number of lags

        Parameters
        ----------
        min_lag   : starting lag
        max_lag   : end lag. If none, the squared root of the number of samples is used
        mode      : the mode of miscoding, possible values are 'regular' for
                    the true miscoding, 'adjusted' for the normalized inverted
                    values, and 'partial' with positive and negative
                    contributions to dataset miscoding.
            
        Returns
        -------
        Return a numpy array with the lagged miscodings
        """

        check_is_fitted(self)

        valid_modes = ('regular', 'adjusted', 'partial')

        if mode not in valid_modes:
            raise ValueError("Valid options for 'mode' are {}. "
                             "Got mode={!r} instead."
                            .format(valid_modes, mode))
        
        lag_mscd = list()
        
        # Use a default value for max_lag
        if max_lag == None:
            max_lag = int(np.sqrt(self.y_.shape[0]))

        for i in np.arange(start=min_lag, stop=max_lag):

            # Compute lagged vectors
            new_y = self.y_.copy()
            new_y = np.roll(new_y, -i)
            new_y = new_y[:-i]
            new_x = self.y_.copy()
            new_x = new_x[:-i]

            # Compute miscoding
            ldm_y  = optimal_code_length(x1=new_y, numeric1=self.y_isnumeric)
            ldm_X  = optimal_code_length(x1=new_x, numeric1=self.y_isnumeric)
            ldm_Xy = optimal_code_length(x1=new_x, numeric1=self.y_isnumeric, x2=new_y, numeric2=self.y_isnumeric)
            mscd   = ( ldm_Xy - min(ldm_X, ldm_y) ) / max(ldm_X, ldm_y)
                
            lag_mscd.append(mscd)
                
        regular = np.array(lag_mscd)

        if mode == 'regular':
            return regular

        elif mode == 'adjusted':

            adjusted = 1 - regular
            if np.sum(adjusted) != 0:
                adjusted = adjusted / np.sum(adjusted)
            return adjusted

        elif mode == 'partial':

            adjusted = 1 - regular
            if np.sum(adjusted) != 0:
                adjusted = adjusted / np.sum(adjusted)

            if np.sum(regular) != 0:
                partial  = adjusted - regular / np.sum(regular)
            else:
                partial  = adjusted
            return partial

        else:
            raise ValueError("Valid options for 'mode' are {}. "
                             "Got mode={!r} instead."
                            .format(valid_modes, mode))


    def cross_miscoding(self, attribute, min_lag=1, max_lag=None, mode='adjusted'):
        """
        Return the cross-miscoding of the target time series and a second time series
        in case of multivariate time series

        Parameters
        ----------
        attribute : the attribute of the second series
        min_lag   : starting lag
        max_lag   : end lag. If none, the squared root of the number of samples is used.
        mode      : the mode of miscoding, possible values are 'regular' for
                    the true miscoding, 'adjusted' for the normalized inverted
                    values, and 'partial' with positive and negative
                    contributions to dataset miscoding.
            
        Returns
        -------
        Return a numpy array with the lagged miscodings
        """        

        check_is_fitted(self)

        valid_modes = ('regular', 'adjusted', 'partial')

        if mode not in valid_modes:
            raise ValueError("Valid options for 'mode' are {}. "
                             "Got mode={!r} instead."
                            .format(valid_modes, mode))
        
        lag_mscd = list()
        
        # Use a default value for max_lag
        if max_lag == None:
            max_lag = int(np.sqrt(self.X_.shape[0]))

        for i in np.arange(start=min_lag, stop=max_lag):

            # Compute lagged vectors
            new_y = self.y_.copy()
            new_y = np.roll(new_y, -i)
            new_y = new_y[:-i]
            new_x = self.X_[:,attribute].copy()
            new_x = new_x[:-i]

            # Compute miscoding
            ldm_y  = optimal_code_length(x1=new_y, numeric1=self.y_isnumeric)
            ldm_X  = optimal_code_length(x1=new_x, numeric1=self.y_isnumeric)
            ldm_Xy = optimal_code_length(x1=new_x, numeric1=self.y_isnumeric, x2=new_y, numeric2=self.y_isnumeric)
            mscd   = ( ldm_Xy - min(ldm_X, ldm_y) ) / max(ldm_X, ldm_y)
                
            lag_mscd.append(mscd)
                
        regular = np.array(lag_mscd)

        if mode == 'regular':
            return regular

        elif mode == 'adjusted':

            adjusted = 1 - regular
            if np.sum(adjusted) != 0:
                adjusted = adjusted / np.sum(adjusted)
            return adjusted

        elif mode == 'partial':

            adjusted = 1 - regular
            if np.sum(adjusted) != 0:
                adjusted = adjusted / np.sum(adjusted)

            if np.sum(regular) != 0:
                partial  = adjusted - regular / np.sum(regular)
            else:
                partial  = adjusted
            return partial

        else:
            raise ValueError("Valid options for 'mode' are {}. "
                             "Got mode={!r} instead."
                            .format(valid_modes, mode))

