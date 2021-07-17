"""
classifier.py

Machine learning
with the minimum nescience principle

@author:    Rafael Garcia Leiva
@mail:      rgarcialeiva@gmail.com
@web:       http://www.mathematicsunknown.com/
@copyright: GNU GPLv3
"""

from .nescience import Nescience

import numpy  as np

from sklearn.base import BaseEstimator, ClassifierMixin												
														
from sklearn.utils            import check_X_y
from sklearn.utils            import check_array
from sklearn.utils.validation import check_is_fitted

from scipy.optimize import differential_evolution

# Supported classifiers

from sklearn.naive_bayes    import MultinomialNB
from sklearn.tree           import DecisionTreeClassifier
from sklearn.svm            import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.svm			import SVC


class Classifier(BaseEstimator, ClassifierMixin):
    """
    Given a dataset X = {x1, ..., xp} composed by p features, and a target
    variable y, automatically select the best model for a classification problem.
    In particular, it computes the optimal subset of features, select
    the best family of models, and the best hyperparameters for the model
    selected.

    Example of usage:

        from nescience.classifier import Classifer
        from sklearn.dataset import load_digits
        X, y = load_digits(return_X_y=True)
        model = Classifer()
        model.fit(X, y)
        model.score(X, y)
    """
    
    def __init__(self, auto=True, fast=True, verbose=False, random_state=None):
        """
        Initialization of the class Classifier
        
        Parameters
        ----------
        auto         : find automatically the optimal model
        fast         : use a greedy approach for fast training
        verbose      : print addtional information
        random_state : seed for the radom numbers generator
        """

        self.auto         = auto
        self.fast         = fast
        self.verbose      = verbose
        self.random_state = random_state
        
        return None

    
    def fit(self, X, y):
        """
        Select the best model that explains y given X.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Sample vectors from which to compute miscoding.
            
        y : array-like, shape (n_samples)
            The target values as numbers.
                    
        Returns
        -------
        self
        """        
                
        # Supported Classifiers
        self.classifiers_ = [
            self.MultinomialNB,
            self.DecisionTreeClassifier,
            self.LinearSVC,
            self.SVC,            
            self.MLPClassifier
        ]

        self.X_, self.y_ = check_X_y(X, y, dtype=None)

        self.nescience_ = Nescience(X_type="numeric", y_type="categorical")
        self.nescience_.fit(self.X_, self.y_)
        
        # new y contains class indexes rather than labels in the range [0, n_classes]
        self.classes_, self.y_ = np.unique(self.y_, return_inverse=True)						  
        
        nsc = 1
        self.model_ = None
        self.viu_   = None
        
        # Find optimal model
        if self.auto:
        
            for clf in self.classifiers_:
            
                if self.verbose:
                    print("Classifier: " + str(clf), end='')
                
                # If X contains negative values, MultinomialNB is skipped
                if clf == self.MultinomialNB and not (self.X_>=0).all():
                    if self.verbose:
                        print("Skipped!")                
                    continue
                
                (new_nsc, new_model, new_viu) = clf()

                if self.verbose:
                    print("Nescience:", new_nsc)                

                if new_nsc < nsc:
                    nsc         = new_nsc
                    self.model_ = new_model
                    self.viu_   = new_viu

        return self


    def predict(self, X):
        """
        Predict class given a dataset
    
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)   

        Returns
        -------
        A list of classes predicted
        """
        
        check_is_fitted(self)
        X = check_array(X)
        
        if self.viu_ is None:
            msdX = X
        else:
            msdX = X[:,np.where(self.viu_)[0]]

        return self.classes_[self.model_.predict(msdX)]


    def predict_proba(self, X):
        """
        Predict the probability of being in a class given a dataset
    
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)   

        Returns
        -------      
        An array of probabilities. The order of the list match
        the order the internal attribute classes_
        """
        
        check_is_fitted(self)
        X = check_array(X)

        if self.viu_ is None:
            msdX = X
        else:
            msdX = X[:,np.where(self.viu_)[0]]
					
        return self.model_.predict_proba(msdX)


    def score(self, X, y):
        """
        Evaluate the performance of the current model given a test dataset

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)   
        y : (optional) array-like, shape (n_samples)

        Returns
        -------
        One minus the mean error
        """
        
        check_is_fitted(self)
        X, y = check_X_y(X, y, dtype=None)
        
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


    def MultinomialNB(self):
        
        # No hyperparameters to optimize
        
        model = MultinomialNB()
        model.fit(self.X_, self.y_)

        nsc = self.nescience_.nescience(model)
            
        return (nsc, model, None)

    
    def LinearSVC(self):
        
        # No hyperparameters to optimize
        
        model = SVC(kernel='linear', probability=True, random_state=self.random_state, decision_function_shape='ovo')
        model.fit(self.X_, self.y_)

        nsc = self.nescience_.nescience(model)
            
        return (nsc, model, None)


    def SVC(self): 
       
        # Different searches are possible to find the best hyperparameters
        # The following one produced good results on the three datasets it was tested on
        
        # The four hyperparameters to optimize
        hyper_param_default = ['degree', 'C', 'gamma', 'coef0']
		
        # The order in which they will be treated in the search
        hyper_param_order = [1, 2, 3, 4]
		
        # gamma is searched among the same values as C and coef0 but is always multiplied by inv (its default value)
        inv = 1/(len(self.X_[0])*np.var(self.X_))
		
        # maximum number of iterations to fit the SVC models in this search. Could be reduced to 1e5 or 1e4
        max_iter = 1e6

        hyper_param = [] 
        for i in range(4):
            hyper_param.append(hyper_param_default[hyper_param_order[i]-1])
        
        # Default values
        param_value = {'degree': 5, 'C': 1, 'gamma': inv, 'coef0': 1}
		
        tmp_model = SVC(kernel='poly', max_iter=max_iter)
        tmp_model.set_params(**param_value)
        tmp_model.fit(self.X_, self.y_)
        tmp_nsc = self.nescience_.nescience(tmp_model)
    
        decreased = True
        while (decreased):
        
            decreased = False
            
            for param in hyper_param:
                
                if param=='degree':
                
                    # Test degree=degree+1
                    tmp_model.set_params(**{param: param_value[param]+1})
                    tmp_model.fit(self.X_, self.y_)
                    new_nsc = self.nescience_.nescience(tmp_model)
                    if new_nsc<tmp_nsc:
                        tmp_nsc = new_nsc
                        param_value[param] += 1
                        decreased = True
                    else:
					
                        # Test degree=degree-1
                        tmp_model.set_params(**{param: param_value[param]-1})
                        tmp_model.fit(self.X_, self.y_)
                        new_nsc = self.nescience_.nescience(tmp_model)
                        if new_nsc<tmp_nsc:
                            tmp_nsc = new_nsc
                            decreased = True
                            param_value[param] -= 1
                        else:
				
                            # Test degree=degree+2
                            tmp_model.set_params(**{param: param_value[param]+2})
                            tmp_model.fit(self.X_, self.y_)
                            new_nsc = self.nescience_.nescience(tmp_model)
                            if new_nsc<tmp_nsc:
                                tmp_nsc = new_nsc
                                param_value[param] += 2
                                decreased = True
                            else:
					
                                # Test degree=degree-2
                                tmp_model.set_params(**{param: param_value[param]-2})
                                tmp_model.fit(self.X_, self.y_)
                                new_nsc = self.nescience_.nescience(tmp_model)
                                if new_nsc<tmp_nsc:
                                    tmp_nsc = new_nsc
                                    decreased = True
                                    param_value[param] -= 2
                                else:
                                    tmp_model.set_params(**{param: param_value[param]})
                
                else: # param = 'C' or 'coef0' or 'gamma'
                
                    # Test param=param*2
                    tmp_model.set_params(**{param: param_value[param]*2})
                    tmp_model.fit(self.X_, self.y_)
                    new_nsc = self.nescience_.nescience(tmp_model)
                    if new_nsc<tmp_nsc:
                        tmp_nsc = new_nsc
                        param_value[param] *= 2
                        decreased = True
                    else:
					
                        # Test param=param/2
                        tmp_model.set_params(**{param: param_value[param]/2})
                        tmp_model.fit(self.X_, self.y_)
                        new_nsc = self.nescience_.nescience(tmp_model)
                        if new_nsc<tmp_nsc:
                            tmp_nsc = new_nsc
                            param_value[param] /= 2
                            decreased = True
                        else:
                            tmp_model.set_params(**{param: param_value[param]})
                            if param=='coef0':
							
                                # Test coef0=-coef0
                                tmp_model.set_params(**{param: -param_value[param]})
                                tmp_model.fit(self.X_, self.y_)
                                new_nsc = self.nescience_.nescience(tmp_model)
                                if new_nsc<tmp_nsc:
                                    tmp_nsc = new_nsc
                                    param_value[param] *= -1
                                    decreased = True
                                else:
                                    tmp_model.set_params(**{param: param_value[param]})
    

        model = tmp_model.fit(self.X_, self.y_)
        nsc = tmp_nsc
        
        if self.auto==False:
            self.model_ = model
            
        return (nsc, model, None)


    def DecisionTreeClassifier(self):
        """
        Find the best model (hyperparameters optimization) in the familiy of decision trees       
		
        Return
        ------
        best_nsc   : best nescience achieved
        best_model : a trained DecisionTreeClassifer
        best_viu   : None, since all the variables are used as input 
        """

        # We restrict ourselves to at least 5 samples per leave,
        # otherwise the algorithm could take too much time to converge,
        # Anyway, the limit of 5 is considered a good practice in ML
        clf  = DecisionTreeClassifier(min_samples_leaf=5)

        # Compute prunning points
        path = clf.cost_complexity_pruning_path(self.X_, self.y_)

        previous_nodes = -1
        best_nsc       = 1
        best_model     = None

        # For every possible prunning point in reverse order
        for ccp_alpha in reversed(path.ccp_alphas):
    
            model = DecisionTreeClassifier(ccp_alpha=ccp_alpha, min_samples_leaf=5, random_state=self.random_state)
            model.fit(self.X_, self.y_)
    
            # Skip evaluation if nothing has changed
            if model.tree_.node_count == previous_nodes:
                continue
    
            previous_nodes = model.tree_.node_count
    
            new_nsc = self.nescience_.nescience(model)
    
            if new_nsc < best_nsc:
                best_nsc   = new_nsc
                best_model = model
            else:
                if self.fast:
                    # Early stop
                    break

        return (best_nsc, best_model, None)

    
    def MLPClassifier(self):
        
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
        viu[np.argmax(msd)] = 1        
        msd[np.where(viu)] = -1
        viu[np.argmax(msd)] = 1
        msd[np.where(viu)] = -1
        
        msdX = self.X_[:,np.where(viu)[0]]
        tmp_nn = nn = MLPClassifier(hidden_layer_sizes = hu, random_state=self.random_state)
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
                new_nn  = MLPClassifier(hidden_layer_sizes = hu, random_state=self.random_state)        
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
            new_nn  = MLPClassifier(hidden_layer_sizes = new_hu, random_state=self.random_state)
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
                new_nn  = MLPClassifier(hidden_layer_sizes = new_hu, random_state=self.random_state)
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