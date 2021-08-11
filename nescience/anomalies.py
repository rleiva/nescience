"""
anomalies.py

Machine learning
with the minimum nescience principle

@author:    Rafael Garcia Leiva
@mail:      rgarcialeiva@gmail.com
@web:       http://www.mathematicsunknown.com/
@copyright: GNU GPLv3
"""

import pandas as pd
import numpy  as np
														
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_is_fitted

from .classifier import Classifier
from .regressor  import Regressor
from .miscoding  import Miscoding
from .utils      import discretize_vector

class Anomalies():
    """
    Anomalies detection and classification

    Given a dataset X = {x1, ..., xp} composed by p features, and a target
    variable y, identify and classify those samples that do not match
    the regularity patterns found in the rest of the dataset.

    The Anomalies class also allow us to classify the identified anomalies
    according to the characteristicts they share.

    Example of usage:
        
        from nescience.anomalies import Anomalies
        from sklearn.datasets import load_beast_cancer

        X, y = load_breast_cancer(return_X_y=True)

        anomalies = Anomalies()
        anomalies.fit(X, y)
        results = anomalies.get_anomalies()
    """

    def __init__(self, X_type="numeric", y_type="numeric"):
        """

        Initialization of the class Anomalies

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

        self.X_type = X_type
        self.y_type = y_type


    def fit(self, X, y, model=None, balancedness_low=0.2, balancedness_high=0.8, redundancy=0.6):
        """
        Learn empirically the anomalies of a the dataset (X, y)
        
        Parameters
        ----------
        X     : array-like, shape (n_samples, n_attributes)            
        y     : array-like, shape (n_samples)
        model : the best possible, non-overfitting, trained, sklearn-based model,
                if None, a model will be learnt with the nescience's AutoML capabilities
        balancedness_low  : low threshold to filter unbalanced clusters.
        balancedness_high : high threshold to filter unbalanced clusters.
        redundancy        : threshold to filter redundant clusters.

        """

        self.X_ = X
        optimal_bins = int(np.log2(y.shape[0])/2)

        # Discretize target if needed
        if self.y_type == "numeric":
            self.y_ = discretize_vector(y, n_bins=optimal_bins)
        else:
            self.y_ = y

        # Thresholds for filters 
        self.balancedness_low  = balancedness_low
        self.balancedness_high = balancedness_high
        self.redundancy        = redundancy

        # If no model is specified, learn it by AutoML
        if model is None:
            # TODO: Force the use of Decision Trees meanwhile the auto-classifier is fixed
            model = Classifier(auto=False, fast=False)
            model.fit(self.X_, self.y_)
            nsc, self.model, viu = model.DecisionTreeClassifier(min_samples_leaf=optimal_bins, n_jobs=-1)
        else:
            self.model = model

        # Compute non-compressible samples
        self.y_hat      = self.model.predict(self.X_)          # Predicted values with model
        self.an_all     = np.where(self.y_ != self.y_hat)[0]   # All anomalies
        self.an_smaller = np.where(self.y_ < self.y_hat)[0]    # Actual value is smaller than predicted
        self.an_greater = np.where(self.y_ > self.y_hat)[0]    # Actual value is greater than predicted


    def get_anomalies(self, an_type="all"):
        """
        Get the list of anomalies

        Parameters
        ----------
        an_type : the type of anomalies in which we are interested.
                  Possible values are "all", "smaller", "greater".

        Returns
        -------
        A list with the indices of the samples that are considered anomalies
        """

        check_is_fitted(self)

        valid_types = ("all", "smaller", "greater")

        if an_type not in valid_types:
            raise ValueError("Valid options for 'X_type' are {}. "
                             "Got vartype={!r} instead."
                             .format(valid_types, an_type))        

        if an_type == "all":
            return self.an_all
        elif an_type == "smaller":
            return self.an_smaller
        else:
            return self.an_greater


    def get_classes(self, n_dims=1, an_type="all", filter_balancedness=True, filter_redundancy=True, filter_repeated_attrs=True):
        """
        Classify the anomalies in groups given their characteristics
        
        Parameters
        ----------
        n_dims : Dimension (number of attributes) used during the classification.
                 Current accepted values are 1 or 2.
        an_type : the type of anomalies in which we are interested.
                  Possible values are "all", "smaller", "greater".

        filter_balancedness   : highly unbalanced clusters are fitered out.
        filter_redundancy     : highly redundant clusters are filtered out.
        filter_repeated_attrs : in case of 2 dimensions, be sure that the clusters
                                do not repeat the same attribute.
            
        Returns
        -------
        
        """

        check_is_fitted(self)

        valid_types = ("all", "smaller", "greater")

        if an_type not in valid_types:
            raise ValueError("Valid options for 'X_type' are {}. "
                             "Got vartype={!r} instead."
                             .format(valid_types, an_type))        

        if an_type == "all":
            anomalies = self.an_all
        elif an_type == "smaller":
            anomalies = self.an_smaller
        else:
            anomalies = self.an_greater

        anomalies = np.array(anomalies)
        km_model  = KMeans(n_clusters=2)

        # Compute all possible clusters

        df = pd.DataFrame(columns = ["Attribute1", "Attribute2", "Inertia", "N Class 0", "N Class 1", "Ratio"])

        if n_dims == 1:

            for i in np.arange(self.X_.shape[1]):
        
                new_X = self.X_[anomalies, i].reshape(-1, 1)
                km_model.fit(new_X)
                y_pred = km_model.predict(new_X)
                n_class0 = np.sum(y_pred==0)
                n_class1 = np.sum(y_pred==1)
                tmp_df = pd.DataFrame([{"Attribute1": i, "Attribute2": None, 
                                        "Inertia": km_model.inertia_,
                                        "N Class 0":n_class0, "N Class 1":n_class1,
                                        "Ratio": n_class0 / (n_class0 + n_class1)}])
        
                df = pd.concat([df, tmp_df], ignore_index=True)

        else:

            for i in np.arange(len(self.X_.shape[1])-1):
                for j in np.arange(i+1, len(self.X_.shape[1])):
        
                    new_X = self.X_[anomalies, [i,j]]
                    km_model.fit(new_X)
                    y_pred = km_model.predict(new_X)
                    n_class0 = np.sum(y_pred==0)
                    n_class1 = np.sum(y_pred==1)
                    tmp_df = pd.DataFrame([{"Attribute1": i, "Attribute2": j, 
                                            "Inertia": km_model.inertia_,
                                            "N Class 0":n_class0, "N Class 1":n_class1,
                                            "Ratio": n_class0 / (n_class0 + n_class1)}])
        
                    df = pd.concat([df, tmp_df], ignore_index=True)

        # Filter repeated attributes (only if n_dims > 1)

        if n_dims > 1 and filter_repeated_attrs:

            attr_in_use = list()
            filtered_df = pd.DataFrame(columns = ["Attribute 1", "Attribute 2", "Cluster", "Inertia"])

            for index, row in df.sort_values(by=['Inertia']).iterrows():
    
                if (row["Attribute 1"] in attr_in_use) or (row["Attribute 2"] in attr_in_use):
                    continue
        
                attr_in_use.append(row["Attribute 1"])
                attr_in_use.append(row["Attribute 2"])
    
                tmp_df      = pd.DataFrame([{"Attribute 1": row["Attribute 1"], "Attribute 2": row["Attribute 2"], "Cluster": row["Cluster"], "Inertia": row["Inertia"]}])
                filtered_df = filtered_df.append(tmp_df, ignore_index=True)

            df = filtered_df

        # Filter non-balanced clusters

        if filter_balancedness:
            df = df[(df['Ratio'] > self.balancedness_low) & (df['Ratio'] < self.balancedness_high)]

        # Filter attributes highly correlated

        if filter_redundancy:

            mscd = Miscoding(y_type="categorical")
            mscd.fit(self.X_, self.y_)
            red_matrix = mscd.features_matrix()

            # Contains a list of redundant attributes
            redundant = list()

            for index, row in df.sort_values(by=['Inertia']).iterrows():

                index = int(index)

                if index in redundant:
                    continue

                new_red   = list(np.where(red_matrix[index] < self.redundancy)[0])    
                redundant = list(set(redundant) | set(new_red))

            redundant = np.intersect1d(df.index, redundant)
            df = df.drop(index=redundant)

        return df


    def get_class_points(self, attribute1, attribute2=None, an_type="all"):
        """
        Get the list of anomalies

        Parameters
        ----------
        attribute1: The first dimension of the anomalies
        attribute2: (optional) The second dimension of the anomalies
        an_type :   the type of anomalies in which we are interested.
                    Possible values are "all", "smaller", "greater".

        Returns
        -------
        A list with the indices of the samples that are considered anomalies
        """

        check_is_fitted(self)

        valid_types = ("all", "smaller", "greater")

        if an_type not in valid_types:
            raise ValueError("Valid options for 'X_type' are {}. "
                             "Got vartype={!r} instead."
                             .format(valid_types, an_type))        

        if an_type == "all":
            anomalies = self.an_all
        elif an_type == "smaller":
            anomalies = self.an_smaller
        else:
            anomalies = self.an_greater

        anomalies = np.array(anomalies)
        km_model  = KMeans(n_clusters=2)

        if attribute2 is None:
            new_X = self.X_[anomalies, attribute1].reshape(-1, 1)
        else:
            new_X = self.X_[anomalies, [attribute1, attribute2]]

        km_model.fit(new_X)
        y_pred = km_model.predict(new_X)
        class0 = new_X[y_pred==0]
        class1 = new_X[y_pred==1]

        return class0, class1
