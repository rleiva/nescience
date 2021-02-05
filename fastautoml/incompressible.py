import pandas as pd
import numpy  as np

import warnings
import math
import re

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin														
														
from sklearn.utils            import check_X_y
from sklearn.utils            import column_or_1d
from sklearn.utils            import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.preprocessing    import KBinsDiscretizer
from sklearn.preprocessing    import LabelEncoder
from sklearn.preprocessing    import MinMaxScaler
from sklearn.calibration      import CalibratedClassifierCV
from sklearn.cluster          import KMeans

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
from sklearn.svm			import SVC

# Supported regressors

from sklearn.linear_model   import LinearRegression
from sklearn.tree           import DecisionTreeRegressor
from sklearn.svm            import LinearSVR
from sklearn.neural_network import MLPRegressor

class IncompressibleClassifier():
    
    # TODO: Class documentation
    
    def __init__(self, auto=False, random_state=None):

        # TODO: Document
        
        self.auto         = auto
        self.random_state = random_state

        return None


    def fit(self, X, y, model=None):

        self.X_, self.y_ = check_X_y(X, y, dtype=None)

        # TODO: check it is a valid model
        # TODO: train if the model if auto = True

        self.model = model

        y_hat = self.model.predict(self.X_)
        self.incompressible = np.where(self.y_ != y_hat)[0]

        return


    def fit_classification(self):

        if self.y_isnumeric:

            regressor = AutoRegressor()

        else:

            model = AutoClassifier()

        return


    def get_incompressible(self):

        # TODO: Document

        return self.incompressible


    def clusters(self, n_clusters="Auto", filter_inertia=True, filter_repeated_attrs=True, filter_balancedness=True, filter_miscoding=True):

        # TODO: Check that the class is fitted
        # TODO: Allow to change the dimension of the cluster

        nis = len(self.incompressible)

        # Automatically select the number of clusters
        if n_clusters == "Auto":
            n_clusters = int(np.log2(nis)/2)

        # TODO : MinMaxScaler() ??
        km_model = KMeans(n_clusters=n_clusters)

        # Compute all possible clusters

        df = pd.DataFrame(columns = ["Attribute 1", "Attribute 2", "Cluster", "Inertia"])

        for i in np.arange(self.X_.shape[1]-1):
    
            for j in np.arange(i+1, self.X_.shape[1]):
        
                new_X = self.X_[np.ix_(self.incompressible,[i, j])]
                # new_X = scaler.fit_transform(new_X)
        
                km_model.fit(new_X)
        
                tmp_df = pd.DataFrame([{"Attribute 1": i, "Attribute 2": j, "Cluster": km_model, "Inertia": km_model.inertia_}])
                df = df.append(tmp_df, ignore_index=True)

        # TODO: Implement a filter based on Inertia

        # Filter repeated attributes

        if filter_repeated_attrs:

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

            filter_ratio_low  = 0.2
            filter_ratio_high = 0.8

            filtered_df = pd.DataFrame(columns = ["Attribute 1", "Attribute 2", "Cluster", "Inertia"])

            for index, row in df.iterrows():

                new_X = self.X_[np.ix_(self.incompressible,[row["Attribute 1"], row["Attribute 2"]])]
                
                km_model = row["Cluster"]
                y_pred = km_model.predict(new_X)

                n_class_0 = np.sum(y_pred == 0)
                n_class_1 = np.sum(y_pred == 1)

                ratio = n_class_0 / (n_class_0 + n_class_1)

                if ratio < filter_ratio_low:
                    continue

                if ratio > filter_ratio_high:
                    continue

                tmp_df      = pd.DataFrame([{"Attribute 1": row["Attribute 1"], "Attribute 2": row["Attribute 2"], "Cluster": row["Cluster"], "Inertia": row["Inertia"]}])
                filtered_df = filtered_df.append(tmp_df, ignore_index=True)

            df = filtered_df

        # Filter attributes highly related

        if filter_miscoding:

            filtered_df = pd.DataFrame(columns = ["Attribute 1", "Attribute 2", "Cluster", "Inertia"])

            filter_miscoding  = 0.2

            mscd = Miscoding()
            mscd.fit(self.X_, self.y_)
            matrix = mscd.features_matrix()

            for index, row in df.iterrows():

                if matrix[row["Attribute 1"], row["Attribute 2"]] > filter_miscoding:
                    continue

                tmp_df      = pd.DataFrame([{"Attribute 1": row["Attribute 1"], "Attribute 2": row["Attribute 2"], "Cluster": row["Cluster"], "Inertia": row["Inertia"]}])
                filtered_df = filtered_df.append(tmp_df, ignore_index=True)

            df = filtered_df

        return df


