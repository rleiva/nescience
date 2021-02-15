"""
utils.py

Fast auto machine learning
with the minimum nescience principle

@author:    Rafael Garcia Leiva
@mail:      rgarcialeiva@gmail.com
@web:       http://www.mathematicsunknown.com/
@copyright: GNU GPLv3
@version:   0.8
"""

import numpy as np
import warnings
														
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import LabelEncoder


__all__ = ["discretize_vector", "unique_count", "optimal_code_length"]


"""
Discretize a continous variable using a "uniform" strategy
    
Parameters
----------
x  : array-like, shape (n_samples)
       
Returns
-------
A new discretized vector of integers.
"""
def discretize_vector(x, n_bins=None):

    length = x.shape[0]
    new_x  = x.copy().reshape(-1, 1)

    # Optimal number of bins
    if n_bins is None:
        optimal_bins = int(np.cbrt(length))
    else:
        optimal_bins = n_bins
    
    # Correct the number of bins if it is too small
    if optimal_bins <= 1:
        optimal_bins = 2
    
    # Repeat the process until we have data in all the intervals

    total_bins    = optimal_bins
    previous_bins = 0
    stop          = False

    while stop == False:

        # Avoid those annoying warnings
        with warnings.catch_warnings():
            
            warnings.simplefilter("ignore")

            est = KBinsDiscretizer(n_bins=total_bins, encode='ordinal', strategy="uniform")
            est.fit(new_x)
            tmp_x = est.transform(new_x)[:,0].astype(dtype=int)

        y = np.bincount(tmp_x)
        actual_bins = len(np.nonzero(y)[0])

        if previous_bins == actual_bins:
            # Nothing changed, better stop here
            stop = True

        if actual_bins < optimal_bins:
            # Too few intervals with data
            add_bins      = optimal_bins - actual_bins
            previous_bins = actual_bins
            # add_bins      = int( np.round( (length * (1 - actual_bins / optimal_bins)) / optimal_bins ) )
            total_bins    = total_bins + add_bins
        else:
            # All intervals have data
            stop = True

    new_x = est.transform(new_x)[:,0].astype(dtype=int)

    return new_x


"""
Count the number of occurences of a discretized 1d or 2d space
for classification or regression problems
    
Parameters
----------
x1, x2, x3: array-like, shape (n_samples)
numeric1, numeric2, numeric3: if the variable is numeric or not
       
Returns
-------
A vector with the frequencies of the unique values computed.
"""
def unique_count(x1, numeric1, x2=None, numeric2=None, x3=None, numeric3=None):

    # Process first variable

    if not numeric1:

        # Econde categorical values as numbers
        le = LabelEncoder()
        le.fit(x1)
        x1 = le.transform(x1)

    else:
        x1 = discretize_vector(x1)

    # Process second variable

    if x2 is not None:

        if not numeric2:

            # Econde categorical values as numbers
            le = LabelEncoder()
            le.fit(x2)
            x2 = le.transform(x2)

        else:

            # Discretize variable
            x2 = discretize_vector(x2)

        x = (x1 + x2) * (x1 + x2 + 1) / 2 + x2
        x = x.astype(int)

        # Process third variable

        if x3 is not None:

            if not numeric3:

                # Econde categorical values as numbers
                le = LabelEncoder()
                le.fit(x3)
                x3 = le.transform(x3)

            else:

                # Discretize variable
                x3 = discretize_vector(x3)

            x = (x + x3) * (x + x3 + 1) / 2 + x3
            x = x.astype(int)        

    else:
        
        x = x1

    # Return count
    
    y     = np.bincount(x)
    ii    = np.nonzero(y)[0]
    count = y[ii]

    return count


"""
Compute the length of a list of features (1d or 2d)
and / or a target variable (classification or regression)
using an optimal code using the frequencies of the categorical variables
or a discretized version of the continuous variables
    
Parameters
----------
x1, x2, x3: array-like, shape (n_samples)
numeric1, numeric2, numeric3: if the variable is numeric or not

Returns
-------
Return the length of the encoded dataset (float)
"""
def optimal_code_length(x1, numeric1, x2=None, numeric2=None, x3=None, numeric3=None):

    count = unique_count(x1=x1, numeric1=numeric1, x2=x2, numeric2=numeric2, x3=x3, numeric3=numeric3)
    ldm = np.sum(count * ( - np.log2(count / len(x1) )))
    
    return ldm
