# BismiLlahirrahmanirrahim
# 27-Dec-2023
# Update: 28-Dec-2023

# ABOUT THE SCRIPT
# ----------------
"""
Author                  : Rizal Purnawan
Date of first creation  : 12/27/2023

Description
-----------

This module contains the basic statistical tools such as algorithms
for computing expectation, covariance, correlations, r-norm and
coefficient of determination.
"""

# 0. REQUIRED LIBRARIES
# ---------------------
from math import sqrt
import numpy as np

# 1. EXPECTATION OPERATOR
# -----------------------
def E(X):
    return sum(X) / len(X)

# 2. COVARIANCE
# -------------
def cov(X, Y):
    # Generating the expectations as constant random variables:
    EX = [E(X)] *len(X)
    EY = [E(Y)] *len(Y)

    # Computing covariance:
    XY = [x *y for x, y in zip(X, Y)]
    cov_XY = E(XY) - E(X) *E(Y)

    return cov_XY

# 3. PEARSON CORRELATION
# ----------------------
def corr_Pearson(X, Y):
    if all(len(set(U)) > 1 for U in [X, Y]):
        return cov(X, Y) / sqrt(cov(X, X) *cov(Y, Y))
    else:
        print("ERROR: Cannot compute almost surely constant random variables!")
        print("INFO: Please refer to the theoretical documentation!")
        raise ValueError

# 4. SPEARMAN CORRELATION
# -----------------------
def corr_Spearman(X, Y):
    if all(len(set(U)) > 1 for U in [X, Y]):
        # 1. Computing rX (the rank of X):
        #    Auxilliary iterables are required.
        X_uniq = sorted(list(set(X)))
        X_dict = dict(zip(X_uniq, range(1, len(X_uniq) + 1)))
        rX = [X_dict[x] for x in X]     # The rank variable of X

        # 2. Computing rY (the rank of Y):
        #    Auxilliary iterables are required.
        Y_uniq = sorted(list(set(Y)))
        Y_dict = dict(zip(Y_uniq, range(1, len(Y_uniq) + 1)))
        rY = [Y_dict[y] for y in Y]     # The rank variable of Y

        # 3. Computing covariance of rank variables:
        cov_XY = cov(rX, rY)
        cov_XX = cov(rX, rX)
        cov_YY = cov(rY, rY)

        # 4. Computing Spearman correlation:
        return cov_XY / sqrt(cov_XX *cov_YY)
    else:
        print("ERROR: Cannot compute almost surely constant random variables!")
        print("INFO: Please refer to the theoretical documentation!")
        raise ValueError
    
# 5. r-NORM
# ---------
def squared_rNorm(Y, Y_):
    """
    Description
    -----------

    > Y     : Variable from the data
    > Y_    : Statistical model
    """

    # Requirement:
    if all(type(U) in [list, np.ndarray] for U in [Y, Y_]) \
            and len(Y) == len(Y_):
        pass
    else:
        print(">>> ERROR: Invalid 'Y' and 'Y_'!")
        raise ValueError
    
    # Main algorithm:
    Y, Y_ = list(Y), list(Y_)
    EY = [E(Y)] *len(Y)
    A = sum([(y - y_)**2 for y, y_ in zip(Y, Y_)])
    B = sum([(y - ey)**2 for y, ey in zip(Y, EY)])
    return A / B

# 6. COEFFICIENT OF DETERMINATION
# -------------------------------
def R2(Y, Y_):
    """
    Description
    -----------

    > Y     : Variable from the data
    > Y_    : Statistical model
    """
    # Requirement:
    if all(type(U) in [list, np.ndarray] for U in [Y, Y_]) \
            and len(Y) == len(Y_):
        pass
    else:
        print(">>> ERROR: Invalid 'Y' and 'Y_'!")
        raise ValueError
    
    # Main algorithm:
    return 1 - squared_rNorm(Y, Y_)
