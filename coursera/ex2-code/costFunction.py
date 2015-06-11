

import numpy as np

from sigmoid import *

def costFunction(theta, X, y):
    '''costFunction() - Logistic Regression Cost Function'''

    m = len(X)
    z = np.dot(X, theta.T)
    h = sigmoid(z)
    pos = np.dot(-y, np.log(h))
    neg = np.dot(1 - y, np.log(1 - h))
    cost = 1. / m * (pos - neg)
    
    return cost
