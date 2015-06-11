
from __future__ import division
import numpy as np

from sigmoid import *


def costFunctionReg(theta, X, y, _lambda):
    '''costFunctionReg() - Regularized Logistic Regression Cost'''

    m = len(X)
    z = np.dot(X, theta.T)
    h = sigmoid(z)

    # cost
    pos = np.dot(-y, np.log(h))
    neg = np.dot(1 - y, np.log(1 - h))
    cost = (pos - neg)

    # regularization term
    reg_para = (_lambda / (2. * m)) * sum([th ** 2 for th in theta[1:]])

    # cost with regularization
    costReg = (1. / m) * (cost + reg_para)

    return costReg
