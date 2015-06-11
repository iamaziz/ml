
import numpy as np

from sigmoid import *


def predict(theta, X):
    '''predict() - Logistic Regression Prediction Function'''
    m = len(X)
    p = np.zeros(m)

    predictions = sigmoid(np.dot(X, theta))
    pos = np.where(predictions >= 0.5)
    neg = np.where(predictions < 0.5)
    p[pos] = 1

    return p
