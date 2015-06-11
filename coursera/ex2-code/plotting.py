
import matplotlib.pyplot as plt
import numpy as np


def plotData(X, y):
    '''plotData() - Function to plot 2D classification data'''

    pos = np.where(y == 1)
    neg = np.where(y == 0)
    plt.scatter(X[pos, 0], X[pos, 1], c='g', marker='o')
    plt.scatter(X[neg, 0], X[neg, 1], c='r', marker='x')

    plt.grid()
