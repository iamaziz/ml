
import matplotlib.pyplot as plt
import numpy as np

from plotData import *
from mapFeature import *

def plotDecisionBoundary(theta, X, y):
    '''plotDecisionBoundary() - Function to plot classifier decision boundary'''

    plotData(X[:, 1:], y)

    if len(X.T) <= 3:
        # linear boundary
        x = np.array([X[:, 1].min(), X[:, 1].max()])
        y = lambda x: (theta[0] + theta[1] * x) / - theta[2]

        plt.plot(x, y(x))
        plt.xlabel('Exam 1 score')
        plt.ylabel('Exam 2 score')
        plt.legend(
            ['Decision boundary', 'y = 1', 'y = 0'], bbox_to_anchor=(1.5, 1))

    else:
        # non-linear boundary
        #% Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros(shape=(len(u), len(v)))
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = (
                    mapFeature(np.array(u[i]), np.array(v[j])).dot(np.array(theta)))
        z = z.T  # % important to transpose z before calling contour
        #% Plot z = 0
        #% Notice you need to specify the range [0, 0]
        CS = plt.contour(u, v, z, 1)
        CS.collections[0].set_label('Decision Boundary')
        plt.xlabel('Microchip Test 1')
        plt.ylabel('Microchip Test 2')
