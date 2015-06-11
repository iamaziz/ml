# python packages
from scipy.optimize import fmin_bfgs
import matplotlib.pyplot as plt
import numpy as np


# solutions
from plotData import *
from plotDecisionBoundary import *
from costFunctionReg import *
from mapFeature import *
from predict import *

#---
# Logisitc Regression for a non-linear (regularized) classifier
#---

def ex2_reg():

    #%% Load Data
    #%  The first two columns contains the exam scores and the third column
    #%  contains the label.
    data = np.loadtxt('data/ex2data2.txt', delimiter=',')
    x = data[:, :2]
    y = data[:, 2]

    # %% ==================== Part 1: Plotting ====================
    #%  We start the exercise by first plotting the data to understand the
    #%  the problem we are working with.
    print(
        'Plotting data with o indicating (y = 1) examples and x indicating (y = 0) examples.\n')
    plotData(x, y)

    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend(['y == 1', 'y == 0'], bbox_to_anchor=(1.5, 1))
    plt.show()

    # %% ============ Part 2: Compute Cost and Gradient ============
    #% Add intercept term to x and X_test
    print(x.shape)
    X = mapFeature(x[:, 0], x[:, 1])

    [m, n] = X.shape

    # % Set regularization parameter lambda to 1
    _lambda = 1

    #% Initialize fitting parameters
    initial_theta = np.zeros(n)

    #% Compute and display initial cost and gradient
    # cost = costFunction(initial_theta, X, y)
    # [cost, grad] = costFunctionReg(initial_theta, X, y, _lambda)
    cost = costFunctionReg(initial_theta, X, y, _lambda)
    print('Cost at initial theta (zeros):\n{}'.format(cost))
    print('Gradient at initial theta (zeros):\n{}'.format(initial_theta))
    print(X.shape)

    # ============= Part 3: Optimizing using scipy.optimize
    # %  In this exercise, you will use a function (scipy.optimize.minimize)
    # %  to find the optimal parameters theta.
    fReg = lambda t: costFunctionReg(t, X, y, _lambda)

    # Using minimize()
    # options = {'maxiter': 400, 'disp': True}
    # try other methods `Powell`, `SLSQP`..etc
    # result = minimize(fReg, initial_theta, method='BFGS', options=options)
    # cost = result['fun']
    # theta = result['x']

    # print('\nCost at theta found by minimize(): {}'.format(cost))
    # print('theta: {}'.format(theta))

    # Using fmin_bfgs()
    options = {'full_output': True, 'retall': True}
    theta, cost, _, _, _, _, _, allvecs = fmin_bfgs(
        fReg, initial_theta, maxiter=400, **options)

    print('\nCost at theta found by fmin_bfgs(): {}'.format(cost))
    print('theta: {}'.format(theta))
    # visualizing the cost change
    costs = [fReg(allvecs[i]) for i in range(157)]
    plt.plot(costs)
    plt.title('cost function $y$ per iteration $x$')
    plt.grid()
    plt.show()
    # % Print theta to screen
    print('Cost at theta found by minimize():\n{}\n'.format(cost))
    print('theta: \n')
    print('{}\n'.format(theta))

    # % Plot Boundary
    plotDecisionBoundary(theta, X, y)

    # % Show plot
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend(
        ['y == 1', 'y == 0', 'Decision Boundary'], bbox_to_anchor=(1.5, 1))
    plt.show()
    # % Compute accuracy on our training set
    p = predict(theta, X)
    print('Train Accuracy: {}\n'.format(np.mean(np.double(p == y)) * 100))

