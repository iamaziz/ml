# python packages
from scipy.optimize import fmin
import matplotlib.pyplot as plt
import numpy as np


# solutions
from plotData import *
from plotDecisionBoundary import *
from sigmoid import *
from costFunction import *
from predict import *

#-----
# Logistic Regression with a linear classifier
#-----


def ex2():
    #%% Load Data
    #%  The first two columns contains the exam scores and the third column
    #%  contains the label.
    data = np.loadtxt('data/ex2data1.txt', delimiter=',')
    x = data[:, :2]
    y = data[:, 2]
    #%% ==================== Part 1: Plotting ====================
    #%  We start the exercise by first plotting the data to understand the
    #%  the problem we are working with.
    print(
        'Plotting data with o indicating (y = 1) examples and x indicating (y = 0) examples.\n')
    plotData(x, y)

    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')
    plt.legend(['Admitted', 'Not admitted'], bbox_to_anchor=(1.5, 1))
    plt.show()
    #%% ============ Part 2: Compute Cost and Gradient ============
    #%  In this part of the exercise, you will implement the cost and gradient
    #%  for logistic regression. You neeed to complete the code in costFunction()

    #%  Setup the data matrix appropriately, and add ones for the intercept term
    [m, n] = x.shape

    #% Add intercept term to x and X_test
    ones = np.ones(m)
    X = np.array([ones, x[:, 0], x[:, 1]]).T

    #% Initialize fitting parameters
    initial_theta = np.zeros(n + 1)

    #% Compute and display initial cost and gradient
    cost = costFunction(initial_theta, X, y)
    print('Cost at initial theta (zeros):\n{}'.format(cost))
    # ============= Part 3: Optimizing using fmin() or minimize()
    print('Gradient at initial theta (zeros):\n{}'.format(initial_theta))
    # %  In this exercise, you will use a built-in function (scipy.optimize.fmin) to find the
    # %  optimal parameters theta.
    f = lambda t: costFunction(t, X, y)  # %  Set options for fmin()
    fmin_opt = {'full_output': True, 'maxiter': 400, 'retall': True}
    # %  Run fmin to obtain the optimal theta
    theta, cost, iters, calls, warnflag, allvecs = fmin(
        f, initial_theta, **fmin_opt)
    print('Cost at theta found by fmin(): {}'.format(cost))
    print('theta: {}'.format(theta))

    # %  Set options for minimize()
    # mini_opt = {'maxiter': 400, 'disp': True}
    # %  Run minimize to obtain the optimal theta
    # results = minimize(f, initial_theta, method='Nelder-Mead', options=mini_opt)
    # cost = results['fun']
    # theta = results['x']
    # print('Cost at theta found by minimize(): {}'.format(cost))
    # print('theta: {}'.format(theta))

    cost_change = [costFunction(allvecs[i], X, y) for i in range(156)]
    plt.plot(cost_change)
    plt.grid()
    plt.title('cost function $y$ per iteration $x$')
    plt.show()  # % Print theta to screen
    print('Cost at theta found by fmin:\n{}\n'.format(cost))
    print('theta: \n')
    print('{}\n'.format(theta))

    # % Plot Boundary
    plotDecisionBoundary(theta, X, y)

    # % Show plot
    plt.show()

    # %% ============== Part 4: Predict and Accuracies ==============
    # %  After learning the parameters, you'll like to use it to predict the outcomes
    # %  on unseen data. In this part, you will use the logistic regression model
    # %  to predict the probability that a student with score 45 on exam 1 and
    # %  score 85 on exam 2 will be admitted.
    # %
    # %  Furthermore, you will compute the training and test set accuracies of
    # %  our model.
    # %
    # %  Your task is to complete the code in predict.m

    # %  Predict probability for a student with score 45 on exam 1
    # %  and score 85 on exam 2

    scores = np.array([1, 45, 85])
    prob = sigmoid(np.dot(scores, theta))
    print(
        'For a student with scores 45 and 85, we predict an admission probability of:\n{}\n\n'.format(prob))

    # % Compute accuracy on our training set
    p = predict(theta, X)

    print('Train Accuracy: {}\n'.format(np.mean(np.double(p == y)) * 100))
