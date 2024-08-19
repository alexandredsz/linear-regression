import numpy as np
import matplotlib.pyplot as plt

def estimate_regression_coeficients(x, y):
    """
    Estimate the coefficients of a simple linear regression.

    Parameters:
    x (numpy array): An array of predictor (independent variable) values.
    y (numpy array): An array of response (dependent variable) values.

    Returns:
    gradient (float): The slope of the regression line.
    y_intercept (float): The intercept of the regression line.
    """

    n = np.size(x)

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    ssxy = sum(x * y) - (n * mean_x * mean_y)
    ssxx = sum(x**2) - (n * (mean_x**2))

    gradient = ssxy / ssxx
    y_intercept = mean_y - gradient * mean_x

    return gradient, y_intercept

def plot_linear_regression(x, y, y_intercept, gradient):
    """
    Plot the linear regression line along with the data points.

    Parameters:
    x (numpy array): An array of predictor (independent variable) values.
    y (numpy array): An array of response (dependent variable) values.
    y_intercept (float): The intercept of the regression line.
    gradient (float): The slope of the regression line.
    """

    plt.scatter(x, y, color='0')

    y_pred = gradient * x + y_intercept

    plt.plot(x, y_pred, color='r')

    plt.xlabel('x')
    plt.ylabel('y')