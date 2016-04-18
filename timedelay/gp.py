import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from sklearn.gaussian_process import GaussianProcess

def make_a_model(pairNo, x, X, y, dy, theta0=1e-3, thetaL=1e-3, thetaU=1):
    """Make a GaussianProcess model for noisy data"""

    # If the original experiment is known to be infinitely
    # differentiable (smooth), then one should use the
    # squared-exponential correlation model.
    gp = GaussianProcess(corr='squared_exponential',
                         regr = "quadratic", #?
                         theta0 = theta0,
                         thetaL = thetaL,
                         thetaU = thetaU,
                         nugget = (dy / y) ** 2, #?
                         random_start=500)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X, y)

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, MSE = gp.predict(x, eval_MSE=True)
    sigma = np.sqrt(MSE)
    return y_pred, sigma

def make_a_perfect_model(pairNo, x, X, y):
    """Make a GaussianProcess model for data without noise (It
    complains for TDC dataset though!)"""
    gp = GaussianProcess(theta0=1e-3,
                         thetaL=1e-3,
                         thetaU=1,
                         random_start=500)
    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X, y)

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, MSE = gp.predict(x, eval_MSE=True)
    sigma = np.sqrt(MSE)
    return y_pred, sigma
