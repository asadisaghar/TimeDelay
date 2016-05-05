import numpy as np
import matplotlib.pyplot as plt

def plot_data(ax, pairNo, dt, X, y, dy, ob):
    """Plot data points with error bars."""
    ax.errorbar(X.ravel() + dt, y, dy, fmt='.c', markersize=10, label='Observations ' + ob)
    return ax

def plot_model(ax, pairNo, dt, x, y_pred, sigma, ob):
    """Plot the best-fit model of data along with 95% uncertainties."""

    ax.plot(x + dt, y_pred, '-', c='k', linewidth=2, label='Prediction ' + ob)
    ax.fill(np.concatenate([x + dt, x[::-1] + dt]),
            np.concatenate([y_pred - 1.9600 * sigma,
                            (y_pred + 1.9600 * sigma)[::-1]]),
            alpha=.5, color='darkorange', ec='None', 
            label='95% confidence interval ' + ob)
    return ax
