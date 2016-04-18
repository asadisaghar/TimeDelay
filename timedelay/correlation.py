import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from sklearn.gaussian_process import GaussianProcess

def cross_correlate_models(model1, model2, mode='same'):
    """Cross correlate two arrays (models or data!) and return the
    index of the maximum correlation and the full corr array"""

    corr = correlate(model1, model2, mode=mode)
    abscorr = np.abs(corr)
    maxcorr = np.max(abscorr)
    return np.where(abscorr == maxcorr)[0], corr
