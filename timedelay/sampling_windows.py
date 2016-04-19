import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from sklearn.gaussian_process import GaussianProcess

def pick_sampling_windows(timestamps):
    """Find border indices of sampling windows in data"""

    dt = timestamps[1:] - timestamps[:-1]

    dt_threshold = (np.mean(dt) + np.max(dt)) / 2.

    right_ends = np.append(np.where(dt>=dt_threshold)[0], len(timestamps)-1)
    left_ends = np.append(0, np.where(dt>=dt_threshold)[0]+1)

    windows = np.zeros((len(right_ends), 2), dtype='i4')
    windows[:,0] = left_ends
    windows[:,1] = right_ends

    return windows
