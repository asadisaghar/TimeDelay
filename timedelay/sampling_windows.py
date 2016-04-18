import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from sklearn.gaussian_process import GaussianProcess

def pick_sampling_windows(timestamps):
    """Find border indices of sampling windows in data"""

    dt = []
    for i in xrange(1, len(timestamps)):
        dt.append(timestamps[i]-timestamps[i-1])
    dt_threshold = np.mean(dt)*1.
    right_ends = np.append(np.where(dt>=dt_threshold)[0], len(timestamps)-1)
    left_ends = np.append(0, np.where(dt>=dt_threshold)[0]+1)
    windows = np.zeros((len(right_ends), 2))
    for i in range(0, len(right_ends)):
        windows[i] = (left_ends[i], right_ends[i])
    return windows
