import numpy as np
import cPickle as pickle
from matplotlib.mlab import frange
from timedelay.gp import *
from timedelay.filtering import *

dt = 0.1
system_type="double"

data = np.load("TimeDelayData/pairs_with_truths_and_windows.npz")['arr_0']
outdata = resample_using_gp_models(data, dt = dt)
np.savez('TimeDelayData/gp_resampled_of_windows_with_truth', outdata)
