import numpy as np
import cPickle as pickle
from matplotlib.mlab import frange
from timedelay.gp import *
from timedelay.filtering import *

dt=0.1
system_type="double"

data = np.load("TimeDelayData/pairs_with_truths_and_windows.npz")['arr_0']
pair_ids = np.unique(data['full_pair_id'])
pair_ids = filter_pairs(pair_ids, system_type)
pair_ids = np.intersect1d(pair_ids, np.unique(data[data['window_id'] < 2320.0]['full_pair_id']))

outdata = resample_using_gp_models(data, pair_ids)
np.savez('TimeDelayData/gp_resampled_of_windows_with_truth', outdata)
