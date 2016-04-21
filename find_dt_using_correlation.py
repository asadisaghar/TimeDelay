import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, detrend
from timedelay.correlation import *

data = np.load("TimeDelayData/gp_resampled_of_windows_with_truth.npz")['arr_0']
pair_ids = np.unique(data['pair_id'])
dt_trues = np.zeros(len(pair_ids))
dt_ests = np.zeros(len(pair_ids))
dt = 0.1

for j, pair_id in enumerate(pair_ids):
    corrval = 0
    print "Correlating pair %s" % pair_id
    pair_data = data[data['pair_id'] == pair_id]
    dt_trues[j] = np.unique(pair_data['dt'])[0]
    windows = np.unique(pair_data['window_id'])

    for i, window in enumerate(windows):
        t = data['t_eval'][data['window_id']==window]
        t = t - np.min(t)
        sigA = data['sig_evalA'][data['window_id']==window]
        #    sigA = detrend(sigA)
        sigA = (sigA - np.mean(sigA)) / np.std(sigA)

        sigB = data['sig_evalB'][data['window_id']==window]
        #    sigB = detrend(sigB)
        sigB = (sigB - np.mean(sigB)) / np.std(sigB)
        
        maxcorr, corr = cross_correlate_models(sigA, sigB)
        corrval += cross_correlate_models(sigA, sigB, mode='valid')
        
    dt_ests[j] = corrval * dt / len(windows)
#    print  "pair time delay: %s" % time_delays[j] # this value changes w/ or w/o detrend, centering, normalization and their combinations. Why/why not use each??

plt.plot(dt_trues, dt_ests, 'o')
plt.plot(dt_trues, dt_trues, '-k', alpha=0.5)
plt.show()
