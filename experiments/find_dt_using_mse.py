import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, detrend
from timedelay.correlation import *


def mse_correlate(x1, x2):
    res = np.zeros(len(x1))
    xlen = len(x1)
    off = xlen / 2
    for i in xrange(0, xlen):
        p = i - off
        left = max(0, -p)
        right = max(0, p)
        xr = x1[left:xlen-right] - x2[right:xlen-left]
        res[i] = np.sum(xr**2) / len(xr)
    return res



data = np.load("TimeDelayData/gp_resampled_of_windows_with_truth.npz")['arr_0']
pair_ids = np.unique(data['pair_id'])
dt_trues = np.zeros(len(pair_ids))
dt_ests = np.zeros(len(pair_ids))
dt = 0.1

for j, pair_id in enumerate(pair_ids):
    print "%s of %s" % (j, len(pair_ids))
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
        
        corr = mse_correlate(sigA, sigB)
        corrval += -np.argmax(corr)
        
    dt_ests[j] = corrval * dt / len(windows)
#    print  "pair time delay: %s" % time_delays[j] # this value changes w/ or w/o detrend, centering, normalization and their combinations. Why/why not use each??


res = np.zeros(len(pair_ids), dtype=[('full_pair_id', 'f4'), ('dt', 'f4'), ('est', 'f4')])
res['full_pair_id'] = pair_ids
res['dt'] = dt_trues
res['est'] = dt_ests
np.savez("TimeDelayData/dt_using_mse.npz", res)


#plt.plot(dt_trues, dt_ests, 'o')
#plt.plot(dt_trues, dt_trues, '-k', alpha=0.5)
#plt.show()
