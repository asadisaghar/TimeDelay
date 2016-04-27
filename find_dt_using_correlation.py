import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr
from timedelay.correlation import *

data = np.load("TimeDelayData/gp_resampled_of_windows_with_truth.npz")['arr_0']
data = data[:500000]
pair_ids = np.unique(data['pair_id'])
dt_trues = np.zeros(len(pair_ids))
dt_ests = np.zeros(len(pair_ids))
dt_est_err = np.zeros(len(pair_ids))
def correlate_to_timeshift(sig_corr, t):
    #Generate an x axis
    xcorr = np.arange(len(sig_corr))
    #Convert this into lag units, but still not really physical
    lags = xcorr - ((len(sig_corr) + 1) / 2 - 1) # because len(correlate(sig1, sig2, mode='full')) = len(sig1) + len(sig2) - 1
    distancePerLag = (t[-1] - t[0]) / float(len(t))  #This is just the t-spacing
    #Convert your lags into physical units
    offsets = -lags * distancePerLag
    maxcorr = np.argmax(sig_corr)
    timeDelay = offsets[maxcorr]
    return timeDelay

for j, pair_id in enumerate(pair_ids):
    pair_data = data[data['pair_id'] == pair_id]
    dt_trues[j] = np.unique(pair_data['dt'])[0]
    windows = np.unique(pair_data['window_id'])
    dt_maxcorr = np.array([])

    for i, window in enumerate(windows):
        t = data['t_eval'][data['window_id']==window]
        dt_window = t.max() - t.min()
        sigA = data['sig_evalA'][data['window_id']==window]
        sigB = data['sig_evalB'][data['window_id']==window]

        # Cenrter, normalize, detrend signals. not sure which one makes more sense to do
        sigA = signal.detrend(sigA)
        sigA = (sigA - np.mean(sigA)) / np.std(sigA)
        sigB = signal.detrend(sigB)
        sigB = (sigB - np.mean(sigB)) / np.std(sigB)

        sig_corr = signal.correlate(sigA, sigB, mode="full")
        # translate correlation signal into time shift
        dt_maxcorr_window = correlate_to_timeshift(sig_corr, t)
        dt_maxcorr = np.append(dt_maxcorr, dt_maxcorr_window)

    dt_ests[j] = np.mean(dt_maxcorr)
    dt_est_err[j] = np.std(dt_maxcorr)
    print "dt_est: ", dt_ests[j], "p/m: ", dt_est_err[j], " - dt_true: ", dt_trues[j]

res = np.zeros(len(pair_ids), dtype=[('full_pair_id', 'f4'), ('dt', 'f4'), ('est', 'f4'), ('est_err', 'f4')])
res['full_pair_id'] = pair_ids
res['dt'] = dt_trues
res['est'] = dt_ests
#np.savez("TimeDelayData/dt_using_correlate_detrended_centered_normalized.npz", res)

plt.errorbar(dt_trues, dt_ests, yerr=dt_est_err, fmt='o')
#plt.plot(dt_trues, dt_ests, 'o')
plt.plot(dt_trues, dt_trues, '-k', alpha=0.5)
plt.show()
