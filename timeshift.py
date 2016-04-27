import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.signal import correlate
from timedelay.correlation import *
from scipy import signal
import sys
import math

def timeshift_mse(x1, x2):
    res = np.zeros(len(x1) * 2 - 1)
    xlen = len(x1)
    off = xlen / 2
    for i in xrange(-xlen+1, xlen):
        left = max(0, -i)
        right = max(0, i)
        xr = x1[left:xlen-right] - x2[right:xlen-left]
        res[i] = np.sum(xr**2) / len(xr)
    return res


def timeshift_correlate(sig1, sig2):
    return signal.correlate(sig1, sig2, mode="full")

def timeshift(data, correlator, dt = 0.1, windows = None, detrend = False):
    output = None
    if windows is None:
        windows = np.unique(data['window_id'])
    for i, window in enumerate(windows):
        print "Window %s" % window
        t = data['t_eval'][data['window_id']==window]
        t = t - np.min(t)
        sigA = data['sig_evalA'][data['window_id']==window]
        sigB = data['sig_evalB'][data['window_id']==window]
        sigA = (sigA - np.mean(sigA)) / np.std(sigA)
        sigB = (sigB - np.mean(sigB)) / np.std(sigB)
        if detrend:
            sigA = scipy.signal.detrend(sigA)
            sigB = scipy.signal.detrend(sigB)

        corr = correlator(sigA, sigB)

        res = np.zeros(len(corr), dtype=[('pair_id', 'f4'), ('window_id', 'f4'), ('offset', 'f4'), ('correlation', 'f4'), ('dt', '<f4'), ('tau', '<f4'), ('sig', '<f4'), ('m1', '<f4'), ('m2', '<f4')])

        res['window_id'] = window
        for name in ('pair_id', 'dt', 'tau', 'sig', 'm1', 'm2'):
            res[name] = data[name][data['window_id']==window][0]

        res['offset'] = dt * np.arange(-len(sigA)+1, len(sigA), 1)
        res['correlation'] = corr

        if output is None:
            output = res
        else:
            output = np.append(output, res)
    return output

argv = sys.argv[1:]
if not argv:
    print """Usage:
    python timeshift.py METHOD OUTPUT.npz
    python timeshift.py METHOD OUTPUT.npz BUCKET TOTAL_BUCKETS

If buckets are specified, divide the data into that many buckets, and
only process the specified bucket.

METHOD is either mse or correlate, METHOD could be mse-detrend or
correlate-detrend to use detrend after normalization.
"""
    sys.exit(-1)

method = argv[0]
output = argv[1]

bucket = None
buckets = None
if len(argv) > 2:
    bucket = int(argv[2])
    buckets = int(argv[3])

data = np.load("TimeDelayData/gp_resampled_of_windows_with_truth.npz")['arr_0']

windows = None
if buckets is not None:
    windows = np.unique(data['window_id'])

    bucketlen = math.ceil(len(windows) / float(buckets))
    start_win = bucket * bucketlen
    end_win = (bucket+1)*bucketlen
    windows = windows[start_win:end_win]
    print "Working on subset: %s - %s (%s total)" % (windows[0], windows[-1], end_win - start_win)

if method == 'mse':
    data = timeshift(data, timeshift_mse, windows=windows)
elif method == 'correlate':
    data = timeshift(data, timeshift_correlate, windows=windows)
elif method == 'mse-detrend':
    data = timeshift(data, timeshift_mse, windows=windows, detrend=True)
elif method == 'correlate-detrend':
    data = timeshift(data, timeshift_correlate, windows=windows, detrend=True)

else:
    raise Exception("Unknown correlation mode")

np.savez(output, data)
