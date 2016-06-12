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
    idxs = np.arange(-xlen+1, xlen, 1)
    for i in idxs:
        left = max(0, -i)
        right = max(0, i)
        xr = x1[left:xlen-right] - x2[right:xlen-left]
        res[i] = np.sum(xr**2) / len(xr)
    return idxs

def timeshift_pearson(x1, x2):
    res = np.zeros(len(x1) * 2 - 1 - 2)
    xlen = len(x1)
    idxs = np.arange(-xlen+2, xlen-1, 1)
    for i in idxs:
        left = max(0, -i)
        right = max(0, i)
        x1i = x1[left:xlen-right]
        x2i = x2[right:xlen-left]
        res[i] = scipy.stats.pearsonr(x1i, x2i)[0]
        assert not np.isnan(res[i])
    return idxs, res

def timeshift_correlate(sig1, sig2):
    return np.arange(-xlen+1, xlen, 1), signal.correlate(sig1, sig2, mode="full")

# def timeshift_pearson_window(x1, x2, window=0.5):
#     xlen = len(x1)
#     window=int(np.floor(xlen * window))
#     res = np.zeros((xlen-window) * 2 - 1)
#     for i in xrange(-xlen+window+1, xlen-window):
#         left = max(0, -i)
#         right = max(0, i)
#         l = np.abs(right - left)
#         offset = np.floor((l - window) / 2)

#         if l < window:
#             import pdb
#             pdb.set_trace()

#         # x1i = x1[left:xlen-right][offset:][:window]

#         x1i = x1[left+offset:left+offset+window]
#         x2i = x2[right+offset:right+offset+window]
#         res[i] = scipy.stats.pearsonr(x1i, x2i)[0]
#     return res

def timeshift(data, correlator, dt = 0.1, windows = None, detrend = False, negative = False):
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

        if negative:
            off, corr = correlator(sigB, sigA)
        else:
            off, corr = correlator(sigA, sigB)

        res = np.zeros(len(corr), dtype=[('pair_id', 'f4'), ('window_id', 'f4'), ('offset', 'f4'), ('correlation', 'f4'), ('dt', '<f4'), ('tau', '<f4'), ('sig', '<f4'), ('m1', '<f4'), ('m2', '<f4')])

        res['window_id'] = window
        for name in ('pair_id', 'dt', 'tau', 'sig', 'm1', 'm2'):
            res[name] = data[name][data['window_id']==window][0]

        res['offset'] = dt * off
        res['correlation'] = corr

        if output is None:
            output = res
        else:
            output = np.append(output, res)
    return output

argv = sys.argv[1:]

args = []
kws = {}
for arg in argv:
    if arg.startswith('--'):
        arg = arg[2:]
        if '=' in arg:
            key, value = arg.split('=', 1)
            kws[key] = value
        else:
            kws[arg] = True
    else:
        args.append(arg)

if len(args) < 3 or 'help' in kws:
    print """Usage: python timeshift.py OPTIONS METHOD INPUT.npz OUTPUT.npz
Available options:
    --bucket BUCKET_NR
    --buckets TOTAL_NR_OF_BUCKETS
    --detrend
    --negative

If buckets are specified, divide the data into that many buckets, and
only process the specified bucket.

Available methods
    mse
    correlate
    pearson

Example:

python timeshift.py \
  --detrend \
  mse \
  TimeDelayData/gp_resampled_of_windows_with_truth.npz \
  TimeDelayData/timeshift_mse_normalized_detrend.measures.npz
"""
    sys.exit(-1)

method = args[0]
input = args[1]
output = args[2]

bucket = None
buckets = None
if 'bucket' in kws:
    bucket = int(kws['bucket'])
if 'buckets' in kws:
    buckets = int(kws['buckets'])

data = np.load(input)['arr_0']

windows = None
if buckets is not None:
    windows = np.unique(data['window_id'])

    bucketlen = math.ceil(len(windows) / float(buckets))
    start_win = bucket * bucketlen
    end_win = (bucket+1)*bucketlen
    windows = windows[start_win:end_win]
    print "Working on subset: %s - %s (%s total)" % (windows[0], windows[-1], end_win - start_win)

method = {'mse': timeshift_mse,
          'correlate': timeshift_correlate,
          'pearson': timeshift_pearson}[method]

data = timeshift(
    data,
    method,
    windows=windows,
    detrend=kws.get('detrend', False),
    negative = kws.get('negative', False))

np.savez(output, data)
