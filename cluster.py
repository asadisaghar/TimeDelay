import numpy as np; import matplotlib.pyplot as plt
import sklearn.linear_model
from numpy.lib.recfunctions import *
import sys

x = np.load(sys.argv[1])['arr_0']

dts = np.unique(x['dt'])
res = np.zeros(len(dts), dtype=[(name, 'f4') for name in ['dt', 'est_dt_mean', 'est_dt_median', 'est_dt_std']])
res['dt'] = dts

def findcluster(x):
    res = np.zeros(len(x))
    for i in xrange(0, len(x)):
        res[i] = (1 / (1 + (x[i] - x)**2)).sum()
    return x[np.argmax(res)]

def clusters_for_column(res, col):
    for i in xrange(0, len(dts)):
        res[col][i] = findcluster(x[x['dt'] == dts[i]][col])

clusters_for_column(res, 'est_dt_mean')
clusters_for_column(res, 'est_dt_median')
clusters_for_column(res, 'est_dt_std')

np.savez(sys.argv[2], res)
